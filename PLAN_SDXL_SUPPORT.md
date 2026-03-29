# SDXL NVFp4 Quantization Support Plan

## 1. 现状分析

### 1.1 已有的 SDXL 支持

代码层面已经有部分 SDXL 基础设施:

| 组件 | 文件 | 状态 |
|------|------|------|
| Pipeline 加载 | `pipeline/config.py:331-334` | 已支持，`name="sdxl"` 映射到 `stabilityai/stable-diffusion-xl-base-1.0` |
| UNet 结构解析 | `nn/struct.py:1267-1516` | `UNetStruct` 已完整处理 `UNet2DConditionModel` |
| Block 解析 | `nn/struct.py:1057-1263` | `UNetBlockStruct` 已处理 `CrossAttnDownBlock2D`, `UNetMidBlock2DCrossAttn`, `CrossAttnUpBlock2D` |
| Pipeline 类型 | `nn/struct.py:106` | `UNET_PIPELINE_CLS` 已包含 `StableDiffusionXLPipeline` |
| 模型提取 | `pipeline/config.py:363` | `pipeline.unet if hasattr(pipeline, "unet")` 自动适配 |
| Text encoder 提取 | `pipeline/config.py:371-393` | 通用逻辑，自动发现 `text_encoder` 和 `text_encoder_2` |

**缺失的**: 只缺 SDXL 模型配置文件 (`configs/model/sdxl.yaml`) 和对应的运行脚本。

### 1.2 SDXL vs Flux 架构对比

| 方面 | SDXL | Flux |
|------|------|------|
| 骨干网络 | `UNet2DConditionModel` | `FluxTransformer2DModel` |
| 块结构 | down/mid/up blocks (ResNet + CrossAttn) | transformer_blocks + single_transformer_blocks |
| Text Encoder | 2 个 (CLIP-L + OpenCLIP-ViT-bigG) | 1 个 (T5) |
| dtype | float16 | bfloat16 |
| 输入嵌入 | `conv_in` (Conv2d) | `x_embedder` (Linear) |
| 输出 | `conv_norm_out` + `conv_out` (Conv2d) | `proj_out` (Linear) |
| 时间嵌入 | `time_embedding` (Linear layers) | `CombinedTimestepTextProjEmbeddings` |
| 额外嵌入 | `add_embedding` (用于 SDXL 的 micro-conditions) | 无 |
| 采样层 | downsample/upsample (Conv2d) | 无 |
| Skip Connections | 有 (down->up residuals) | 无 |

### 1.3 SDXL UNet 中的 Layer 类型与对应 Key

框架通过 `named_key_modules()` 遍历所有 `nn.Linear` 和 `nn.Conv2d` 层，并为每个层分配一个 key，通过 skip 列表决定是否量化。

#### Conv2d 层 (不量化，skip)

| Key | 来源 | 说明 |
|-----|------|------|
| `embed` | `conv_in`, `conv_norm_out`, `conv_out`, `time_embedding`, `add_embedding` | 所有 embedding 和输出层 |
| `down_resblock_conv` / `mid_resblock_conv` / `up_resblock_conv` | ResnetBlock2D 内的 conv1, conv2 | ResNet 块的卷积 |
| `down_resblock_shortcut` / `mid_resblock_shortcut` / `up_resblock_shortcut` | ResnetBlock2D 的 skip connection | 残差捷径卷积 |
| `down_sample` | CrossAttnDownBlock2D 的 downsampler | 下采样 Conv2d |
| `up_sample` | CrossAttnUpBlock2D 的 upsampler | 上采样 Conv2d |

#### Linear 层 — skip (与 Flux 对齐)

| Key | 来源 | 说明 |
|-----|------|------|
| `down_resblock_time_proj` / `mid_...` / `up_...` | ResnetBlock2D 的 `time_emb_proj` | 时间嵌入投影，与 Flux 对齐 skip |

#### Linear 层 — 量化目标

| Key | 来源 | 说明 |
|-----|------|------|
| `transformer_proj_in` | `Transformer2DModel.proj_in` | Transformer 输入投影 (nn.Linear)，**SDXL 特有** |
| `transformer_proj_out` | `Transformer2DModel.proj_out` | Transformer 输出投影 (nn.Linear)，**SDXL 特有** |
| `down_transformer_qkv_proj` / `mid_...` / `up_...` | Attention 的 to_q, to_k, to_v | 自注意力 QKV |
| `down_transformer_out_proj` / `mid_...` / `up_...` | Attention 的 to_out | 注意力输出投影 |
| `down_transformer_add_qkv_proj` / `mid_...` / `up_...` | Cross-Attention 的 to_q, to_k, to_v | 交叉注意力 QKV |
| `down_transformer_add_out_proj` / `mid_...` / `up_...` | Cross-Attention 的 to_out | 交叉注意力输出投影 |
| `down_transformer_ffn_up_proj` / `mid_...` / `up_...` | FeedForward 的 up projection (GEGLU) | FFN 上投影 |
| `down_transformer_ffn_down_proj` / `mid_...` / `up_...` | FeedForward 的 down projection | FFN 下投影 |

#### `transformer_proj_in` / `transformer_proj_out` 在 Flux vs SDXL 中的区别

| | Flux | SDXL |
|---|---|---|
| `transformer_proj_in` | **不存在** (DiTStruct 映射到 `input_embed`) | **存在，是 `nn.Linear`** (`Transformer2DModel.proj_in`) |
| `transformer_proj_out` | **不存在** (DiTStruct 映射到 `output_embed`) | **存在，是 `nn.Linear`** (`Transformer2DModel.proj_out`) |
| Flux 配置 skip 它们 | 保护性 skip，不匹配任何实际层 | — |
| SDXL 配置 | — | **不应 skip**，它们是真实的 Linear 层，应该量化 |

### 1.4 量化方案

**使用 NVFp4** (`nvfp4_block.yaml`) + **GPTQ** (`gptq.yaml`) + **Smooth+SVD** (`__default__.yaml`):

- Weight: `sfp4_e2m1_all`, block group `[16,16]`, scale `sfp8_e4m3_nan`
- Input activation: `sfp4_e2m1_all`, per-channel group `[1,16]`, scale `sfp8_e4m3_nan`
- GPTQ: `damp_percentage=0.01`, `block_size=128`
- Smooth: projection + attention smoothing
- SVD low-rank: rank=32

## 2. 需要的改动

### 2.1 需要新建的文件

#### (1) `examples/diffusion/configs/model/sdxl.yaml`

```yaml
pipeline:
  name: sdxl
  dtype: torch.float16
eval:
  num_steps: 50
  guidance_scale: 3.5
  protocol: fmeuler{num_steps}-g{guidance_scale}
  benchmarks: ["MJHQ"]
quant:
  calib:
    batch_size: 1
  wgts:
    calib_range:
      element_batch_size: 64
      sample_batch_size: 16
      element_size: 512
      sample_size: -1
    low_rank:
      sample_batch_size: 16
      sample_size: -1
    skips:
    - embed                    # conv_in, time_embedding, add_embedding, conv_norm_out, conv_out
    - resblock_conv            # ResNet Conv2d (SDXL 特有)
    - resblock_shortcut        # ResNet skip connection Conv2d
    - resblock_time_proj       # ResNet 时间投影 Linear (与 Flux 对齐 skip)
    - down_sample              # 下采样 Conv2d
    - up_sample                # 上采样 Conv2d
    # 注意: 不 skip transformer_proj_in / transformer_proj_out
    # 它们在 SDXL 中是 nn.Linear，应该被量化
    # (Flux 配置中 skip 它们是因为 Flux 不存在这些层，仅为保护性 skip)
  ipts:
    calib_range:
      element_batch_size: 64
      sample_batch_size: 16
      element_size: 512
      sample_size: -1
    skips:
    - embed
    - resblock_conv
    - resblock_shortcut
    - resblock_time_proj
    - transformer_norm
    - transformer_add_norm
    - down_sample
    - up_sample
  opts:
    calib_range:
      element_batch_size: 64
      sample_batch_size: 16
      element_size: 512
      sample_size: -1
  smooth:
    proj:
      element_batch_size: -1
      sample_batch_size: 16
      element_size: -1
      sample_size: -1
    attn:
      sample_batch_size: 16
      sample_size: -1
```

**与 Flux 配置的关键区别**:
1. `dtype: torch.float16` (Flux 用 bfloat16)
2. skip 列表多了 `resblock_conv` (Flux 纯 transformer 没有 ResNet Conv2d)
3. **不 skip** `transformer_proj_in` / `transformer_proj_out` (Flux skip 它们是因为不存在，SDXL 中它们是真实 Linear 层)

#### (2) `examples/diffusion/scripts/ptq_sdxl.sh`

```bash
python -m deepcompressor.app.diffusion.ptq \
    configs/model/sdxl.yaml configs/svdquant/nvfp4_block.yaml configs/svdquant/gptq.yaml \
    --skip-eval --skip-gen --save-model /data/zihan/sdxl_nvfp4_block
```

### 2.2 需要修改的代码

**不需要修改任何 Python 代码**。原因:

1. **Pipeline 加载**: `_default_build()` 已支持 `name == "sdxl"`
2. **模型结构解析**: `UNetStruct._default_construct()` 已完整处理 `UNet2DConditionModel`
3. **Block 解析**: `UNetBlockStruct._default_construct()` 已处理所有 SDXL block 类型
4. **量化流程**: `ptq()` 函数对 UNet 和 DiT 使用相同的流程
5. **NVFp4 + GPTQ**: 已在框架中实现，通过配置文件组合使用

### 2.3 潜在风险点

| 风险 | 说明 | 应对 |
|------|------|------|
| GEGLU activation | SDXL FFN 用 GEGLU (gated)，框架已支持 (`struct.py:12` import) | `FeedForwardStruct` 已处理 GEGLU，应无问题 |
| Cross-attention | SDXL 用标准 cross-attention (非 Flux 的 joint attention) | `DiffusionAttentionStruct` 已区分 self/cross/joint，应无问题 |
| Eval protocol | 与 Flux 对齐用 `fmeuler`，SDXL 实际常用 `euler` | 先对齐，如 eval 报错再调整 |

## 3. 已确认的决策

| 问题 | 决策 | 原因 |
|------|------|------|
| `resblock_time_proj` | skip | 与 Flux 对齐 |
| `transformer_proj_in/out` | **不 skip，量化** | SDXL 中是真实 Linear 层；Flux skip 它们仅因不存在 |
| Eval benchmark | MJHQ | 与 Flux 对齐 |
| Weight + Activation quant | 都做 | 需求明确 |
| `shift_activations` | 不开启 | SDXL 不需要 |
| `add_embedding` (embed key) | skip | 与 Flux 对齐，embedding 不量化 |
| 量化格式 | NVFp4 + GPTQ | 使用 `nvfp4_block.yaml` + `gptq.yaml` |

## 4. 执行步骤

```
Step 1: 创建 examples/diffusion/configs/model/sdxl.yaml
Step 2: 创建 examples/diffusion/scripts/ptq_sdxl.sh
Step 3: 小规模验证 (dry run)
        - 加载 SDXL pipeline
        - 确认 UNetStruct 正确解析所有 block
        - 打印 named_key_modules 输出，验证 key 与 skip 匹配
        - 确认 transformer_proj_in/out 被选中量化
        - 确认所有 Conv2d 层被 skip
Step 4: 运行完整 nvfp4 + GPTQ 量化流程
Step 5: 评估量化后模型质量
```

## 5. 总结

### 需要新建的文件

| 文件 | 说明 |
|------|------|
| `examples/diffusion/configs/model/sdxl.yaml` | SDXL 模型配置 |
| `examples/diffusion/scripts/ptq_sdxl.sh` | 运行脚本 (nvfp4_block + gptq) |

### 不需要修改代码

框架已完整支持 SDXL 的 UNet 架构，只需配置文件。

### 量化目标层 (全部为 nn.Linear)

- **Transformer 投影**: `transformer_proj_in`, `transformer_proj_out`
- **Attention**: `qkv_proj`, `out_proj`, `add_qkv_proj`, `add_out_proj`
- **FFN**: `ffn_up_proj`, `ffn_down_proj`

### Skip 的层

- **Conv2d**: `resblock_conv`, `resblock_shortcut`, `down_sample`, `up_sample`, `embed` (conv_in/conv_out)
- **Embedding**: `embed` (time_embedding, add_embedding)
- **ResNet Linear**: `resblock_time_proj`
