"""Converts a DeepCompressor SDXL state dict to a Nunchaku state dict."""

import argparse
import os

import safetensors.torch
import torch
import tqdm

from .convert import (
    convert_to_nunchaku_transformer_block_state_dict,
    convert_to_nunchaku_w4x4y16_linear_state_dict,
    update_state_dict,
)


def convert_to_nunchaku_sdxl_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a single SDXL BasicTransformerBlock to Nunchaku format.

    SDXL BasicTransformerBlock contains:
      - attn1 (self-attention): to_q, to_k, to_v, to_out.0
      - attn2 (cross-attention): to_q, to_k, to_v, to_out.0
      - ff.net.0.proj (GEGLU up), ff.net.2 (down)
      - norm1, norm2, norm3 (LayerNorm, not quantized)
    """
    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            # norm layers (not quantized, will be copied as-is since scale is None)
            "norm1": "norm1",
            "norm2": "norm2",
            "norm3": "norm3",
            # self-attention
            "self_attn_qkv_proj": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
            "self_attn_out_proj": "attn1.to_out.0",
            # cross-attention
            "cross_attn_q_proj": "attn2.to_q",
            "cross_attn_kv_proj": ["attn2.to_k", "attn2.to_v"],
            "cross_attn_out_proj": "attn2.to_out.0",
            # feedforward
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": "ff.net.2",
        },
        smooth_name_map={
            "self_attn_qkv_proj": "attn1.to_q",
            "self_attn_out_proj": "attn1.to_out.0",
            "cross_attn_q_proj": "attn2.to_q",
            "cross_attn_kv_proj": "attn2.to_k",
            "cross_attn_out_proj": "attn2.to_out.0",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": "ff.net.2",
        },
        branch_name_map={
            "self_attn_qkv_proj": "attn1.to_q",
            "self_attn_out_proj": "attn1.to_out.0",
            "cross_attn_q_proj": "attn2.to_q",
            "cross_attn_kv_proj": "attn2.to_k",
            "cross_attn_out_proj": "attn2.to_out.0",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": "ff.net.2",
        },
        convert_map={
            "norm1": "copy",
            "norm2": "copy",
            "norm3": "copy",
            "self_attn_qkv_proj": "linear",
            "self_attn_out_proj": "linear",
            "cross_attn_q_proj": "linear",
            "cross_attn_kv_proj": "linear",
            "cross_attn_out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_sdxl_attention_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    attn_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a single SDXL Transformer2DModel (attentions.X) to Nunchaku format.

    Contains: norm (GroupNorm), proj_in (Linear), proj_out (Linear), transformer_blocks.*
    """
    converted: dict[str, torch.Tensor] = {}

    # Find all transformer_blocks under this attention module
    tb_names = set()
    for key in state_dict:
        if key.startswith(f"{attn_name}.transformer_blocks."):
            parts = key.split(".")
            # Extract "transformer_blocks.N"
            tb_idx_pos = parts.index("transformer_blocks")
            tb_name = ".".join(parts[: tb_idx_pos + 2])
            tb_names.add(tb_name)
    tb_names = sorted(tb_names, key=lambda x: int(x.split(".")[-1]))

    # Convert proj_in
    proj_in_name = f"{attn_name}.proj_in"
    if f"{proj_in_name}.weight" in state_dict:
        scale = scale_dict.get(f"{proj_in_name}.weight.scale.0", None)
        if scale is not None:
            subscale = scale_dict.get(f"{proj_in_name}.weight.scale.1", None)
            smooth = smooth_dict.get(proj_in_name, None)
            branch = branch_dict.get(proj_in_name, None)
            if branch is not None:
                branch = (branch["a.weight"], branch["b.weight"])
            weight = state_dict[f"{proj_in_name}.weight"]
            bias = state_dict.get(f"{proj_in_name}.bias", None)
            print(f"  - Converting {proj_in_name} (float_point={float_point})")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x4y16_linear_state_dict(
                    weight=weight,
                    scale=scale,
                    bias=bias,
                    smooth=smooth,
                    lora=branch,
                    float_point=float_point,
                    subscale=subscale,
                ),
                prefix="proj_in",
            )
        else:
            # Not quantized, copy as-is
            converted["proj_in.weight"] = state_dict[f"{proj_in_name}.weight"].clone().cpu()
            if f"{proj_in_name}.bias" in state_dict:
                converted["proj_in.bias"] = state_dict[f"{proj_in_name}.bias"].clone().cpu()

    # Convert proj_out
    proj_out_name = f"{attn_name}.proj_out"
    if f"{proj_out_name}.weight" in state_dict:
        scale = scale_dict.get(f"{proj_out_name}.weight.scale.0", None)
        if scale is not None:
            subscale = scale_dict.get(f"{proj_out_name}.weight.scale.1", None)
            smooth = smooth_dict.get(proj_out_name, None)
            branch = branch_dict.get(proj_out_name, None)
            if branch is not None:
                branch = (branch["a.weight"], branch["b.weight"])
            weight = state_dict[f"{proj_out_name}.weight"]
            bias = state_dict.get(f"{proj_out_name}.bias", None)
            print(f"  - Converting {proj_out_name} (float_point={float_point})")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x4y16_linear_state_dict(
                    weight=weight,
                    scale=scale,
                    bias=bias,
                    smooth=smooth,
                    lora=branch,
                    float_point=float_point,
                    subscale=subscale,
                ),
                prefix="proj_out",
            )
        else:
            converted["proj_out.weight"] = state_dict[f"{proj_out_name}.weight"].clone().cpu()
            if f"{proj_out_name}.bias" in state_dict:
                converted["proj_out.bias"] = state_dict[f"{proj_out_name}.bias"].clone().cpu()

    # Copy norm (GroupNorm, not quantized)
    norm_name = f"{attn_name}.norm"
    for suffix in ["weight", "bias"]:
        key = f"{norm_name}.{suffix}"
        if key in state_dict:
            converted[f"norm.{suffix}"] = state_dict[key].clone().cpu()

    # Convert each transformer_block
    for tb_name in tb_names:
        tb_local = tb_name[len(attn_name) + 1 :]  # e.g., "transformer_blocks.0"
        tb_converted = convert_to_nunchaku_sdxl_transformer_block_state_dict(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            block_name=tb_name,
            float_point=float_point,
        )
        update_state_dict(converted, tb_converted, prefix=tb_local)

    return converted


def convert_to_nunchaku_sdxl_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert the full SDXL UNet state dict to Nunchaku format.

    Returns:
        converted: quantized attention layers in Nunchaku format
        other: unquantized layers (resnets, conv_in/out, time_embedding, etc.)
    """
    # Find all attention modules (down_blocks.X.attentions.Y, mid_block.attentions.Y, up_blocks.X.attentions.Y)
    attn_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}

    for param_name in state_dict.keys():
        # Check if this key belongs to an attention module
        if ".attentions." in param_name:
            parts = param_name.split(".")
            attn_idx = parts.index("attentions")
            attn_name = ".".join(parts[: attn_idx + 2])  # e.g., "down_blocks.1.attentions.0"
            attn_names.add(attn_name)
        else:
            other[param_name] = state_dict[param_name]

    attn_names = sorted(attn_names)
    print(f"Found {len(attn_names)} attention modules to convert.")
    for name in attn_names:
        print(f"  {name}")

    converted: dict[str, torch.Tensor] = {}
    for attn_name in attn_names:
        print(f"\nConverting {attn_name}...")
        attn_converted = convert_to_nunchaku_sdxl_attention_state_dict(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            attn_name=attn_name,
            float_point=float_point,
        )
        update_state_dict(converted, attn_converted, prefix=attn_name)

    return converted, other


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default="sdxl", help="name of the model.")
    parser.add_argument("--float-point", action="store_true", help="use float-point 4-bit quantization.")
    args = parser.parse_args()

    if not args.output_root:
        args.output_root = args.quant_path

    model_name = args.model_name
    assert model_name, "Model name must be provided."

    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale_blockwise.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")

    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"Loading checkpoints from {args.quant_path}...")
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}

    converted_state_dict, other_state_dict = convert_to_nunchaku_sdxl_state_dicts(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=args.float_point,
    )

    output_dirpath = os.path.join(args.output_root, model_name)
    os.makedirs(output_dirpath, exist_ok=True)
    safetensors.torch.save_file(converted_state_dict, os.path.join(output_dirpath, "attention_blocks.safetensors"))
    safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
    print(f"Quantized model saved to {output_dirpath}.")
