#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack random Linear layer tensors for Nunchaku w4x4y16 (NVFP4) testing",
    )
    parser.add_argument("--oc", type=int, default=256, help="Output channels (rows) of Linear weight")
    parser.add_argument("--ic", type=int, default=1024, help="Input channels (cols) of Linear weight; must be divisible by 16")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank; set 0 to disable LoRA")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Data type for inputs (weight/scale/etc.)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to allocate tensors on, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path.cwd() / "nunchaku_linear_w4x4y16_nvfp4.pt"),
        help="Output file path (.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure project root is importable when run directly
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from deepcompressor.backend.nunchaku.utils import (
        convert_to_nunchaku_w4x4y16_linear_weight,
    )

    assert args.ic % 16 == 0, "ic must be divisible by 16 for subscale grouping"
    assert args.oc % 16 == 0, "oc must be divisible by 16 for block-wise subscale along N"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device(args.device)
    print(f"dtype: {dtype}")
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    # Random tensors
    weight = torch.randn(args.oc, args.ic, generator=g, device=device, dtype=dtype) * 0.5

    # Per-tensor scale (single element); positive to avoid sign flips in division
    scale = torch.rand(1, generator=g, device=device, dtype=dtype) * 1.5 + 0.5  # in (0.5, 2.0]

    # Subscale (block-wise):
    #   - First create base blocks with shape [oc/16, 1, ic/16, 1]
    #   - Forward subscale repeats each block 16x along N -> [oc, 1, ic/16, 1]
    #   - Backward subscale uses transposed base blocks (swap N-blocks and K-groups)
    #     then repeat 16x along N -> [ic, 1, oc/16, 1]
    subgroup_size = 16
    blocks_n = args.oc // subgroup_size
    groups_k = args.ic // subgroup_size
    subscale_base_blocks = (
        torch.rand(blocks_n, 1, groups_k, 1, generator=g, device=device, dtype=dtype) * 1.5 + 0.5
    )
    # forward subscale
    subscale = subscale_base_blocks.repeat_interleave(subgroup_size, dim=0)

    # Bias and smooth vectors
    bias = torch.randn(args.oc, generator=g, device=device, dtype=dtype) * 0.1
    smooth = torch.rand(args.ic, generator=g, device=device, dtype=dtype) * 0.2 + 0.9  # near-1 (forward, len=ic)

    # LoRA weights (optional)
    lora = None
    if args.rank and args.rank > 0:
        lora_down = torch.randn(args.rank, args.ic, generator=g, device=device, dtype=dtype) * 0.02
        lora_up = torch.randn(args.oc, args.rank, generator=g, device=device, dtype=dtype) * 0.02
        lora = (lora_down, lora_up)

    # Pack FORWARD with float_point=True (NVFP4)
    weight_packed, scale_packed, bias_packed, smooth_packed, lora_packed, subscale_packed = (
        convert_to_nunchaku_w4x4y16_linear_weight(
            weight=weight,
            scale=scale,
            bias=bias,
            smooth=smooth,
            lora=lora,
            float_point=True,
            subscale=subscale,
        )
    )

    # Prepare BACKWARD (transpose) variants and pack
    weight_T = weight.transpose(0, 1).contiguous()
    # transpose base blocks: [blocks_n,1,groups_k,1] -> [groups_k,1,blocks_n,1]
    subscale_bw_base_blocks = subscale_base_blocks.permute(2, 1, 0, 3).contiguous()
    subscale_bw = subscale_bw_base_blocks.repeat_interleave(subgroup_size, dim=0)

    # prepare LoRA for backward if available: W^T = Down^T @ Up^T
    lora_bw_in = None
    if lora is not None:
        lora_up_bw = lora[1].transpose(0, 1).contiguous()  # (rank, oc)
        lora_down_bw = lora[0].transpose(0, 1).contiguous()    # (ic, rank)
        lora_bw_in = (lora_up_bw, lora_down_bw)

    weight_bw, scale_bw, bias_bw, smooth_bw_packed, lora_bw_packed, subscale_bw_packed = (
        convert_to_nunchaku_w4x4y16_linear_weight(
            weight=weight_T,
            scale=scale,              # per-tensor; reuse
            bias=None,                # bias not used for backward here
            smooth=smooth,         # use backward smooth
            lora=lora_bw_in,          # pack LoRA for backward
            float_point=True,
            subscale=subscale_bw,
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "weight": weight_packed,
        "scale": scale_packed,
        "bias": bias_packed,
        "smooth": smooth_packed,
        "subscale": subscale_packed,
        "lora_down": None if lora_packed is None else lora_packed[0],
        "lora_up": None if lora_packed is None else lora_packed[1],
        # forward originals (unpacked)
        "orig_weight": weight,
        "orig_scale": scale,
        "orig_bias": bias,
        "orig_smooth": smooth,
        "orig_subscale_base_blocks": subscale_base_blocks,
        "orig_subscale_forward": subscale,
        "orig_lora_down": None if lora is None else lora[0],
        "orig_lora_up": None if lora is None else lora[1],
        # backward packed (using transposed weight and transposed block-wise subscale)
        "weight_bw": weight_bw,
        "scale_bw": scale_bw,
        "bias_bw": bias_bw,
        "smooth_bw": smooth_bw_packed,
        "subscale_bw": subscale_bw_packed,
        "lora_up_bw": None if lora_bw_packed is None else lora_bw_packed[0],
        "lora_down_bw": None if lora_bw_packed is None else lora_bw_packed[1],
        # backward originals (unpacked helpers)
        "orig_weight_T": weight_T,
        "orig_subscale_bw": subscale_bw,
        "orig_smooth_bw": smooth,
        "orig_lora_up_T": None if lora is None else lora[1].transpose(0, 1).contiguous(),
        "orig_lora_down_T": None if lora is None else lora[0].transpose(0, 1).contiguous(),
        "meta": {
            "format": "nunchaku_w4x4y16_nvfp4",
            "oc": args.oc,
            "ic": args.ic,
            "rank": args.rank,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "scale_per_tensor": True,
            "subscale_group_size": subgroup_size,
            "subscale_blockwise_forward": True,
            "subscale_blockwise_backward": True,
        },
    }

    torch.save(save_dict, str(out_path))
    print(f"Saved packed tensors to: {out_path}")


if __name__ == "__main__":
    main()


