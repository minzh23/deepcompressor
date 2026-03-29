#!/usr/bin/env python

import safetensors.torch
import os
import json

def prompt_file(prompt, default):
    path = input(f"{prompt} [{default}]: ").strip()
    return path if path else default

# Prompt for input/output files
quant_file = prompt_file("Enter path to quantized blocks", "attention_blocks.safetensors")
unquant_file = prompt_file("Enter path to unquantized layers", "unquantized_layers.safetensors")
output_file = prompt_file("Enter path to save combined model", "combined.safetensors")
metadata_file = prompt_file("Enter path to metadata.json (optional)", "metadata.json")

# Validate file existence
for f in [quant_file, unquant_file]:
    if not os.path.isfile(f):
        raise FileNotFoundError(f"File not found: {f}")

# Load state dicts
quantized_blocks = safetensors.torch.load_file(quant_file)
unquantized_layers = safetensors.torch.load_file(unquant_file)

# Warn about key overlaps
overlap = set(quantized_blocks) & set(unquantized_layers)
if overlap:
    print(f"Warning: Overlapping keys (unquantized will override): {overlap}")

# Merge state dicts
combined_state_dict = {**quantized_blocks, **unquantized_layers}

# Attempt to load metadata.json
metadata = {}
if os.path.isfile(metadata_file):
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # Convert nested objects to JSON strings
        for k, v in metadata.items():
            if isinstance(v, dict):
                metadata[k] = json.dumps(v)
    except Exception as e:
        print(f"Failed to load metadata from {metadata_file}: {e}")
        print("Falling back to hardcoded metadata...")
        metadata = {}

# Hardcoded SDXL metadata fallback
if not metadata:
    metadata = {
        "model_class": "NunchakuSDXLUNet2DConditionModel",
        "config": json.dumps({
            "_class_name": "UNet2DConditionModel",
            "_diffusers_version": "0.19.0.dev0",
            "act_fn": "silu",
            "addition_embed_type": "text_time",
            "addition_embed_type_num_heads": 64,
            "addition_time_embed_dim": 256,
            "attention_head_dim": [5, 10, 20],
            "block_out_channels": [320, 640, 1280],
            "center_input_sample": False,
            "class_embed_type": None,
            "class_embeddings_concat": False,
            "conv_in_kernel": 3,
            "conv_out_kernel": 3,
            "cross_attention_dim": 2048,
            "cross_attention_norm": None,
            "down_block_types": [
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D"
            ],
            "downsample_padding": 1,
            "dual_cross_attention": False,
            "encoder_hid_dim": None,
            "encoder_hid_dim_type": None,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_only_cross_attention": None,
            "mid_block_scale_factor": 1,
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "num_attention_heads": None,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "out_channels": 4,
            "projection_class_embeddings_input_dim": 2816,
            "resnet_out_scale_factor": 1.0,
            "resnet_skip_time_act": False,
            "resnet_time_scale_shift": "default",
            "sample_size": 128,
            "time_cond_proj_dim": None,
            "time_embedding_act_fn": None,
            "time_embedding_dim": None,
            "time_embedding_type": "positional",
            "timestep_post_act": None,
            "transformer_layers_per_block": [1, 2, 10],
            "up_block_types": [
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D"
            ],
            "upcast_attention": None,
            "use_linear_projection": True
        }),
        "quantization_config": json.dumps({
            "method": "svdquant",
            "weight": {
                "dtype": "fp4_e2m1_all",
                "scale_dtype": [None, "fp8_e4m3_nan"],
                "group_size": 16
            },
            "activation": {
                "dtype": "fp4_e2m1_all",
                "scale_dtype": "fp8_e4m3_nan",
                "group_size": 16
            }
        })
    }

# Save the combined file
safetensors.torch.save_file(combined_state_dict, output_file, metadata=metadata)

print(f"\nCombined model saved to: {output_file}")
print(f"Metadata keys included: {', '.join(metadata.keys())}")
