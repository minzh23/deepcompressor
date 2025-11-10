import os, shutil, glob
from tqdm import tqdm

dst_root = "images/dev_int4_block_images/samples/MJHQ/MJHQ-512"
os.makedirs(dst_root, exist_ok=True)

pattern = "/FirstIntelligence/home/zihanm/deepcompressor/examples/diffusion/eval_results/diffusion/flux.1/flux.1-dev/**/eval_dev_int4_block.c[0-7]*/samples/MJHQ/**/*.png"
files = glob.glob(pattern, recursive=True)

for path in tqdm(files, desc="Copying PNGs", unit="file"):
    shutil.copy(path, os.path.join(dst_root, os.path.basename(path)))

print(f"✅ 已收集 {len(files)} 张 PNG 到 {dst_root}")