CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.1-dev.yaml \
    --pipeline-device cuda \
    --eval-num-samples 512 \
    --output-root eval_metrics \
    --skip-gen \
    --eval-gen-root images/dev_int4_block_images \
    --eval-ref-root baselines/torch.bfloat16/flux.1-dev/fmeuler50-g3.5