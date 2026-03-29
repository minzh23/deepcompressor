CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.diffusion.ptq \
    configs/model/sdxl.yaml \
    --pipeline-name sdxl \
    --pipeline-device cuda \
    --eval-num-samples 5 \
    --eval-chunk-start 0 \
    --eval-chunk-step 1 \
    --output-root eval_results \
    --skip-eval \
    --output-job eval_sdxl_nvfp4_block
