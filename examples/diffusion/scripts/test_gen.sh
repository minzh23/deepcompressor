CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.1-dev.yaml configs/svdquant/nvfp4.yaml \
    --load-from /data/zihan/dev_nvfp4 \
    --pipeline-name flux.1-dev \
    --pipeline-device cuda \
    --eval-num-samples 5 \
    --eval-chunk-start 0 \
    --eval-chunk-step 1 \
    --output-root eval_results \
    --skip-eval \
    --output-job eval_dev_nvfp4