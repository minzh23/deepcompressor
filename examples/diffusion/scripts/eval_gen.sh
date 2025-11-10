for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.1-dev.yaml configs/svdquant/int4_block.yaml configs/svdquant/gptq.yaml \
    --load-from /data/zihan/dev_int4_block \
    --pipeline-name flux.1-dev \
    --pipeline-device cuda \
    --eval-num-samples 512 \
    --eval-chunk-start $i \
    --eval-chunk-step 8 \
    --output-root eval_results \
    --skip-eval \
    --output-job eval_dev_int4_block &
done
wait