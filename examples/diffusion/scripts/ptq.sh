python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.1-dev.yaml configs/svdquant/int4_block.yaml configs/svdquant/gptq.yaml \
    --skip-eval --skip-gen --save-model ~/deepcompressor/dev_int4_block