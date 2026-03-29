python -m deepcompressor.app.diffusion.ptq \
    configs/model/sdxl.yaml configs/svdquant/nvfp4_block.yaml configs/svdquant/gptq.yaml \
    --skip-eval --skip-gen --save-model ~/deepcompressor/sdxl_nvfp4_block
