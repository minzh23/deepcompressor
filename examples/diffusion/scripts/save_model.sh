CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.diffusion.ptq \
    configs/model/flux.1-schnell.yaml configs/svdquant/int4.yaml \
    --load-from examples/diffusion/runs/diffusion/flux.1/flux.1-schnell/w.4-x.4-y.16/w.sint4-x.sint4.u-y.bf16/w.v64.bf16-x.v64.bf16-y.tnsr.bf16/smooth.proj-w.static.lowrank/shift-skip.x.[[w]+tan+tn].w.[e+rs+rtp+s+tpi+tpo]-low.r32.i100.e.skip.[rc+tan+tn]-smth.proj.GridSearch.bn2.[AbsMax].lr.skip.[rc+tan+tn]-qdiff.128-t4.g0-s5000.ERROR/run-251004.162157.ERROR/cache \
    --pipeline-name flux.1-schnell \
    --pipeline-device cuda \
    --skip-eval \
    --skip-gen \
    --save-model int4 \