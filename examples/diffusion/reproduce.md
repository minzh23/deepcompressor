| Model                | Precision | Method       | FID (↓) | IR (↑) | LPIPS (↓) | PSNR (↑) |
|----------------------|-----------|--------------|---------|--------|-----------|----------|
| FLUX.1-schnell (4 Steps) | BF16      | --           | 19.2    | 0.938  | --        | --       |
|                          | BF16      | **reproduce**| **19.2**| **0.967**| --      | --       |
|                      | INT W4A4  | SVDQ         | 18.3    | 0.957  | 0.289     | 17.6     |
|                      | INT W4A4     | **SVDQ (reproduce)** | **18.8** | **0.936** | **0.288** | **17.6** |
|                      | INT W4A4  | SVDQ (Weight-Block-Wise 32 $\times$ 32)        | 19.0    | 0.880  | 0.393     | 15.7     |
|                      | INT W4A4  | SVDQ (Weight-Block-Wise 16 $\times$ 16)        | 19.0    | 0.872  | 0.343     | 17.0     |
|                      | NVFP4     | SVDQ         | 18.7    | 0.979  | 0.247     | 18.4     |
|                      | NVFP4     | **SVDQ (reproduce)** | **18.8** | **0.970** | **0.247** | **18.4** |
|                      | NVFP4     | SVDQ (Weight-Block-Wise 16 $\times$ 16)        | 18.5    | 0.932  | 0.295     | 17.6     |
|                      | NVFP4     | SVDQ (Weight-Block-Wise 16 $\times$ 16) + GPTQ       | 19.0    | 0.959  | 0.244     | 18.5     |
