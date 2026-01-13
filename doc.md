## Reproduction Commands

### on rocm
```
cd /root/sglang
export PYTHONPATH=/root/sglang/python
export CUDA_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

python script.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --lora1 reverentelusarca/elusarca-anime-style-lora-z-image-turbo \
  --lora2 tarn59/pixel_art_style_lora_z_image_turbo \
  --prompt "Doraemon is eating dorayaki" \
  --size 1024x1024 \
  --output outputs/AMD.json \
  --warmup \
  --warmup-resolutions 1024x1024 \
  --profile-target multi    # baseline/single/multi/all
```

### on cuda
```
cd /root/sglang
export PYTHONPATH=/root/sglang/python
export CUDA_VISIBLE_DEVICES=0

python script.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --lora1 reverentelusarca/elusarca-anime-style-lora-z-image-turbo \
  --lora2 tarn59/pixel_art_style_lora_z_image_turbo \
  --prompt "Doraemon is eating dorayaki" \
  --size 1024x1024 \
  --output outputs/Nvidia.json \
  --warmup \
  --warmup-resolutions 1024x1024 \
  --profile-target multi    # baseline/single/multi/all
```
Get trace file in `./logs` and performance summary in `./outputs`

### Result  
**Metadata**  
- model: Tongyi-MAI/Z-Image-Turbo
- size: 1024x1024
- prompt: "Doraemon is eating dorayaki"

**Baseline**  
| metric (ms) | AMD (MI350) | Nvidia (B200) |
| --- | --- | --- |
| e2e | 837.88 | 541.61 |
| denoise | 777.92 | 419.79 |
| text_encode | 53.76 | 102.06 |
| decode | 4.11 | 17.46 |

**Single LoRA**  
| metric (ms) | AMD (MI350) | Nvidia (B200) |
| --- | --- | --- |
| e2e | 842.52 | 536.66 |
| denoise | 780.88 | 415.40 |
| text_encode | 54.53 | 101.82 |
| decode | 4.77 | 17.20 |

**Multi LoRA**  
| metric (ms) | AMD (MI350) | Nvidia (B200) |
| --- | --- | --- |
| e2e | 843.34 | 534.01 |
| denoise | 781.69 | 413.38 |
| text_encode | 54.39 | 101.78 |
| decode | 5.03 | 16.81 |
