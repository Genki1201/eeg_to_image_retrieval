#!/bin/bash

# シェルスクリプトをエラーが発生したら終了する設定
set -e

# 実行コマンド cfgがリアルさを調整
PYTHONWARNINGS='ignore' python3 main_mage.py \
  --pretrained_enc_arch mocov3_vit_large \
  --pretrained_enc_path ./pretrained_enc_ckpts/mocov3/vitl.pth.tar \
  --rep_drop_prob 0.1 \
  --use_rep \
  --rep_dim 256 \
  --pretrained_enc_withproj \
  --pretrained_enc_proj_dim 256 \
  --pretrained_rdm_cfg config/rdm/mocov3vitl_simplemlp_l12_w1536.yaml \
  --pretrained_rdm_ckpt ./pretrained_rdm_ckpts/rdm-mocov3vitb.pth \
  --rdm_steps 250 \
  --eta 1.0 \
  --temp 11.0 \
  --num_iter 20 \
  --num_images 50000 \
  --cfg 0.0 \
  --batch_size 64 \
  --input_size 256 \
  --model mage_vit_large_patch16 \
  --mask_ratio_min 0.5 \
  --mask_ratio_max 1.0 \
  --mask_ratio_mu 0.75 \
  --mask_ratio_std 0.25 \
  --epochs 200 \
  --warmup_epochs 10 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --output_dir output_dir \
  --data_path ./ImageNet_dir \
  --evaluate \
  --resume ./pretrained_mage_ckpts/mage-l.pth