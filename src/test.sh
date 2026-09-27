#!/bin/bash

start=`date +%s`

python extract.py \
    --data_dir ./data/thumos/videos \
    --save_dir ./data/example \
    --base_model vit_small_patch16_224 \
    --ckpt_path ./ckpt/videomae/vit_s_k710_dl_from_giant.pth \
    --device cuda:0 \
    --subset_path data/example/example_subset.json

python train.py \
    --model_config ./configs/model/base_model.yaml \
    --dataset_config ./configs/dataset/base_data.yaml \
    --save_dir ./ckpt/example/ \
    --data_dir ./data/example/ \
    --validation_step 5 \
    --ckpt_freq 10 \
    --device cuda:0

python eval.py \
    --model_config ./configs/model/base_model.yaml \
    --dataset_config ./configs/dataset/base_data.yaml \
    --data_dir ./data/example/ \
    --ckpt_path ./ckpt/example/epoch_029.pth.tar \
    --device cuda:0

python deploy.py \
    --model_config ./configs/model/base_model.yaml \
    --dataset_config ./configs/dataset/base_data.yaml \
    --ckpt_path ./ckpt/example/epoch_029.pth.tar \
    --data_dir ./data/example \
    --save_dir ./data/example \
    --subset_path data/example/example_subset.json \
    --device cuda:0

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.