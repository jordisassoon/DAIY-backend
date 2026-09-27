#!/bin/bash

device="cuda:0"
ckpt_dir="./ckpt/thumos_internvideo2_6b_tridet_l"
data_dir="./data/thumos/internvideo2"
annotations_file="./data/thumos/annotations/thumos14.json"
model_config="./configs/model/large_model.yaml"
dataset_config="./configs/dataset/base_data.yaml"

yaml_template="./configs/model/large_model.yaml"
temp_yaml="./configs/model/temp_model.yaml"

# Hyperparameter search space
declare -a learning_rates=(0.0001 0.00025 0.005)
declare -a model_sizes=(512 576 640)
declare -a k_values=(2 3 4)
declare -a sgp_mlp_dims=(768 1024 1280)
declare -a iou_weight_power=(0.2 0.5 0.8)

best_map=0
best_params=""

for lr in "${learning_rates[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        for k in "${k_values[@]}"; do
            for sgp_mlp_dim in "${sgp_mlp_dims[@]}"; do
                # Modify YAML file
                cp "$yaml_template" "$temp_yaml"
                sed -i "s/learning_rate: .*/learning_rate: $lr,/" "$temp_yaml"
                sed -i "s/embd_dim: .*/embd_dim: $model_size,/" "$temp_yaml"
                sed -i "s/head_dim: .*/head_dim: $model_size,/" "$temp_yaml"
                sed -i "s/fpn_dim: .*/fpn_dim: $model_size,/" "$temp_yaml"
                sed -i '/^[[:space:]]*k:/ s/k: .*/k: '"$k",'/' "$temp_yaml"
                sed -i "s/sgp_mlp_dim: .*/sgp_mlp_dim: $sgp_mlp_dim,/" "$temp_yaml"
                
                run_name="lr_${lr}_model_${model_size}_k_${k}_sgp_${sgp_mlp_dim}"
                save_dir="$ckpt_dir/$run_name"
                mkdir -p "$save_dir"
                
                echo "Training with lr=$lr, model_size=$model_size, k=$k, sgp_mlp_dim=$sgp_mlp_dim"
                python train.py \
                    --model_config "$temp_yaml" \
                    --dataset_config "$dataset_config" \
                    --save_dir "$save_dir" \
                    --data_dir "$data_dir" \
                    --annotations_file "$annotations_file" \
                    --validation_step 0 \
                    --ckpt_freq 0 \
                    --device "$device" \
                    --epochs 10 \
                    --batch_size 2
                
                # Find the latest checkpoint
                latest_ckpt=$(ls -t "$save_dir"/*.pth.tar | head -n 1)
                
                if [[ -f "$latest_ckpt" ]]; then
                    echo "Evaluating checkpoint: $latest_ckpt"
                    map=$(python eval.py \
                        --training_config "$save_dir/config.yaml" \
                        --data_dir "$data_dir" \
                        --ckpt_path "$latest_ckpt" \
                        --device "$device" | grep -oP '(?<=mAP: )[0-9]+\.[0-9]+')
                    
                    if (( $(echo "$map > $best_map" | bc -l) )); then
                        best_map=$map
                        best_params="lr=$lr, model_size=$model_size, k=$k, sgp_mlp_dim=$sgp_mlp_dim"
                    fi
                    echo "Best mAP: $best_map"
                    echo "Best parameters: $best_params"
                else
                    echo "No valid checkpoint found for $run_name"
                fi
            done
        done
    done
done

echo "Best mAP: $best_map with params: $best_params"