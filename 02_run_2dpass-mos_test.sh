#!/bin/bash

set -e # exit if any command fails

prev_config="prev_config/02_run_2dpass-mos_test.yaml"
is_prev=false

if [ -f "$prev_config" ]; then
    prev_config_path=$(grep "config_path:" $prev_config | awk '{print $2}')
    prev_model_path=$(grep "model_path:" $prev_config | awk '{print $2}')
    prev_tta_num=$(grep "tta_num:" $prev_config | awk '{print $2}')
    prev_ds_type=$(grep "ds_type:" $prev_config | awk '{print $2}')
    
    echo "Previous Configuration:"
    echo " - Config Path: $prev_config_path"
    echo " - Model Path: $prev_model_path"
    echo " - TTA Number: $prev_tta_num"
    echo " - Dataset Type: $prev_ds_type"
    
    read -p "Load previous config? [y/N]: " is_prev_input
    if [[ "$is_prev_input" =~ ^[Yy]$ ]]; then
        is_prev=true
    fi
fi

if [ "$is_prev" = false ]; then
    read -e -p "Enter the config path: " config_path 
    # config/2DPASS-semantickitti-mos-1f.yaml for 1 frame
    # config/2DPASS-semantickitti-mos-2f.yaml for 2 frame
    # config/2DPASS-semantickitti.yaml for semantic segmentation

    read -e -p "Enter the model path: " model_path
    # models/model_2dpass-mos_frames-1_batch-8_epoch-64.ckpt for 1 frame pretrain model
    # models/model_2dpass-mos_frames-2_batch-4_epoch-64.ckpt for 2 frame pretrain model
    # models/model_2dpass-original.ckpt for semantic pretrain model
    # Your own trained models probablly in the logs folder

    read -p "Enter the TTA number: " tta_num
    # TTA is the number of views for the test-time-augmentation. 
    # We set this value to 12 as default, and if you use other GPUs with smaller memory, you can choose a smaller value.
    # 1 denotes there is no TTA used.

    read -p "Enter dataset type [val, test]: " ds_type
    # Select val if you want to save the val predictions to show results
    # Select test if you want to save the test predictions to submit to server
    
    echo "config_path: $config_path" > $prev_config
    echo "model_path: $model_path" >> $prev_config
    echo "tta_num: $tta_num" >> $prev_config
    echo "ds_type: $ds_type" >> $prev_config
else
    config_path=$prev_config_path
    model_path=$prev_model_path
    tta_num=$prev_tta_num
    ds_type=$prev_ds_type
fi

cleanup() {
    if [ "$ds_type" == "val" ]; then
        python swap_test_valid.py
    fi
}

trap cleanup EXIT

if [ "$ds_type" == "val" ]; then
    python swap_test_valid.py
fi

python main.py \
    --config "$config_path" \
    --gpu 0 \
    --test \
    --submit_to_server \
    --num_vote $tta_num \
    --checkpoint "$model_path"

