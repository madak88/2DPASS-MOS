#!/bin/bash

prev_config="prev_config/01_run_2dpass-mos_train.yaml"
is_prev=false

if [ -f "$prev_config" ]; then
    prev_config_path=$(grep "config_path:" $prev_config | awk '{print $2}')
    prev_log_dir=$(grep "log_dir:" $prev_config | awk '{print $2}')
    
    echo "Previous Configuration:"
    echo " - Config Path: $prev_config_path"
    echo " - Log Directory Name: $prev_log_dir"
    
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

    read -e -p "Enter the log directory name: " log_dir
    
    echo "config_path: $config_path" > $prev_config
    echo "log_dir: $log_dir" >> $prev_config
else
    config_path=$prev_config_path
    log_dir=$prev_log_dir
fi

python main.py \
    --log_dir "$log_dir" \
    --config "$config_path" \
    --gpu 0

