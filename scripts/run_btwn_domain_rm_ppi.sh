#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 7 ]; then
    echo "Usage: $0 <dataset_idx> <pretrain_model> <ppi_type> <batch_size> <learning_rate> <max_epoch> <percent_gold_label> <lambda> <pseudo_label_model>"
    exit 1
fi

# Input parameters
dataset_idx=$1
pretrain_model=$2
ppi_type=$3
batch_size=$4
lr=$5
max_epoch=$6
percent_gold_label=$7
lbda=$8
pseudo_label_model=$9

# Dataset configurations
target_datasets=("esfrankel17/HelpSteer2_binarized_w_weak_preferences_cleaned" "esfrankel17/ChatbotArena55k_binarized_w_weak_preferences_cleaned" "esfrankel17/ChatbotArena55k_binarized_w_weak_preferences_cleaned" "esfrankel17/Nectar_10_pct_subsample_binarized_w_weak_preferences_cleaned")
target_splits=("goodness_score" "winner" "winner" "rank")
dataset=("esfrankel17/UltraFeedback_binarized_w_weak_preferences_cleaned" "esfrankel17/Nectar_binarized_w_weak_preferences_cleaned" "esfrankel17/UltraFeedback_binarized_w_weak_preferences_cleaned" "esfrankel17/ChatbotArena55k_binarized_w_weak_preferences_cleaned")
dataset_splits=("average_rating" "rank" "average_rating" "winner")

# Validation checks
if [ ${#dataset[@]} -ne ${#dataset_splits[@]} ] || [ ${#dataset[@]} -ne ${#target_datasets[@]} ] || [ ${#dataset[@]} -ne ${#target_splits[@]} ]; then
    echo "Error: The length of dataset, dataset_splits, target_datasets, and target_splits must be the same"
    exit 1
fi

format_dataset_name() {
    echo $1 | sed 's/^esfrankel17\///' | sed 's/_binarized_w_weak_preferences_cleaned//'
}

format_pretrain_model_name() {
    echo $1 | sed 's/\//--/'
}

# Get dataset details
dataset=${dataset[${dataset_idx}]}
dataset_split=${dataset_splits[${dataset_idx}]}
target_dataset=${target_datasets[${dataset_idx}]}
target_split=${target_splits[${dataset_idx}]}

formatted_dataset_name=$(format_dataset_name ${dataset})
formatted_target_dataset_name=$(format_dataset_name ${target_dataset})
formatted_pretrain_model_name=$(format_pretrain_model_name ${pretrain_model})

# Print the command, then

echo "deepspeed --module --master_port 12345 openrlhf.cli.train_rm_ppi \
--save_path ./checkpoint/btwn_domain_rm_ppi \
--save_pct 0.1 \
--logging_steps 1 \
--eval_pct 0.05 \
--train_batch_size ${batch_size} \
--micro_train_batch_size 8 \
--pretrain ${pretrain_model} \
--bf16 \
--flash_attn \
--max_epochs ${max_epoch} \
--max_len 2048 \
--zero_stage 1 \
--learning_rate ${lr} \
--dataset ${dataset} \
--train_split ${dataset_split} \
--target_dataset ${target_dataset} \
--target_split ${target_split} \
--apply_chat_template \
--chosen_key chosen \
--rejected_key rejected \
--load_checkpoint \
--packing_samples \
--gradient_checkpointing \
--percent_gold_label ${percent_gold_label} \
--pseudo_label_model ${pseudo_label_model} \
--ppi_train_type ${ppi_type} \
--lbda ${lbda} \
--use_wandb $WANDB_TOKEN \
--wandb_project ppi-rm-btwn-domain-${formatted_dataset_name}-to-${formatted_target_dataset_name}-pretrain-${formatted_pretrain_model_name}-ps-${pseudo_label_model}-pct-gold${percent_gold_label} \
--wandb_run_name ppi_type${ppi_type}-lbda${lbda}-ep${max_epoch}-bs${batch_size}-lr${lr}"

# Instead of creating a file, directly run the command
deepspeed --module openrlhf.cli.train_rm_ppi \
--save_path ./checkpoint/btwn_domain_rm_ppi \
--save_pct 0.1 \
--logging_steps 1 \
--eval_pct 0.05 \
--train_batch_size ${batch_size} \
--micro_train_batch_size 8 \
--pretrain ${pretrain_model} \
--bf16 \
--flash_attn \
--max_epochs ${max_epoch} \
--max_len 2048 \
--zero_stage 1 \
--learning_rate ${lr} \
--dataset ${dataset} \
--train_split ${dataset_split} \
--target_dataset ${target_dataset} \
--target_split ${target_split} \
--apply_chat_template \
--chosen_key chosen \
--rejected_key rejected \
--load_checkpoint \
--packing_samples \
--gradient_checkpointing \
--percent_gold_label ${percent_gold_label} \
--pseudo_label_model ${pseudo_label_model} \
--ppi_train_type ${ppi_type} \
--lbda ${lbda} \
--use_wandb $WANDB_TOKEN \
--wandb_project ppi-rm-btwn-domain-${formatted_dataset_name}-to-${formatted_target_dataset_name}-pretrain-${formatted_pretrain_model_name}-ps-${pseudo_label_model}-pct-gold${percent_gold_label} \
--wandb_run_name ppi_type${ppi_type}-lbda${lbda}-ep${max_epoch}-bs${batch_size}-lr${lr}
