#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 7 ]; then
    echo "Usage: $0 <dataset_idx> <pretrain_model> <ppi_type> <batch_size> <learning_rate> <max_epoch> <percent_gold_label> <lambda> <pseudo_label_model> <beta>"
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
beta=${10}

# Dataset configurations
dataset=("esfrankel17/HelpSteer2_binarized_w_weak_preferences_cleaned" "esfrankel17/ChatbotArena55k_binarized_w_weak_preferences_cleaned" "esfrankel17/UltraFeedback_binarized_w_weak_preferences_cleaned" "esfrankel17/Nectar_binarized_w_weak_preferences_cleaned")
dataset_splits=("goodness_score" "winner" "average_rating" "rank")

format_dataset_name() {
    echo $1 | sed 's/^esfrankel17\///' | sed 's/_binarized_w_weak_preferences_cleaned//'
}

format_pretrain_model_name() {
    echo $1 | sed 's/\//--/'
}

# Get dataset details
dataset=${dataset[${dataset_idx}]}
dataset_split=${dataset_splits[${dataset_idx}]}

formatted_dataset_name=$(format_dataset_name ${dataset})
formatted_pretrain_model_name=$(format_pretrain_model_name ${pretrain_model})

wandb_project=ppi-dpo-in-domain-${formatted_dataset_name}-pretrain-${formatted_pretrain_model_name}-ps-${pseudo_label_model}-pct-gold${percent_gold_label}
wandb_run_name=ppi_type${ppi_type}-lbda${lbda}-ep${max_epoch}-bs${batch_size}-lr${lr}-beta${beta}

if python check_wandb_run.py --project $wandb_project --name $wandb_run_name; then
    echo "Run is finished"
    exit 0
else
    echo "Run is not finished yet"
    # Continue execution to start/resume the run
fi


# Print the command
echo "deepspeed --module openrlhf.cli.train_dpo_ppi \
--save_path ./checkpoint/in_domain_dpo_ppi \
--save_steps 1.0 \
--logging_steps 1 \
--eval_steps 1.0 \
--train_batch_size ${batch_size} \
--micro_train_batch_size 2 \
--pretrain ${pretrain_model} \
--bf16 \
--max_epochs 1 \
--max_len 2048 \
--zero_stage 1 \
--learning_rate ${lr} \
--beta ${beta} \
--dataset ${dataset} \
--train_split ${dataset_split} \
--apply_chat_template \
--chosen_key chosen \
--rejected_key rejected \
--flash_attn \
--use_liger_kernel \
--load_checkpoint \
--gradient_checkpointing \
--packing_samples \
--percent_gold_label ${percent_gold_label} \
--pseudo_label_model ${pseudo_label_model} \
--ppi_train_type ${ppi_type} \
--lbda ${lbda} \
--use_wandb ${WANDB_TOKEN} \
--wandb_project ${wandb_project} \
--wandb_run_name ${wandb_run_name}"

deepspeed --module openrlhf.cli.train_dpo_ppi \
--save_path ./checkpoint/in_domain_dpo_ppi \
--save_pct 1.0 \
--logging_steps 1 \
--eval_pct 0.1 \
--train_batch_size ${batch_size} \
--micro_train_batch_size 4 \
--pretrain ${pretrain_model} \
--bf16 \
--max_epochs 1 \
--max_len 2048 \
--zero_stage 1 \
--learning_rate ${lr} \
--beta ${beta} \
--dataset ${dataset} \
--train_split ${dataset_split} \
--apply_chat_template \
--chosen_key chosen \
--rejected_key rejected \
--flash_attn \
--use_liger_kernel \
--load_checkpoint \
--gradient_checkpointing \
--packing_samples \
--percent_gold_label ${percent_gold_label} \
--pseudo_label_model ${pseudo_label_model} \
--ppi_train_type ${ppi_type} \
--lbda ${lbda} \
--use_wandb ${WANDB_TOKEN} \
--wandb_project ${wandb_project} \
--wandb_run_name ${wandb_run_name}
