set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/baseline-qwen2.5-7b-helpsteer-average_rating-rm-$(date +%Y%m%d-%H%M%S) \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --bf16 \
   --max_epochs 3 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-5 \
   --dataset esfrankel17/original_HelpSteer_binarized \
   --train_split average_rating \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --use_wandb $WANDB_TOKEN \
   --wandb_org esfrankel17 \
   --wandb_project ppi-rm-training-new \
   --wandb_run_name baseline-qwen2.5-7b-helpsteer-average_rating-rm-$(date +%Y%m%d-%H%M%S)
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
