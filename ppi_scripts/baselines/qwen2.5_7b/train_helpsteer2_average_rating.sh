set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --logging_steps 1 \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --bf16 \
   --max_epochs 3 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-5 \
   --dataset esfrankel17/original_HelpSteer2_binarized \
   --train_split average_rating \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --use_wandb $WANDB_TOKEN \
   --wandb_project ppi-rm-training-test \
   --wandb_run_name TEST_RUN_baseline-qwen2.5-7b-helpsteer2-average_rating-rm-$(date +%Y%m%d-%H%M%S) \
   --save_ckpt_pct 0.2 \
   --save_model_pct 0.25 \
   --eval_pct 0.25
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
