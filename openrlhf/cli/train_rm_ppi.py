import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import PPIRewardDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer import PPIRewardModelTrainer
from openrlhf.utils import get_strategy, get_tokenizer, create_raw_ppi_datasets


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    
    if strategy.is_rank_0():
        strategy.print("*" * 100)
    
    train_dataset, val_dataset, test_dataset = create_raw_ppi_datasets(
        dataset=args.dataset,
        percent_train=args.percent_train,
        percent_val=args.percent_val,
        train_split=args.train_split,
        target_dataset=args.target_dataset,
        target_split=args.target_split,
        percent_gold_label=args.percent_gold_label,
        pseudo_label_model=args.pseudo_label_model,
        strategy=strategy,
        seed=args.seed,
        debug=args.debug
    )
    
    if strategy.is_rank_0():
        strategy.print("*" * 100)
        strategy.print(f"PPI training type: {args.ppi_train_type}.")
    orig_train_dataset_len = len(train_dataset)
    
    assert abs(len(train_dataset.filter(lambda x: x["gold_label_agreement"] == 1)) / orig_train_dataset_len - args.percent_gold_label) < 0.02
    
    if args.ppi_train_type in [0, 3, 4, 5]:
        # no need to filter anything
        strategy.print("No filtering needed.")
        strategy.print(f"Training on {len(train_dataset)}/{orig_train_dataset_len} samples.")
    # training on only the gold labels!
    else:
        assert args.ppi_train_type in [1, 2]
        strategy.print("Training only on the gold labels.")
        # filter only where we are "allowed" to train on the gold labels
        train_dataset = train_dataset.filter(lambda x: x["gold_label_agreement"] == 1)
        # this should be approximately the same as the percent_gold_label
        strategy.print(f"Training on {len(train_dataset)}/{orig_train_dataset_len} samples.")
        
    train_dataset = PPIRewardDataset(
        train_dataset,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )
    val_dataset = PPIRewardDataset(
        val_dataset,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )
    test_dataset = PPIRewardDataset(
        test_dataset,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    val_dataloader = strategy.setup_dataloader(
        val_dataset,
        args.micro_train_batch_size,
        True,
        False,
        val_dataset.packing_collate_fn if args.packing_samples else val_dataset.collate_fn,
    )
    test_dataloader = strategy.setup_dataloader(
        test_dataset,
        args.micro_train_batch_size,
        True,
        False,
        test_dataset.packing_collate_fn if args.packing_samples else test_dataset.collate_fn,
    )
    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    if args.ppi_train_type == 2:
        max_steps = orig_train_dataset_len // args.train_batch_size * args.max_epochs
        # number of epochs is however many needed to reach force_steps
        max_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
    else:
        max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
        max_epochs = args.max_epochs
    strategy.print(f"Training for {max_steps} steps, requires {max_epochs} epochs")
    breakpoint()

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = PPIRewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        ppi_train_type=args.ppi_train_type,
        lbda=args.lbda,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=max_epochs,
        max_steps=max_steps,
        loss=args.loss,
        debug=args.debug,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # Save value_head_prefix
    strategy.print("Save value_head_prefix in config")
    unwrap_model = strategy._unwrap_model(model)
    unwrap_model.config.value_head_prefix = args.value_head_prefix

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # RM training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=512)
    
    # PPI parameters
    parser.add_argument("--percent_train", type=float, default=0.7)
    parser.add_argument("--percent_val", type=float, default=0.15)
    parser.add_argument("--percent_test", type=float, default=0.15)
    
    parser.add_argument("--target_dataset", type=str, default=None)
    parser.add_argument("--target_split", type=str, default=None)
    parser.add_argument("--percent_gold_label", type=float)
    parser.add_argument("--pseudo_label_model", type=str, default=None)
    
    # Different PPI training types
    # 0: training only on the pseudo labels
    # 1: training only on the gold labels
    # 2: training only on the gold labels for same number of gradient steps as training on the pseudo labels
    # 3: training on both the pseudo and gold labels
    # 4: training on the pseudo labels and gold labels with DRPA, switching from training on only the pseudo labels to training on both the pseudo and gold labels with DR loss after fraction lbda of total training steps
    # 5: training on the pseudo labels and gold labels, where we initially train on only the pseudo labels, then switch to the gold labels after fraction lbda of total training steps
    # 6: training on the pseudo labels and gold labels, where we initially train on only the pseudo labels, then switch to the gold labels after fraction lbda of total training steps. Small set of gold labels
    # 7: (time permitting) training only on the high-confidence pseudo labels + gold labels
    parser.add_argument("--ppi_train_type", type=int, choices=[0, 1, 2, 3, 4, 5, 6], required=True)
    parser.add_argument("--lbda", type=float)
    parser.add_argument("--force_steps", type=int, default=-1)
    
    parser.add_argument("--debug", action="store_true", default=False)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()
        
    assert args.percent_train + args.percent_val + args.percent_test == 1

    train(args)