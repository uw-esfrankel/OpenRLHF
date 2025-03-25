import json
import os
from abc import ABC
import shutil

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from torch import distributed as dist
from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class PPIRewardModelTrainer(ABC):
    """
    Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        ppi_train_type,
        lbda,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        max_steps: int = -1,
        loss="sigmoid",
        debug=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.ppi_train_type = ppi_train_type
        self.lbda = lbda
        self.max_steps = max_steps
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.debug = debug
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            
            # add the args as a file to wandb
            with open("args_for_wandb.json", "w") as f:
                json.dump(strategy.args.__dict__, f)
            wandb.save("args_for_wandb.json")

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("val/global_step")
            wandb.define_metric("val/*", step_metric="val/global_step", step_sync=True)
            wandb.define_metric("test/global_step")
            wandb.define_metric("test/*", step_metric="test/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

        # Add tracking for best models
        self.best_val_loss_model = (float("inf"), "")

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
            
        # get total number of steps
        total_steps = self.epochs * num_update_steps_per_epoch
            
        self.strategy.print(f"Taking a total of {total_steps} steps")
        self.strategy.print(f"Evaluating every {args.eval_steps} steps")
        self.strategy.print(f"Saving every {args.save_steps} steps")
            
        if self.ppi_train_type in [4, 5]:
            # get the number of steps to switch from training on only the pseudo labels to training on both the pseudo and gold labels
            switch_steps = int(total_steps * self.lbda)
            self.strategy.print(f"Switching from training on only the pseudo labels to training on both the pseudo and gold labels after {switch_steps} steps")

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        global_step = step // self.strategy.accumulated_gradient
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        acc_sum = 0
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, pseudo_label_agreement, gold_label_agreement, margin = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, pseudo_label_agreement, gold_label_agreement, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, reject_reward, aux_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )
                                        

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                    
                # Different PPI training types
                # 0: training only on the pseudo labels
                # 1: training only on the gold labels
                # 2: training only on the gold labels for same number of gradient steps as training on the pseudo labels
                # 3: training on both the pseudo and gold labels
                # 4: training on the pseudo labels and gold labels with DRPA, switching from training on only the pseudo labels to training on both the pseudo and gold labels with DR loss after fraction lbda of total training steps
                # 5: training on the pseudo labels and gold labels, where we initially train on only the pseudo labels, then switch to the gold labels after fraction lbda of total training steps
                # 6: training on the pseudo labels and gold labels, where we initially train on only the pseudo labels, then switch to the gold labels after fraction lbda of total training steps. Small set of gold labels
                # 7: training on the pseudo labels and gold labels with DRPA, switching from training on only the gold labels to training on both the pseudo and gold labels with DR loss after fraction lbda of total training steps
                
                assert margin is None, "Margin should be None for PPI training type 0"
                
                # 0: training only on the pseudo labels
                if self.ppi_train_type == 0:
                    preference_loss = self.compute_pseudo_label_loss(chosen_reward, reject_reward, pseudo_label_agreement)                
                # 1: training only on the gold labels
                # 2: training only on the gold labels for same number of gradient steps as training on the pseudo labels
                elif self.ppi_train_type in [1, 2]:
                    # chosen_reward and reject_reward are already the gold labels
                    # 1 has reduced dataset size, so we don't need to do anything
                    # 2 just has more gradient steps, so we don't need to do anything
                    if self.debug:
                        assert torch.all(gold_label_agreement == 1), "gold_label_agreement should be 1"
                    preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                # 3: training on both the pseudo and gold labels
                elif self.ppi_train_type == 3:
                    # when gold_label_agreement is 1, we use those gold labels; when gold_label_agreement is -1, we use the pseudo labels
                    # same_mask is the indices where the gold labels are 1 OR the pseudo labels are 1
                    same_mask = torch.logical_or(gold_label_agreement == 1, pseudo_label_agreement == 1)
                    diff_mask = ~same_mask
                    
                    # Initialize combined tensors for chosen and reject rewards
                    final_chosen = torch.empty_like(chosen_reward)
                    final_reject = torch.empty_like(reject_reward)
                    
                    # Assign values based on masks
                    final_chosen[same_mask] = chosen_reward[same_mask]
                    final_chosen[diff_mask] = reject_reward[diff_mask]
                    
                    final_reject[same_mask] = reject_reward[same_mask]
                    final_reject[diff_mask] = chosen_reward[diff_mask]
                    
                    # final_chosen and final_reject should be non-zero and have different values
                    if self.debug:
                        assert torch.all(final_chosen != 0), "final_chosen should be non-zero"
                        assert torch.all(final_reject != 0), "final_reject should be non-zero"
                        if torch.all(chosen_reward != reject_reward):
                            assert torch.all(final_chosen != final_reject), "final_chosen and final_reject should have different values"
                    
                    preference_loss = self.loss_fn(final_chosen, final_reject)
                    
                    if self.debug:
                        # Identify samples with gold labels
                        gold_label_samples = gold_label_agreement == 1
                        # Use pseudo-labels for the rest
                        pseudo_label_samples = ~gold_label_samples
                        
                        final_chosen2 = torch.empty_like(chosen_reward)
                        final_reject2 = torch.empty_like(reject_reward)

                        # For gold samples, use directly
                        final_chosen2[gold_label_samples] = chosen_reward[gold_label_samples]
                        final_reject2[gold_label_samples] = reject_reward[gold_label_samples]
                        
                        # For pseudo samples, apply pseudo-label logic
                        pseudo_same = pseudo_label_agreement[pseudo_label_samples] == 1
                        pseudo_diff = ~pseudo_same
                        
                        pseudo_chosen = torch.empty_like(final_chosen2[pseudo_label_samples])
                        pseudo_reject = torch.empty_like(final_chosen2[pseudo_label_samples])
                        
                        pseudo_chosen[pseudo_same] = chosen_reward[pseudo_label_samples][pseudo_same]
                        pseudo_chosen[pseudo_diff] = reject_reward[pseudo_label_samples][pseudo_diff]
                        
                        pseudo_reject[pseudo_same] = reject_reward[pseudo_label_samples][pseudo_same]
                        pseudo_reject[pseudo_diff] = chosen_reward[pseudo_label_samples][pseudo_diff]
                        
                        final_chosen2[pseudo_label_samples] = pseudo_chosen
                        final_reject2[pseudo_label_samples] = pseudo_reject
                        
                        preference_loss2 = self.loss_fn(final_chosen2, final_reject2)
                        
                        assert preference_loss2.item() == preference_loss.item(), "preference_loss2 should be the same as preference_loss"
                        assert torch.all(final_chosen2 == final_chosen), "final_chosen2 should be the same as final_chosen"
                        assert torch.all(final_reject2 == final_reject), "final_reject2 should be the same as final_reject"
                # 4: training on the pseudo labels and gold labels with DRPA, switching from training on only the pseudo labels to training on both the pseudo and gold labels with DR loss after fraction lbda of total training steps
                elif self.ppi_train_type == 4:
                    # if we are before the switch step, we only use the pseudo labels
                    if global_step < switch_steps:
                        preference_loss = self.compute_pseudo_label_loss(chosen_reward, reject_reward, pseudo_label_agreement)
                    else:                    
                        # has already taken the mean, this is over all available
                        total_pseudo_label_loss = self.compute_pseudo_label_loss(chosen_reward, reject_reward, pseudo_label_agreement)
                        
                        # when gold_label_agreement is 1, we can "see" the gold labels
                        gold_label_indices = gold_label_agreement == 1
                        
                        # Check if we have any gold labels
                        if gold_label_indices.any():
                            # get the gold label loss on only the gold labels
                            small_gold_label_loss = self.loss_fn(chosen_reward[gold_label_indices], reject_reward[gold_label_indices])
                            # get the pseudo label loss on the same indices
                            small_pseudo_label_loss = self.compute_pseudo_label_loss(chosen_reward[gold_label_indices], reject_reward[gold_label_indices], pseudo_label_agreement[gold_label_indices])
                            
                            # doubly robust preference loss
                            preference_loss = total_pseudo_label_loss - small_pseudo_label_loss + small_gold_label_loss
                        else:
                            # If no gold labels available, just use the total pseudo label loss
                            preference_loss = total_pseudo_label_loss
                    
                    assert not torch.isnan(preference_loss), "Preference loss is nan"
                # 5: training on the pseudo labels and gold labels, where we initially train on only the pseudo labels, then switch to the gold labels after fraction lbda of total training steps
                elif self.ppi_train_type == 5:
                    if global_step < switch_steps:
                        preference_loss = self.compute_pseudo_label_loss(chosen_reward, reject_reward, pseudo_label_agreement)
                    else:
                        preference_loss = self.loss_fn(chosen_reward, reject_reward)
                # 6: training on the pseudo labels and gold labels, where we initially train on only the pseudo labels, then switch to the gold labels after fraction lbda of total training steps
                elif self.ppi_train_type == 6:
                    if global_step < switch_steps:
                        preference_loss = self.compute_pseudo_label_loss(chosen_reward, reject_reward, pseudo_label_agreement)
                    else:
                        gold_label_indices = gold_label_agreement == 1
                        if gold_label_indices.any():
                            preference_loss = self.loss_fn(chosen_reward[gold_label_indices], reject_reward[gold_label_indices])
                        else:
                            preference_loss = 0
                            
                    assert not torch.isnan(preference_loss), "Preference loss is nan"
                # 7: training on the pseudo labels and gold labels with DRPA, switching from training on only the gold labels to training on both the pseudo and gold labels with DR loss after fraction lbda of total training steps
                elif self.ppi_train_type == 7:
                    # if we are before the switch step, we only use the gold labels
                    if global_step < switch_steps:
                        gold_label_indices = gold_label_agreement == 1
                        if gold_label_indices.any():
                            preference_loss = self.loss_fn(chosen_reward[gold_label_indices], reject_reward[gold_label_indices])
                        else:
                            preference_loss = 0
                    else:
                        # has already taken the mean, this is over all available
                        total_pseudo_label_loss = self.compute_pseudo_label_loss(chosen_reward, reject_reward, pseudo_label_agreement)
                        
                        # when gold_label_agreement is 1, we can "see" the gold labels
                        gold_label_indices = gold_label_agreement == 1
                        
                        # Check if we have any gold labels
                        if gold_label_indices.any():
                            # get the gold label loss on only the gold labels
                            small_gold_label_loss = self.loss_fn(chosen_reward[gold_label_indices], reject_reward[gold_label_indices])
                            # get the pseudo label loss on the same indices
                            small_pseudo_label_loss = self.compute_pseudo_label_loss(chosen_reward[gold_label_indices], reject_reward[gold_label_indices], pseudo_label_agreement[gold_label_indices])
                            
                            # doubly robust preference loss
                            preference_loss = total_pseudo_label_loss - small_pseudo_label_loss + small_gold_label_loss
                        else:
                            # If no gold labels available, just use the total pseudo label loss
                            preference_loss = total_pseudo_label_loss
                    
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                                        
                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
                if self.max_steps > 0 and global_step >= self.max_steps:
                    break
            epoch_bar.update()
            
        self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.val_dataloader) > 0:
                self.evaluate(self.val_dataloader, global_step, is_val=True)
            if len(self.test_dataloader) > 0:
                self.evaluate(self.test_dataloader, global_step, is_val=False)
            
        # save ckpt
        if global_step % args.save_steps == 0 or global_step == self.max_steps:
            # Make sure all processes are synchronized before evaluation
            dist.barrier()
            
            val_loss = float("inf")
            val_acc_mean = 0
            val_reward_mean = 0
            val_reward_std = 0
            test_loss = float("inf")
            test_acc_mean = 0
            test_reward_mean = 0
            test_reward_std = 0
            
            # Perform validation if the dataloader exists
            if len(self.val_dataloader) > 0:
                val_loss, val_acc_mean, val_reward_mean, val_reward_std = self.evaluate(self.val_dataloader, global_step, is_val=True)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss_model[0]
            
            # Broadcast the is_best value from rank 0 to all ranks to ensure consistency
            is_best_tensor = torch.tensor([1 if is_best else 0], device=torch.cuda.current_device())
            dist.broadcast(is_best_tensor, src=0)
            is_best = bool(is_best_tensor.item())
            
            # If this is the best model, evaluate on test set (if available)
            if is_best and len(self.test_dataloader) > 0:
                test_loss, test_acc_mean, test_reward_mean, test_reward_std = self.evaluate(self.test_dataloader, global_step, is_val=False)
            
            # Make sure all processes are synchronized before model saving
            dist.barrier()
            
            if is_best:
                old_step = self.best_val_loss_model[1]
                tag = f"global_step{global_step}"
                
                if self.strategy.is_rank_0():
                    self.strategy.print(f"New best val loss {val_loss} at step {global_step}")
                    
                    # Remove old model if it exists
                    if old_step != "":
                        old_path = os.path.join(args.save_path, old_step)
                        if os.path.exists(old_path):
                            shutil.rmtree(old_path)
                            self.strategy.print(f"Removed saved model weights at step {old_step}")
                    
                    # Save value_head_prefix in config
                    self.strategy.print("Save value_head_prefix in config")
                
                # Update best model info
                self.best_val_loss_model = (val_loss, tag)
                
                # Set value_head_prefix in config
                unwrap_model = self.strategy._unwrap_model(self.model)
                unwrap_model.config.value_head_prefix = args.value_head_prefix
                
                # Save the model
                output_dir = os.path.join(args.save_path, tag)
                self.strategy.save_model(self.model, self.tokenizer, output_dir)
                
                # Save metrics (only on rank 0)
                if self.strategy.is_rank_0():
                    metrics_dict = {
                        "val_loss": val_loss,
                        "val_acc_mean": val_acc_mean,
                        "val_reward_mean": val_reward_mean,
                        "val_reward_std": val_reward_std,
                        "test_loss": test_loss,
                        "test_acc_mean": test_acc_mean,
                        "test_reward_mean": test_reward_mean,
                        "test_reward_std": test_reward_std,
                    }
                    
                    with open(os.path.join(output_dir, "train_metrics.json"), "w") as f:
                        json.dump(metrics_dict, f)
            
            # Final barrier to ensure all processes are synchronized before continuing
            dist.barrier()
            

    def evaluate(self, eval_dataloader, steps=0, is_val=False):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc=f"{'val' if is_val else 'test'} evaluation stage of steps {steps}",
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            
            for data in eval_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, _, _, margin = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward, reject_reward, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, _, _, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())

                    chosen_reward, reject_reward, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                loss = self.loss_fn(chosen_reward, reject_reward, margin)

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()

            # Make sure we synchronize before computing metrics
            dist.barrier()
            
            # Calculate local metrics first
            local_acc_mean = acc / eval_dataloader.__len__()
            local_loss_mean = loss_sum / eval_dataloader.__len__()
            
            # Gather and compute global metrics
            rewards = torch.cat(rewards).float()
            
            # Use all_gather to collect rewards from all processes
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # Save mean std to model config
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            if is_val:
                self.strategy.print("Saving val metrics")
            else:
                self.strategy.print("Saving test metrics -- TEST")

            # Calculate global metrics using all_reduce
            bar_dict = {
                "eval_loss": local_loss_mean,
                "acc_mean": local_acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                # Only log histogram on rank 0
                histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
                self.strategy.print("histgram")
                self.strategy.print(histgram)
                
                # Log to wandb or tensorboard
                if self._wandb is not None:
                    logs = {f"{'val' if is_val else 'test'}/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        
        # Make sure all processes finish evaluation
        dist.barrier()
        
        # Reset model to training mode
        self.model.train()
        
        return local_loss_mean, local_acc_mean, reward_mean.item(), reward_std.item()

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens):
        all_values, output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        half_len = len(packed_seq_lens) // 2
        chosen_rewards = all_values[:half_len]
        rejected_rewards = all_values[half_len:]
        aux_loss = output.aux_loss if "aux_loss" in output else []

        return chosen_rewards, rejected_rewards, aux_loss

    def compute_pseudo_label_loss(self, chosen_reward, reject_reward, pseudo_label_agreement):
        """Compute loss using only pseudo labels by swapping rewards based on agreement.
        
        Args:
            chosen_reward (torch.Tensor): Rewards for chosen responses
            reject_reward (torch.Tensor): Rewards for rejected responses 
            pseudo_label_agreement (torch.Tensor): Binary tensor indicating agreement (1) or disagreement (0)
            
        Returns:
            torch.Tensor: Computed loss using pseudo labels
        """
        # Create boolean masks for same/different predictions
        same_mask = pseudo_label_agreement == 1
        diff_mask = ~same_mask

        # Initialize combined tensors for chosen and reject rewards
        final_chosen = torch.empty_like(chosen_reward)
        final_reject = torch.empty_like(reject_reward)

        # Assign values based on masks
        final_chosen[same_mask] = chosen_reward[same_mask]
        final_chosen[diff_mask] = reject_reward[diff_mask]

        final_reject[same_mask] = reject_reward[same_mask]
        final_reject[diff_mask] = chosen_reward[diff_mask]
        
        # final_chosen and final_reject should be non-zero and have different values
        if self.debug:
            assert torch.all(final_chosen != 0), "final_chosen should be non-zero"
            assert torch.all(final_reject != 0), "final_reject should be non-zero"
            if torch.all(chosen_reward != reject_reward):
                assert torch.all(final_chosen != final_reject), "final_chosen and final_reject should have different values"

        return self.loss_fn(final_chosen, final_reject)