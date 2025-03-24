import os
import random

from datasets import interleave_datasets, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def create_raw_ppi_datasets(
    dataset,
    percent_train,
    percent_val,
    train_split,
    target_dataset=None,
    target_split=None,
    percent_gold_label=None,
    pseudo_label_model=None,
    strategy=None,
    seed=42,
    debug=False,
):
    if strategy.is_rank_0():
            strategy.print(f"Pseudo label dataset: {dataset}")
            
    dataset = load_dataset(dataset, split=train_split).shuffle(seed=seed)
    weak_label_column = f"{pseudo_label_model.lower().replace('-', '_')}_agreement"
    dataset = dataset.rename_column(weak_label_column, "pseudo_label_agreement")
    cols_to_drop = [col for col in dataset.column_names if col not in ["prompt", "chosen", "rejected", "pseudo_label_agreement"]]
    
    # "transfer learning" setting
    if target_dataset:
        train_dataset_pseudo_label_split = dataset.remove_columns(cols_to_drop)
        train_dataset_pseudo_label_split = train_dataset_pseudo_label_split.map(lambda x: {"gold_label_agreement": -1})
        
        # remove the rows where the pseudo_label_agreement is null
        original_train_dataset_pseudo_label_split_len = len(train_dataset_pseudo_label_split)
        train_dataset_pseudo_label_split = train_dataset_pseudo_label_split.filter(lambda x: x["pseudo_label_agreement"] is not None)
        if strategy.is_rank_0():
            strategy.print(f"Removed {original_train_dataset_pseudo_label_split_len - len(train_dataset_pseudo_label_split)}/{original_train_dataset_pseudo_label_split_len} rows where {weak_label_column} is null")
        
        if strategy.is_rank_0():
            strategy.print(f"GT target label dataset: {target_dataset}")
        
        target_dataset = load_dataset(target_dataset, split=target_split).shuffle(seed=seed)
        target_dataset = target_dataset.rename_column(weak_label_column, "pseudo_label_agreement").remove_columns(cols_to_drop)
        # remove the rows where the pseudo_label_agreement is null
        original_target_dataset_len = len(target_dataset)
        target_dataset = target_dataset.filter(lambda x: x["pseudo_label_agreement"] is not None)
        if strategy.is_rank_0():
            strategy.print(f"Removed {original_target_dataset_len - len(target_dataset)}/{original_target_dataset_len} rows where {weak_label_column} is null")
        
        # map the gold_label_agreement to 1
        target_dataset = target_dataset.map(lambda x: {"gold_label_agreement": 1})
                
        train_dataset_gold_label_split_end_idx = int(len(target_dataset) * percent_train)
        val_dataset_end_idx = train_dataset_gold_label_split_end_idx + int(len(target_dataset) * percent_val)
        
        train_dataset_gold_label_split = target_dataset.select(range(train_dataset_gold_label_split_end_idx))
        val_dataset = target_dataset.select(range(train_dataset_gold_label_split_end_idx, val_dataset_end_idx))
        test_dataset = target_dataset.select(range(val_dataset_end_idx, len(target_dataset)))
                
        current_gold_ratio = len(train_dataset_gold_label_split) / (len(train_dataset_gold_label_split) + len(train_dataset_pseudo_label_split))
        if strategy.is_rank_0():
            strategy.print(f"Current ratio of gold labels: {current_gold_ratio:.4f} (target: {percent_gold_label:.4f})")
            
        # If there are not enough gold labels in the combined dataset
        if current_gold_ratio < percent_gold_label:
            # We need to remove weak labels to increase the gold ratio
            # Calculate how many weak samples to keep
            total_samples_after = len(train_dataset_gold_label_split) / percent_gold_label
            weak_samples_to_keep = int(total_samples_after - len(train_dataset_gold_label_split))
            
            # Ensure we're not trying to keep more than we have
            weak_samples_to_keep = min(weak_samples_to_keep, len(train_dataset_pseudo_label_split))
            
            if strategy.is_rank_0():
                strategy.print(f"Keeping {weak_samples_to_keep} samples from train_dataset_weak (removing {len(train_dataset_pseudo_label_split) - weak_samples_to_keep})")
            train_dataset_pseudo_label_split = train_dataset_pseudo_label_split.select(range(weak_samples_to_keep))
        # If there are too many gold labels in the combined dataset
        elif current_gold_ratio > percent_gold_label:
            # We need to remove gold labels to decrease the gold ratio
            # Calculate how many gold samples to keep
            total_samples = len(train_dataset_gold_label_split) + len(train_dataset_pseudo_label_split)
            gold_samples_to_keep = int(total_samples * percent_gold_label)
            
            # Ensure we're not trying to keep more than we have
            gold_samples_to_keep = min(gold_samples_to_keep, len(train_dataset_gold_label_split))
            
            if strategy.is_rank_0():
                strategy.print(f"Keeping {gold_samples_to_keep} samples from train_dataset_gold_label_split (removing {len(train_dataset_gold_label_split) - gold_samples_to_keep})")
            train_dataset_gold_label_split = train_dataset_gold_label_split.select(range(gold_samples_to_keep))
            
        # Recalculate and verify the new ratio
        new_total = len(train_dataset_gold_label_split) + len(train_dataset_pseudo_label_split)
        new_gold_ratio = len(train_dataset_gold_label_split) / new_total
        if strategy.is_rank_0():
            strategy.print(f"New ratio of gold labels: {new_gold_ratio:.4f} (target: {percent_gold_label:.4f})")
            
        train_dataset = concatenate_datasets([train_dataset_gold_label_split, train_dataset_pseudo_label_split]).shuffle(seed=seed)                        
    # "within domain" setting
    else: 
        dataset = dataset.remove_columns(cols_to_drop)
        original_dataset_len = len(dataset)
        dataset = dataset.filter(lambda x: x["pseudo_label_agreement"] is not None)
        if strategy.is_rank_0():
            strategy.print(f"Removed {original_dataset_len - len(dataset)}/{original_dataset_len} rows where {weak_label_column} is null")
        
        # Split dataset into train, val, and test
        train_end_idx = int(len(dataset) * percent_train)
        val_end_idx = train_end_idx + int(len(dataset) * percent_val)
        
        train_dataset = dataset.select(range(train_end_idx))
        val_dataset = dataset.select(range(train_end_idx, val_end_idx))
        test_dataset = dataset.select(range(val_end_idx, len(dataset)))
        
        if strategy.is_rank_0() and debug:
            strategy.print(f"Dataset splits: train[0:{train_end_idx}], val[{train_end_idx}:{val_end_idx}], test[{val_end_idx}:]")
            
        # randomly select percent_gold_label indices from train_dataset
        gold_label_indices_train = random.sample(range(len(train_dataset)), int(len(train_dataset) * percent_gold_label))
        if strategy.is_rank_0() and debug:
            strategy.print(f"Selected {len(gold_label_indices_train)}/{len(train_dataset)} = {len(gold_label_indices_train) / len(train_dataset):.4f} gold label indices: {gold_label_indices_train}")
        
        # -1 means we don't "know" the gold label in this case
        train_dataset = train_dataset.map(lambda x, idx: {
            "gold_label_agreement": 1 if idx in gold_label_indices_train else -1
        }, with_indices=True)
        # for val and test, gold_label_agreement is 1. Doesn't really matter
        val_dataset = val_dataset.map(lambda x: {"gold_label_agreement": 1})
        test_dataset = test_dataset.map(lambda x: {"gold_label_agreement": 1})
                
    if debug:
        for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
            assert dataset.filter(lambda x: x['pseudo_label_agreement'] is None).num_rows == 0, f"There are {dataset.filter(lambda x: x['pseudo_label_agreement'] is None).num_rows} in {dataset} where pseudo_label_agreement is None"
            assert dataset.filter(lambda x: x['gold_label_agreement'] is None).num_rows == 0, f"There are {dataset.filter(lambda x: x['gold_label_agreement'] is None).num_rows} in {dataset} where gold_label_agreement is None"
                
            assert dataset.filter(lambda x: x['prompt'] is None).num_rows == 0
            assert dataset.filter(lambda x: x['chosen'] is None).num_rows == 0
            assert dataset.filter(lambda x: x['rejected'] is None).num_rows == 0
            
            # assert that all entries in 'chosen' and 'rejected', which are lists of dictionaries, have first dictionary with key 'role' and value 'user'
            assert dataset.filter(lambda x: x['chosen'][0]['role'] != 'user').num_rows == 0
            assert dataset.filter(lambda x: x['rejected'][0]['role'] != 'user').num_rows == 0
            
            # assert that the gold_label_agreement is either 1 or -1
            assert dataset.filter(lambda x: x['gold_label_agreement'] not in [1, -1]).num_rows == 0
            # assert that the pseudo_label_agreement is either 1 or 0
            assert dataset.filter(lambda x: x['pseudo_label_agreement'] not in [1, 0]).num_rows == 0
            
            if i == 0:
                for j in range(10):
                    chunk = dataset.select(range(j * len(dataset) // 10, (j + 1) * len(dataset) // 10))
                    # check that the proportion of labels with gold_label_agreement is roughly equal to percent_gold_label
                    assert abs(chunk.filter(lambda x: x['gold_label_agreement'] == 1).num_rows / len(chunk) - percent_gold_label) < 0.03
                    if strategy.is_rank_0():
                        strategy.print(f"Chunk {j} has {chunk.filter(lambda x: x['gold_label_agreement'] == 1).num_rows}/{len(chunk)} = {chunk.filter(lambda x: x['gold_label_agreement'] == 1).num_rows / len(chunk):.4f} gold labels")
            
    return train_dataset, val_dataset, test_dataset


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
