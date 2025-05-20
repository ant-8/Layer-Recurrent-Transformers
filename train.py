import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import gc
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch._dynamo.config.capture_scalar_outputs = True
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_inverse_sqrt_schedule
from datasets import load_dataset, load_from_disk

from modeling_gpt2 import LoopedGPT2ForCausalLM
from modeling_llama import LoopedLlamaForCausalLM
from looped_gpt2_configuration import LoopedGPT2Config
from looped_llama_configuration import LoopedLlamaConfig
from torch.cuda.amp import autocast

from ptflops import get_model_complexity_info

import wandb

def truncate_sequences(input_ids, attention_mask):
    valid_lengths = attention_mask.sum(dim=1)
    max_valid_length = valid_lengths.max().item()

    truncated_input_ids = input_ids[:, :max_valid_length]
    truncated_attention_mask = attention_mask[:, :max_valid_length]

    return truncated_input_ids, truncated_attention_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Train a (looped) GPT-2 model.")

    parser.add_argument("--project_name", type=str, default="looped-gpt2-fineweb-20M")
    parser.add_argument("--run_group_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--compute_budget", type=float, default=0.0, 
                        help="Compute budget in FLOPs (can be large). 0 means no budget.")
    parser.add_argument("--val_split_percent", type=float, default=0.0,
                        help="Ratio (0 to 1) used to create a validation subset from training data if no separate validation dataset is provided.")
    parser.add_argument("--tokenized_dataset_path", type=str, default="tokenized_dataset_512",
                        help="Path to an *already tokenized* dataset. Used if --hf_dataset is not specified.")
    parser.add_argument("--tokenized_validation_dataset_path", type=str, default="tokenized_dataset_512_val",
                        help="Path to an *already tokenized* validation dataset. Overridden by --hf_dataset_validation_split if that is provided.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--loop_map", nargs='+', type=int, default=[1,1,1,1],
                        help="Defines loop factors per layer; e.g. 1 1 1 1.")
    parser.add_argument("--sequential_looping", action='store_true', default=False)
    parser.add_argument("--attention_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--ctx_len", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--logging_step_interval", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=512)  # default 50257 in GPT-2
    parser.add_argument("--deterministic", action='store_true', default=True)
    parser.add_argument("--save_model", action='store_true', default=False)
    parser.add_argument("--auto_adjust_log_interval", action='store_true', default=True)
    parser.add_argument("--use_rope", default=True)
    parser.add_argument("--use_nope", default=False)

    parser.add_argument("--lr_ratio", type=float, default=0.02, 
                        help="Ratio of warmup steps to total steps. E.g. 0.02 for 2% warmup.")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_loops", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--architecture", type=str, default="gpt2", 
                        help="Only 'gpt2' supported in this script.")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Train device")

    parser.add_argument("--hf_subset", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path or HF hub name of the tokenizer. Used only if --hf_dataset is specified.")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="Hugging Face dataset name or path. Overrides tokenized_dataset_path if provided.")
    parser.add_argument("--hf_dataset_validation_split", type=str, default=None,
                        help="Name of the validation split if using a Hugging Face dataset. Overrides tokenized_validation_dataset_path if set.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader.")

    args = parser.parse_args()
    return args


def run(args):
    torch.backends.cudnn.deterministic = True
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.set_float32_matmul_precision('high')

    is_baseline = args.use_baseline_model
    print(f"using baseline model: {is_baseline}")
    if args.architecture == "gpt2":
        if is_baseline:
            config = GPT2Config(
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                n_head=args.n_head,
                vocab_size=args.vocab_size, 
                n_positions=args.ctx_len,
                _attn_implementation=args.attention_implementation
            )
            model = GPT2LMHeadModel(config)
        else:
            config = LoopedGPT2Config(
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                n_head=args.n_head,
                sequential_looping=args.sequential_looping,
                vocab_size=args.vocab_size,
                max_position_embeddings=args.ctx_len,
                loop_map=args.loop_map,
                num_loops=args.num_loops,
                _attn_implementation=args.attention_implementation
            )
            model = LoopedGPT2ForCausalLM(config)
    elif args.architecture == "llama":
        if is_baseline:
            config = LlamaConfig(
                hidden_size=args.n_embd,
                num_hidden_layers=args.n_layer,
                num_attention_heads=args.n_head,
                vocab_size=args.vocab_size, 
                max_position_embeddings=args.ctx_len,
                tie_word_embeddings=True,
                _attn_implementation=args.attention_implementation,
                use_cache=False
            )
            config.intermediate_size = config.hidden_size * 4
            print(config)
            #print(model)
            model = LlamaForCausalLM(config)
        else:
            config = LoopedLlamaConfig(
                hidden_size=args.n_embd,
                num_hidden_layers=args.n_layer,
                num_attention_heads=args.n_head,
                sequential_looping=args.sequential_looping,
                vocab_size=args.vocab_size,
                max_position_embeddings=args.ctx_len,
                loop_map=args.loop_map,
                use_loop_encoding=args.use_loop_encoding,
                use_adaptive_layer_norm=args.use_adaptive_layer_norm,
                use_recurrent_embeds=args.use_recurrent_embeds,
                num_loops=args.num_loops,
                positional_encoding=args.positional_encoding or "rope",
                tie_word_embeddings=True,
                attn_scaling=args.attn_scaling,
                intermediate_size_map=args.intermediate_size_map,
                use_head_scale=args.use_head_scale,
                _attn_implementation=args.attention_implementation,
                use_cache=False
            )
            config.intermediate_size = config.hidden_size * 4
            print(config)
            model = LoopedLlamaForCausalLM(config)
    else:
        raise ValueError("Only 'gpt2' and 'llama' architecture is currently supported by this script.")

    model = model.to(args.device) if torch.cuda.is_available() else model
    print(f"Model Size (# parameters): {sum(p.numel() for p in model.parameters())}")

    wandb.init(
        project=args.project_name,
        group=args.run_group_name,
        name=args.run_name,
        config = {
            "dims": args.n_embd,
            "layers": args.n_layer,
            "epochs": args.num_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "warmup_ratio": args.lr_ratio,
            "attn_impl": args.attention_implementation,
            "deterministic": args.deterministic,
            "architecture": args.architecture,
            "loop_map": args.loop_map,  
        }
    )
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = True

    #pad_token_id = None
    if args.tokenizer:
        print(f"Loading tokenizer from: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        #pad_token_id = tokenizer.pad_token_id
    
    if args.hf_dataset:
        if not args.tokenizer:
            raise ValueError("Please provide a tokenizer path/name via --tokenizer when using --hf_dataset.")

        print(f"Loading Hugging Face dataset: {args.hf_dataset}")
        dataset_all = load_dataset(args.hf_dataset, args.hf_subset)
        if "train" in dataset_all:
            train_dataset = dataset_all["train"]
        else:
            train_dataset = dataset_all[list(dataset_all.keys())[0]]

        def tokenize_collate_fn(examples):
            texts = [ex["text"] for ex in examples]
        
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.ctx_len,
                return_tensors="pt"
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }


        print("Creating train DataLoader...")
        if args.val_split_percent == 0.0 and not args.hf_dataset_validation_split:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=tokenize_collate_fn,
                num_workers=args.num_workers
            )
            val_loader = None

        else:
            if args.hf_dataset_validation_split:
                if args.hf_dataset_validation_split not in dataset_all:
                    raise ValueError(f"Split '{args.hf_dataset_validation_split}' not found in dataset. Available splits: {list(dataset_all.keys())}")
                val_dataset = dataset_all[args.hf_dataset_validation_split]
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=tokenize_collate_fn,
                    num_workers=args.num_workers
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=tokenize_collate_fn,
                    num_workers=args.num_workers
                )
            else:
                dataset_length = len(train_dataset)
                val_size = int(dataset_length * args.val_split_percent)
                train_size = dataset_length - val_size

                train_split, val_split = random_split(
                    train_dataset, 
                    [train_size, val_size], 
                    generator=torch.Generator().manual_seed(args.seed)
                )

                train_loader = DataLoader(
                    train_split,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=tokenize_collate_fn,
                    num_workers=args.num_workers
                )
                val_loader = DataLoader(
                    val_split,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=tokenize_collate_fn,
                    num_workers=args.num_workers
                )

    else:
        print(f"Loading tokenized dataset from: {args.tokenized_dataset_path}")
        train_dataset = load_from_disk(args.tokenized_dataset_path)

        if args.tokenized_validation_dataset_path and os.path.exists(args.tokenized_validation_dataset_path) and os.path.isdir(args.tokenized_validation_dataset_path):
            print(f"Found tokenized validation dataset at: {args.tokenized_validation_dataset_path}")
            val_dataset = load_from_disk(args.tokenized_validation_dataset_path)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        elif args.val_split_percent > 0:
            val_size = int(len(train_dataset) * args.val_split_percent)
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = random_split(
                train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed)
            )
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = None

    def input_constructor(input_shape):
        return torch.ones(*input_shape, dtype=torch.long).to(args.device)

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    if args.force_cast_bf16:
        model = model.bfloat16()
    
    model.eval()
    with autocast(dtype=dtype):
        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                model,
                (args.batch_size, args.ctx_len),
                as_strings=False,
                print_per_layer_stat=False,
                input_constructor=input_constructor
            )
    flops_per_forward = macs * 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_per_epoch = len(train_loader)

    
    if args.compute_budget > 0:
        compute_budget_int = int(args.compute_budget)
        num_training_steps = (compute_budget_int // flops_per_forward) // args.gradient_accumulation_steps

        num_epochs = 999
    else:
        num_training_steps = (args.num_epochs * steps_per_epoch) // args.gradient_accumulation_steps
        if args.batch_size_tokens:
            if args.total_dataset_tokens:
                total_dataset_tokens = args.total_dataset_tokens
            else:
                total_dataset_tokens = steps_per_epoch * args.ctx_len * args.batch_size
                
            num_training_steps = (args.num_epochs * total_dataset_tokens) // args.batch_size_tokens
        num_epochs = args.num_epochs

    scheduler_fn = get_linear_schedule_with_warmup if args.lr_scheduler == "linear" else get_cosine_schedule_with_warmup 
    scheduler = scheduler_fn(
        optimizer, 
        num_warmup_steps=int(num_training_steps * args.lr_ratio), 
        num_training_steps=num_training_steps+100
    )

    import time
    
    print("Starting training...")

    if args.loop_warmup_ratio:
        loop_warmup_steps = int(num_training_steps * args.loop_warmup_ratio)
        loop_weight = 1e-8 if loop_warmup_steps > 0 else 1
    else:
        loop_weight = None
    
    step = 0
    update_step = 0
    total_tokens = 0
    total_batch_tokens = 0
    total_flops = 0
    
    logging_step_interval = args.logging_step_interval
    target_logs_per_second = 3.0
    last_log_time = time.time()
    
    model.train()
    
    uncompiled_model = model
    model = torch.compile(model)

    import math
    for epoch in range(math.ceil(num_epochs)):
        if args.compute_budget > 0 and total_flops >= args.compute_budget:
            break
    
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            input_ids, attention_mask = truncate_sequences(input_ids, attention_mask)
            labels = input_ids.clone()
    
            tokens_processed = torch.sum(attention_mask == 1).item()
            total_tokens += tokens_processed
            total_batch_tokens += tokens_processed
            
            total_flops += flops_per_forward

            with autocast(dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)#, loop_weight=loop_weight)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
    
            # Update step
            if args.batch_size_tokens:
                should_update = total_batch_tokens >= args.batch_size_tokens
                if should_update:
                    total_batch_tokens = total_batch_tokens % args.batch_size_tokens
            else:
                should_update = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            if should_update or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                update_step += 1
                if loop_weight is not None:
                    if update_step > loop_warmup_steps * 5:
                        loop_weight = 5
                    elif loop_warmup_steps > 0: 
                        loop_weight = min(5, update_step / loop_warmup_steps) ** 2
                if update_step % logging_step_interval == 0:
                    _log = {
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "tokens": total_tokens,
                        "flops": total_flops
                    }
                    if loop_weight is not None:
                        _log["loop_weight"] = loop_weight
                    wandb.log(_log)
                    progress_bar.set_postfix({
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0]
                    })
    
                    if args.auto_adjust_log_interval:
                        current_time = time.time()
                        dt = current_time - last_log_time
                        last_log_time = current_time

                        desired_dt = 1.0 / target_logs_per_second

                        if dt > desired_dt:
                            logging_step_interval = max(1, logging_step_interval - 1)
                        else:
                            logging_step_interval += 1
    
                epoch_loss += loss.item()
                if args.compute_budget > 0 and total_flops >= args.compute_budget:
                    print(f"Total FLOPs budget of {args.compute_budget} reached, ending training. {step / len(train_loader):.2f} epochs trained.")
                    break
            step += 1
            if step >= (args.num_epochs * steps_per_epoch):
                print(f"Max steps of reached, ending training. {step / len(train_loader):.2f} epochs trained.")
                break
    
        avg_epoch_loss = (epoch_loss / len(train_loader)) * args.gradient_accumulation_steps
    
    epochs_trained = step / len(train_loader) if len(train_loader) > 0 else 0.0
    print("Training completed.")

    avg_val_loss = None
    if val_loader is not None:
        print("Starting validation...")
        model.eval()

        torch.cuda.empty_cache()
        gc.collect()
        
        if args.test_ctx_lens:
            for test_seq_len in args.test_ctx_lens:
                torch.cuda.empty_cache()
                gc.collect()
                val_loss = 0
                with torch.no_grad():
                    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
                    for batch in progress_bar:
                        input_ids = batch["input_ids"].to(args.device)
                        input_ids = input_ids[:, :test_seq_len]
                        attention_mask = batch["attention_mask"].to(args.device)[:, :test_seq_len]
                        labels = input_ids.clone()
                        loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation completed. Average Loss: {avg_val_loss:.4f}")
                wandb.log({"validation_loss": avg_val_loss, "text_seq_len": test_seq_len})
        else:
            val_loss = 0
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc="Validation", leave=False)
                for batch in progress_bar:
                    input_ids = batch["input_ids"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device)[:, :test_seq_len]
                    labels = input_ids.clone()
                    if args.architecture == "llama":
                        loss = uncompiled_model(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss
                    else:
                        loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation completed. Average Loss: {avg_val_loss:.4f}")
            wandb.log({"validation_loss": avg_val_loss})

    if args.tokenizer is not None:
        model.eval()
        prompt = '''Once'''
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids.to(args.device), 
            max_new_tokens=32,
            do_sample=False
        )
    
        output = tokenizer.decode(output_ids[0])
    print("Sample generation:")
    print(output)

    output_dir = f"./results/{wandb.run.id}"
    os.makedirs(output_dir, exist_ok=True)

    num_params = sum(p.numel() for p in model.parameters())
    result_data = {
        "Run ID": wandb.run.id,
        "Run Name": args.run_name,
        "Dataset": (args.hf_dataset if args.hf_dataset else args.tokenized_dataset_path),
        "Vocab Size": config.vocab_size,
        "Ctx Len": args.ctx_len,
        "# Params": num_params,
        "# dims": args.n_embd,
        "# layers": args.n_layer,
        "# FLOPs": total_flops,
        "Epochs Trained": epochs_trained,
        "Tokens": total_tokens,
        "LR": args.lr,
        "BSz": args.batch_size,
        "Warmup Ratio": args.lr_ratio,
        "Attention Impl": args.attention_implementation,
        "Determinism": args.deterministic,
        "Architecture": args.architecture,
        "Loop Map": args.loop_map,
        "Run Group Name": args.run_group_name,
        "Valid Loss": avg_val_loss,
        "Wandb Link": wandb.run.url,
    }

    wandb.log({
        "lr": args.lr,
        "tokens_trained": total_tokens,
        "epochs_trained": epochs_trained,
        "flops": total_flops
    })
    
    filename = f"./results/{wandb.run.id}/train_config.json"
    with open(filename, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"Run info saved to {filename}")

    if args.save_model:
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
    wandb.finish()
    return result_data


if __name__ == "__main__":
    args = parse_args()
    run(args)
