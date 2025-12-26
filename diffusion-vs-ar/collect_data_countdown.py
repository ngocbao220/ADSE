"""
Thu thập dữ liệu Introspection cho bài toán COUNTDOWN
"""
import os
import sys
import torch
import json
import numpy as np
import argparse
from tqdm import tqdm

sys.path.insert(0, 'src')

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments


def collect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to checkpoint folder")
    parser.add_argument("--data_path", required=True, help="Path to countdown . json/. jsonl file")
    parser.add_argument("--save_path", default="introspection_data_countdown.pt")
    parser.add_argument("--num_samples", type=int, default=10000, help="Số lượng mẫu train")
    parser.add_argument("--model_config", default="model_config_tiny", help="Model config name")
    args = parser.parse_args()

    # ==========================================
    # 1. LOAD MODEL
    # ==========================================
    print(f"Loading Model from {args.model_path}...")
    model_args = ModelArguments(
        model_name_or_path=args.model_config,
        checkpoint_dir=args.model_path,
        cache_dir="./cache"
    )
    finetuning_args = FinetuningArguments(stage="mdm", finetuning_type="full")
    diffusion_args = DiffusionArguments(
        diffusion_steps=20,
        topk_decoding=True,
        token_reweighting=True,
        time_reweighting="linear",
        alpha=0.25,
        gamma=1,
        decoding_strategy="stochastic0.5-linear"
    )
    
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, 
        is_trainable=False, 
        diffusion_args=diffusion_args, 
        stage="mdm"
    )
    model = model.cuda().eval()
    
    # Lấy hidden dimension
    if hasattr(model.model, 'config'):
        config = model.model.config
    else:
        config = model. config
    
    hidden_dim = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
    print(f"✓ Model loaded - Hidden Dimension: {hidden_dim}")

    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    print(f"Reading data from {args.data_path}...")
    data = []
    
    with open(args.data_path, 'r') as f:
        if args.data_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        else:  # .json
            data = json.load(f)
    
    # Sample random subset
    if len(data) > args.num_samples:
        import random
        random.seed(42)
        data = random.sample(data, args.num_samples)
    
    print(f"✓ Loaded {len(data)} samples")
    
    data_buffer_x = []
    data_buffer_y = []

    # ==========================================
    # 3. COLLECT DATA
    # ==========================================
    print("\nCollecting Introspection Data...")
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer. pad_token_id
    sep_token_id = tokenizer. sep_token_id
    
    cutoff_len = 128  # Max sequence length for countdown
    
    for item in tqdm(data, desc="Processing"):
        input_str = item['input']
        output_str = item['output']
        
        # Encode full sequence:  input + sep + output
        input_ids = tokenizer.encode(input_str)
        output_ids = tokenizer.encode(output_str)
        
        # Construct full sequence
        src_ids = input_ids + [sep_token_id]
        tgt_ids = output_ids
        
        src_len = len(src_ids)
        full_ids = src_ids + tgt_ids
        
        # Pad/truncate to cutoff_len
        if len(full_ids) < cutoff_len:
            full_ids = full_ids + [pad_token_id] * (cutoff_len - len(full_ids))
        else:
            full_ids = full_ids[:cutoff_len]
        
        x0 = torch.tensor([full_ids]).cuda()  # [1, cutoff_len]
        
        # Create maskable mask (only target part can be masked)
        maskable_mask = torch.zeros(cutoff_len, dtype=torch.bool).cuda()
        maskable_mask[src_len:len(src_ids + tgt_ids)] = True
        
        if maskable_mask.sum() == 0:
            continue  # No target to mask
        
        # Random timestep (1 to 19)
        t = np.random.randint(1, 20)
        
        # Create noise mask based on timestep
        # Higher t = more masking
        mask_prob = t / 20.0
        
        # Only mask positions in maskable region
        rand_mask = torch.rand(cutoff_len).cuda() < mask_prob
        mask = rand_mask & maskable_mask
        
        if mask.sum() == 0:
            continue  # Nothing masked
        
        # Create noisy input
        xt = x0.clone()
        xt[0, mask] = mask_token_id
        
        with torch.no_grad():
            # Forward pass to get hidden states
            t_tensor = torch.full((1,), t, device=x0.device)
            attention_mask = torch.ones_like(xt)
            
            # Get hidden states
            backbone = model.model  # GPT2LMHeadModel or similar
            outputs = backbone.transformer(
                inputs_embeds=backbone.transformer.wte(xt),
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            last_hidden = outputs.last_hidden_state  # [1, cutoff_len, hidden_dim]
            
            # Get predictions
            logits = backbone.lm_head(last_hidden)
            preds = logits. argmax(dim=-1)  # [1, cutoff_len]
            
            # Label:  1 = wrong prediction (hard), 0 = correct (easy)
            is_wrong = (preds != x0).long()
            
            # Only collect data at masked positions
            selected_hidden = last_hidden[0, mask]  # [N, hidden_dim]
            selected_labels = is_wrong[0, mask]      # [N]
            
            if selected_hidden.size(0) == 0:
                continue
            
            # Store in buffer (use FP16 to save memory)
            data_buffer_x.append(selected_hidden.cpu().half())
            data_buffer_y.append(selected_labels.cpu())

    # ==========================================
    # 4. SAVE DATA
    # ==========================================
    if len(data_buffer_x) == 0:
        print("ERROR: No data collected!")
        return
    
    print("\nConcatenating data...")
    X = torch.cat(data_buffer_x, dim=0)
    Y = torch.cat(data_buffer_y, dim=0)
    
    print(f"\n{'='*60}")
    print("DATASET CREATED")
    print(f"{'='*60}")
    print(f"Shape:         {X.shape}")
    print(f"Hidden Dim:   {hidden_dim}")
    print(f"Total Tokens:  {X.size(0)}")
    print(f"Error Rate:   {Y.float().mean():.2%} (label=1 ratio)")
    print(f"Hard Tokens:  {Y.sum().item()} / {Y.size(0)}")
    print(f"Easy Tokens:  {(Y == 0).sum().item()} / {Y.size(0)}")
    print(f"\nSaving to {args.save_path}...")
    
    torch.save({
        "x": X,
        "y": Y,
        "hidden_dim": hidden_dim,
        "num_samples": len(data),
        "cutoff_len": cutoff_len
    }, args.save_path)
    
    print("✓ DONE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    collect()