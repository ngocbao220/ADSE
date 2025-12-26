"""
Arena để so sánh Static Diffusion vs Adaptive Diffusion với Introspection Net
"""
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import time
from tqdm import tqdm

sys.path.insert(0, 'src')
from llmtuner.tuner. core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments


# ==========================================
# INTROSPECTION NET
# ==========================================
class IntrospectionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


# ==========================================
# HELPER FUNCTIONS  
# ==========================================
def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    if stochastic:  
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index. gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def verify_sudoku(solution_str):
    """Verify solution"""
    if len(solution_str) != 81:
        return False
    try:
        for i in range(9):
            row = [int(solution_str[i*9+j]) for j in range(9)]
            if len(set(row)) != 9 or sum(row) != 45:
                return False
            col = [int(solution_str[j*9+i]) for j in range(9)]
            if len(set(col)) != 9 or sum(col) != 45:
                return False
            br, bc = (i//3)*3, (i%3)*3
            box = [int(solution_str[r*9+c]) for r in range(br,br+3) for c in range(bc,bc+3)]
            if len(set(box)) != 9 or sum(box) != 45:
                return False
        return True
    except:
        return False


# ==========================================
# ARENA
# ==========================================
class Arena: 
    def __init__(self, model_path, intro_path, csv_path, max_samples=None, 
                 conf_thresh=0.95, intro_thresh=0.3, diffusion_steps=20):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        self.intro_thresh = intro_thresh
        self.diffusion_steps = diffusion_steps
        
        print(f"{'='*60}")
        print("ARENA INITIALIZATION")
        print(f"{'='*60}")
        
        # Load Diffusion Model
        print(f"\n[1/3] Loading Diffusion Model...")
        model_args = ModelArguments(
            model_name_or_path="model_config_tiny",
            checkpoint_dir=model_path,
            cache_dir="./cache"
        )
        finetuning_args = FinetuningArguments(stage="mdm", finetuning_type="full")
        diffusion_args = DiffusionArguments(
            diffusion_steps=diffusion_steps,
            topk_decoding=True,
            token_reweighting=True,
            time_reweighting="linear",
            alpha=0.25,
            gamma=1,
            decoding_strategy="stochastic0. 5-linear"
        )
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_args, finetuning_args, is_trainable=False,
            diffusion_args=diffusion_args, stage="mdm"
        )
        self.model = self.model.to(self. device).eval()
        print("✓ Diffusion model loaded")
        
        # Load Introspection Net
        print(f"\n[2/3] Loading Introspection Net...")
        ckpt = torch.load(intro_path, map_location=self.device)
        input_dim = ckpt['input_dim']
        self.intro_net = IntrospectionNet(input_dim).to(self.device)
        self.intro_net. load_state_dict(ckpt['state_dict'])
        self.intro_net. eval()
        print(f"✓ Introspection net loaded (hidden_dim={input_dim})")
        
        # Load Dataset
        print(f"\n[3/3] Loading dataset...")
        self.df = pd.read_csv(csv_path)
        if max_samples: 
            self.df = self. df.sample(min(max_samples, len(self.df)), random_state=42)
        print(f"✓ Loaded {len(self.df)} samples\n")

    def decode_sudoku(self, token_ids):
        """Decode tokens to 81-digit string"""
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        digits = ''.join(c for c in text if c.isdigit())
        return digits[: 81] if len(digits) >= 81 else digits

    def solve_static(self, x, src_mask):
        """Standard TopK diffusion - Copy từ trainer logic"""
        init_maskable_mask = maskable_mask = ~src_mask
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                scores = torch.log_softmax(logits, dim=-1)
                scores[: ,: ,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                
                # Keep non-mask positions
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    # TopK decoding
                    rate = t / self.diffusion_steps
                    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
                    
                    # Stochastic with gumbel noise
                    noise_scale = 0.5
                    gumbel = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                    _scores = _scores_for_topk + noise_scale * rate * gumbel
                    
                    sorted_idx = _scores.sort(-1)[0]
                    cutoff_val = sorted_idx.gather(dim=-1, index=cutoff_len)
                    lowest_k_mask = _scores < cutoff_val
                    
                    xt = x0.masked_fill(lowest_k_mask, self.tokenizer. mask_token_id)
                else:
                    xt = x0
        
        return xt, self.diffusion_steps

    def solve_adaptive(self, x, src_mask):
        """Adaptive with:  Aggressive Unmask + Step Skipping + Introspection"""
        steps_taken = 0
        init_maskable_mask = maskable_mask = ~src_mask
        
        t = self.diffusion_steps - 1
        early_stopped = False
        stop_reason = None
        
        MIN_STEPS = max(int(self.diffusion_steps * 0.5), 8)  # Reduce to 50%
        
        while t >= 0:
            steps_taken += 1
            with torch. no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x. size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                # Forward
                outputs = self.model.model.transformer(
                    inputs_embeds=self.model.model.transformer.wte(xt),
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )
                hidden = outputs. last_hidden_state
                logits = self.model.model. lm_head(hidden)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                probs = torch.softmax(logits, dim=-1)
                confidence, x0_preds = probs.max(dim=-1)
                
                # Introspection
                b, s, h = hidden.shape
                hidden_flat = hidden.float().view(-1, h)
                intro_scores = self.intro_net(hidden_flat).view(b, s)
                
                # Get x0
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,:,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                # AGGRESSIVE TopK schedule (exponential)
                rate = (t / self.diffusion_steps) ** 0.6  # Faster unmask
                cutoff_len = (init_maskable_mask. sum(1, keepdim=True) * rate).long()
                _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
                
                # Stochastic
                noise_scale = 0.5
                gumbel = -torch.log(-torch. log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                _scores = _scores_for_topk + noise_scale * rate * gumbel
                sorted_idx = _scores.sort(-1)[0]
                cutoff_val = sorted_idx.gather(dim=-1, index=cutoff_len)
                mask_sched = _scores < cutoff_val
                
                # Confidence/Intro thresholds
                relaxed_conf_thresh = max(0.6, self.conf_thresh - 0.3 * (1 - rate))
                relaxed_intro_thresh = self.intro_thresh + 0.3 * (1 - rate)
                
                is_confident = confidence > relaxed_conf_thresh
                is_easy = intro_scores < relaxed_intro_thresh
                
                # HYBRID UNMASK:  Unmask if EITHER confident OR easy
                should_unmask = is_confident | is_easy
                
                # Masking decision
                if t > (self.diffusion_steps * 0.75):
                    new_mask = mask_sched & init_maskable_mask
                else:
                    # Only mask if schedule says so AND should NOT unmask
                    new_mask = mask_sched & (~should_unmask) & init_maskable_mask
                
                xt = x0.masked_fill(new_mask, self.tokenizer.mask_token_id)
                maskable_mask = new_mask
                
                # ===== EARLY STOPPING + STEP SKIPPING =====
                if steps_taken >= MIN_STEPS and t > 0:  # Thêm t > 0
                    num_masks_left = maskable_mask.sum().item()
                    
                    # Condition 1: No masks
                    if num_masks_left == 0:
                        early_stopped = True
                        stop_reason = f"no_masks@{steps_taken}"
                        break
                    
                    # Condition 2: Few masks + high conf → SKIP ahead
                    # CHỈ skip nếu còn nhiều hơn 3 ô
                    if 3 < num_masks_left < 8 and t > 3:  # Điều chỉnh threshold
                        target_positions = init_maskable_mask[0]
                        target_conf = confidence[0, target_positions]
                        target_intro = intro_scores[0, target_positions]
                        
                        avg_conf = target_conf.mean().item()
                        max_intro = target_intro.max().item()
                        
                        if avg_conf > 0.95 and max_intro < 0.15:  # Raise threshold
                            skip_steps = max(1, min(t - 2, 2))  # Giảm số steps skip
                            t -= skip_steps
                            steps_taken += (skip_steps - 1)
                            early_stopped = True
                            stop_reason = f"skip@{steps_taken}(n={num_masks_left})"
                            continue
                
                    # Force unmask ở bước cuối
                    if t == 0:
                        new_mask = torch.zeros_like(maskable_mask)
                    
                    xt = x0.masked_fill(new_mask, self. tokenizer.mask_token_id)
                    maskable_mask = new_mask
            
            t -= 1
        
        return xt, steps_taken, early_stopped, stop_reason

    def run_battle(self, save_details=None):
        results = {
            "Static":  {"correct": 0, "valid": 0, "steps": 0, "time": 0},
            "Adaptive": {"correct": 0, "valid": 0, "steps": 0, "time": 0, "early_stops": 0}
        }
        
        total = 0
        stop_reasons = []
        
        print(f"{'='*60}")
        print(f"BATTLE START: {len(self.df)} samples")
        print(f"{'='*60}\n")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Battle"):
            quiz = row['quizzes']
            gt_solution = row['solutions']
            
            quiz_digits = ''.join(c for c in quiz if c.isdigit())
            if len(quiz_digits) != 81:
                continue
            
            input_ids = self.tokenizer. encode(quiz_digits) + [self.tokenizer.sep_token_id]
            src_len = len(input_ids)
            
            cutoff_len = 164
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
            input_ids = input_ids[:cutoff_len]
            
            x = torch.tensor([input_ids]).to(self.device)
            src_mask = torch.zeros(cutoff_len, dtype=torch.bool).to(self.device)
            src_mask[: src_len] = True
            src_mask = src_mask. unsqueeze(0)
            
            # STATIC
            start = time. time()
            out_s, steps_s = self.solve_static(x, src_mask)
            time_s = time. time() - start
            
            target_ids_s = out_s[0, src_len:]. cpu().tolist()
            pred_s_decode = self.tokenizer.decode(target_ids_s, skip_special_tokens=True)
            pred_s_str = ''.join(c for c in pred_s_decode if c.isdigit())[:81]
            
            is_correct_s = (pred_s_str == gt_solution)
            is_valid_s = verify_sudoku(pred_s_str)
            
            # ADAPTIVE
            start = time.time()
            out_a, steps_a, early_stopped, stop_reason = self.solve_adaptive(x, src_mask)
            time_a = time.time() - start
            
            target_ids_a = out_a[0, src_len:].cpu().tolist()
            pred_a_decode = self.tokenizer.decode(target_ids_a, skip_special_tokens=True)
            pred_a_str = ''.join(c for c in pred_a_decode if c.isdigit())[:81]
            
            is_correct_a = (pred_a_str == gt_solution)
            is_valid_a = verify_sudoku(pred_a_str)
            
            # Update stats
            total += 1
            
            if is_correct_s:  
                results["Static"]["correct"] += 1
            if is_valid_s:
                results["Static"]["valid"] += 1
            results["Static"]["steps"] += steps_s
            results["Static"]["time"] += time_s
            
            if is_correct_a: 
                results["Adaptive"]["correct"] += 1
            if is_valid_a:
                results["Adaptive"]["valid"] += 1
            results["Adaptive"]["steps"] += steps_a
            results["Adaptive"]["time"] += time_a
            
            if early_stopped:
                results["Adaptive"]["early_stops"] += 1
                stop_reasons.append(stop_reason)
            
            # Debug first sample
            if total == 1:
                print("\n[DEBUG] First sample:")
                print(f"Quiz: {quiz[: 30]}...")
                print(f"GT:   {gt_solution[:30]}...")
                print(f"Static:    {pred_s_str[:30]}...  | Correct: {is_correct_s}")
                print(f"Adaptive: {pred_a_str[: 30]}... | Correct: {is_correct_a} | Steps: {steps_a}/{steps_s} |     Early:  {early_stopped} ({stop_reason})")
                print()
        
        # PRINT RESULTS
        print(f"\n{'='*60}")
        print("BATTLE RESULTS")
        print(f"{'='*60}")
        print(f"Total:  {total}\n")
        
        print(">>> STATIC")
        acc_s = results['Static']['correct'] / total
        val_s = results['Static']['valid'] / total
        avg_steps_s = results['Static']['steps'] / total
        print(f"  Accuracy:   {acc_s:.2%} ({results['Static']['correct']}/{total})")
        print(f"  Valid:     {val_s:.2%} ({results['Static']['valid']}/{total})")
        print(f"  Avg Steps: {avg_steps_s:.2f}")
        
        print("\n>>> ADAPTIVE")
        acc_a = results['Adaptive']['correct'] / total
        val_a = results['Adaptive']['valid'] / total
        avg_steps_a = results['Adaptive']['steps'] / total
        early_stop_rate = results['Adaptive']['early_stops'] / total
        print(f"  Accuracy:    {acc_a:.2%} ({results['Adaptive']['correct']}/{total})")
        print(f"  Valid:       {val_a:.2%} ({results['Adaptive']['valid']}/{total})")
        print(f"  Avg Steps:   {avg_steps_a:.2f}")
        print(f"  Early Stops: {early_stop_rate:.1%} ({results['Adaptive']['early_stops']}/{total})")
        
        if stop_reasons:
            from collections import Counter
            reason_counts = Counter(stop_reasons)
            print(f"\n  Stop reasons:")
            for reason, count in reason_counts.most_common():
                print(f"    {reason}: {count}")
        
        print(f"\n>>> COMPARISON")
        speedup = avg_steps_s / avg_steps_a if avg_steps_a > 0 else 0
        print(f"  Speedup:     {speedup:.2f}x ({avg_steps_s:.1f} → {avg_steps_a:.1f} steps)")
        print(f"  Acc Delta:   {acc_a - acc_s:.2%}")
        print(f"  Time saved: {(1 - avg_steps_a/avg_steps_s)*100:.1f}%")
        print(f"{'='*60}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--intro_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--conf_thresh", type=float, default=0.95)
    parser.add_argument("--intro_thresh", type=float, default=0.3)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--save_details", default=None)
    args = parser.parse_args()
    
    arena = Arena(
        args.model_path, args.intro_path, args.csv_path,
        max_samples=args.max_samples,
        conf_thresh=args. conf_thresh,
        intro_thresh=args.intro_thresh,
        diffusion_steps=args.diffusion_steps
    )
    
    arena.run_battle(save_details=args.save_details)


if __name__ == "__main__":
    main()