"""
Arena để so sánh Static Diffusion vs Adaptive Diffusion với Introspection Net
+ FLOPs Measurement + Re-noising Strategy
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
from collections import defaultdict

sys.path.insert(0, 'src')
from llmtuner.tuner. core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments


# ==========================================
# FLOPS COUNTER
# ==========================================
class FLOPsCounter: 
    """Count FLOPs for transformer operations"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_flops = 0
        self.breakdown = defaultdict(int)
    
    def count_transformer_forward(self, batch_size, seq_len, hidden_dim, num_layers, vocab_size):
        """Count FLOPs for one transformer forward pass"""
        flops_per_layer = 0
        
        # Self-attention:  Q, K, V projections + attention + output projection
        flops_per_layer += 3 * (2 * batch_size * seq_len * hidden_dim * hidden_dim)
        flops_per_layer += 2 * batch_size * seq_len * seq_len * hidden_dim
        flops_per_layer += 2 * batch_size * seq_len * seq_len * hidden_dim
        flops_per_layer += 2 * batch_size * seq_len * hidden_dim * hidden_dim
        
        # FFN
        ffn_dim = 4 * hidden_dim
        flops_per_layer += 2 * batch_size * seq_len * hidden_dim * ffn_dim
        flops_per_layer += 2 * batch_size * seq_len * ffn_dim * hidden_dim
        
        transformer_flops = num_layers * flops_per_layer
        lm_head_flops = 2 * batch_size * seq_len * hidden_dim * vocab_size
        
        total = transformer_flops + lm_head_flops
        
        self.breakdown['transformer'] += transformer_flops
        self.breakdown['lm_head'] += lm_head_flops
        self.total_flops += total
        
        return total
    
    def count_introspection_forward(self, batch_size, seq_len, hidden_dim):
        """Count FLOPs for introspection net"""
        flops = 0
        flat_size = batch_size * seq_len
        
        flops += 2 * flat_size * hidden_dim * 256
        flops += flat_size * 256
        flops += 5 * flat_size * 256
        flops += 2 * flat_size * 256 * 64
        flops += flat_size * 64
        flops += 2 * flat_size * 64 * 1
        flops += 4 * flat_size * 1
        
        self.breakdown['introspection'] += flops
        self.total_flops += flops
        
        return flops
    
    def count_topk_masking(self, batch_size, seq_len):
        """Count FLOPs for TopK masking"""
        flops = 0
        flops += 10 * batch_size * seq_len
        
        import math
        flops += batch_size * seq_len * math.log2(seq_len + 1)
        flops += 5 * batch_size * seq_len
        
        self.breakdown['topk_masking'] += flops
        self.total_flops += flops
        
        return flops
    
    def count_confidence_computation(self, batch_size, seq_len, vocab_size):
        """Count FLOPs for confidence computation"""
        flops = 0
        flops += 3 * batch_size * seq_len * vocab_size
        flops += batch_size * seq_len * (vocab_size - 1)
        
        self.breakdown['confidence'] += flops
        self.total_flops += flops
        
        return flops
    
    def count_renoising(self, batch_size, seq_len):
        """Count FLOPs for re-noising operation"""
        flops = 0
        # Mask token replacement:  ~2 ops per token
        flops += 2 * batch_size * seq_len
        
        self. breakdown['renoising'] += flops
        self.total_flops += flops
        
        return flops
    
    def get_summary(self):
        return {
            'total_gflops': self.total_flops / 1e9,
            'breakdown_gflops':  {k: v/1e9 for k, v in self.breakdown.items()}
        }


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
# ARENA WITH FLOPS + RE-NOISING
# ==========================================
class Arena:
    def __init__(self, model_path, intro_path, csv_path, max_samples=None, 
                 conf_thresh=0.95, intro_thresh=0.3, diffusion_steps=20,
                 renoising_thresh=0.8, renoising_steps=3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        self.intro_thresh = intro_thresh
        self.diffusion_steps = diffusion_steps
        self.renoising_thresh = renoising_thresh  # Introspection threshold for re-noising
        self. renoising_steps = renoising_steps    # Number of steps to go back
        
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
        
        # Get model config
        config = self.model.model. config
        self.hidden_dim = config.n_embd
        self.num_layers = config.n_layer
        self.vocab_size = config.vocab_size
        
        print(f"✓ Diffusion model loaded")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - Num layers: {self.num_layers}")
        print(f"  - Vocab size: {self.vocab_size}")
        
        # Load Introspection Net
        print(f"\n[2/3] Loading Introspection Net...")
        ckpt = torch.load(intro_path, map_location=self. device)
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
        print(f"✓ Loaded {len(self.df)} samples")
        print(f"✓ Re-noising enabled: threshold={renoising_thresh}, steps={renoising_steps}\n")

    def decode_sudoku(self, token_ids):
        """Decode tokens to 81-digit string"""
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        digits = ''.join(c for c in text if c.isdigit())
        return digits[: 81] if len(digits) >= 81 else digits

    def solve_static(self, x, src_mask, flops_counter):
        """Standard TopK diffusion with FLOPs counting"""
        init_maskable_mask = maskable_mask = ~src_mask
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                # Forward pass
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                flops_counter.count_transformer_forward(
                    batch_size, seq_len, self.hidden_dim, self.num_layers, self.vocab_size
                )
                
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Confidence computation
                scores = torch.log_softmax(logits, dim=-1)
                flops_counter.count_confidence_computation(batch_size, seq_len, self.vocab_size)
                
                scores[: ,: ,self.tokenizer.vocab_size: ] = -1000
                x0_scores, x0 = scores.max(-1)
                
                x0 = xt. masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    # TopK masking
                    rate = t / self.diffusion_steps
                    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                    _scores_for_topk = x0_scores. masked_fill(~init_maskable_mask, 1000.0)
                    
                    noise_scale = 0.5
                    gumbel = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                    _scores = _scores_for_topk + noise_scale * rate * gumbel
                    
                    flops_counter.count_topk_masking(batch_size, seq_len)
                    
                    sorted_idx = _scores.sort(-1)[0]
                    cutoff_val = sorted_idx. gather(dim=-1, index=cutoff_len)
                    lowest_k_mask = _scores < cutoff_val
                    
                    xt = x0.masked_fill(lowest_k_mask, self.tokenizer.mask_token_id)
                else:
                    xt = x0
        
        return xt, self.diffusion_steps

    def solve_adaptive_with_renoising(self, x, src_mask, flops_counter):
        """
        Adaptive decoding with RE-NOISING strategy
        
        Re-noising concept:
        - Monitor introspection scores during decoding
        - If detect difficult tokens (high introspection score), trigger re-noising
        - Re-noising:  Jump back N steps and re-mask those difficult tokens
        - Continue diffusion from earlier timestep with fresh perspective
        """
        steps_taken = 0
        init_maskable_mask = maskable_mask = ~src_mask
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        t = self.diffusion_steps - 1
        early_stopped = False
        stop_reason = None
        renoising_count = 0
        
        MIN_STEPS = max(int(self.diffusion_steps * 0.5), 8)
        MAX_RENOISING = 2  # Maximum re-noising attempts per sample
        
        # History for re-noising
        history = {}  # {timestep: (xt, maskable_mask)}
        
        while t >= 0:
            steps_taken += 1
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self. tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                # Forward
                outputs = self.model. model. transformer(
                    inputs_embeds=self.model.model.transformer.wte(xt),
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )
                hidden = outputs.last_hidden_state
                logits = self.model.model.lm_head(hidden)
                flops_counter.count_transformer_forward(
                    batch_size, seq_len, self.hidden_dim, self.num_layers, self.vocab_size
                )
                
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Confidence
                probs = torch.softmax(logits, dim=-1)
                confidence, x0_preds = probs.max(dim=-1)
                flops_counter.count_confidence_computation(batch_size, seq_len, self.vocab_size)
                
                # Introspection
                b, s, h = hidden.shape
                hidden_flat = hidden.float().view(-1, h)
                intro_scores = self.intro_net(hidden_flat).view(b, s)
                flops_counter.count_introspection_forward(batch_size, seq_len, self.hidden_dim)
                
                # Get x0
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,:,self.tokenizer. vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                # ===== RE-NOISING TRIGGER =====
                # Check if there are difficult tokens in UNMASKED positions
                if (renoising_count < MAX_RENOISING and 
                    t < self.diffusion_steps * 0.8 and  # Not too early
                    t > self.diffusion_steps * 0.2):     # Not too late
                    
                    # Get intro scores for unmasked target tokens
                    target_positions = init_maskable_mask[0]
                    unmasked_positions = ~maskable_mask[0] & target_positions
                    
                    if unmasked_positions.any():
                        unmasked_intro = intro_scores[0, unmasked_positions]
                        max_difficulty = unmasked_intro.max().item()
                        
                        # TRIGGER:  If any unmasked token is too difficult
                        if max_difficulty > self.renoising_thresh:
                            # Find difficult tokens
                            difficult_mask = (intro_scores[0] > self.renoising_thresh) & unmasked_positions
                            num_difficult = difficult_mask.sum().item()
                            
                            if num_difficult > 0:
                                # RE-NOISE: Jump back and re-mask difficult tokens
                                jump_back = min(self.renoising_steps, t)
                                new_t = t + jump_back
                                
                                # Re-mask difficult tokens
                                new_maskable = maskable_mask.clone()
                                new_maskable[0, difficult_mask] = True
                                maskable_mask = new_maskable
                                
                                # Re-apply masking
                                xt = x0.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                                
                                # Update timestep
                                t = new_t
                                renoising_count += 1
                                
                                flops_counter.count_renoising(batch_size, seq_len)
                                
                                # Continue from new timestep
                                continue
                
                # TopK schedule
                rate = (t / self.diffusion_steps) ** 0.6
                cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
                
                noise_scale = 0.5
                gumbel = -torch.log(-torch. log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                _scores = _scores_for_topk + noise_scale * rate * gumbel
                flops_counter.count_topk_masking(batch_size, seq_len)
                
                sorted_idx = _scores.sort(-1)[0]
                cutoff_val = sorted_idx. gather(dim=-1, index=cutoff_len)
                mask_sched = _scores < cutoff_val
                
                # Thresholds
                relaxed_conf_thresh = max(0.6, self.conf_thresh - 0.3 * (1 - rate))
                relaxed_intro_thresh = self.intro_thresh + 0.3 * (1 - rate)
                
                is_confident = confidence > relaxed_conf_thresh
                is_easy = intro_scores < relaxed_intro_thresh
                
                # Masking decision
                if t > (self.diffusion_steps * 0.75):
                    new_mask = mask_sched & init_maskable_mask
                else:
                    should_unmask = is_confident | is_easy
                    new_mask = mask_sched & (~should_unmask) & init_maskable_mask
                
                xt = x0.masked_fill(new_mask, self.tokenizer.mask_token_id)
                maskable_mask = new_mask
                
                # Early stopping
                if steps_taken >= MIN_STEPS and t > 0:
                    num_masks_left = maskable_mask.sum().item()
                    
                    if num_masks_left == 0:
                        early_stopped = True
                        stop_reason = f"no_masks@{steps_taken}(renoise={renoising_count})"
                        break
                    
                    if 3 < num_masks_left < 8 and t > 3:
                        target_positions = init_maskable_mask[0]
                        target_conf = confidence[0, target_positions]
                        target_intro = intro_scores[0, target_positions]
                        
                        avg_conf = target_conf.mean().item()
                        max_intro = target_intro.max().item()
                        
                        if avg_conf > 0.95 and max_intro < 0.15:
                            skip_steps = max(1, min(t - 2, 2))
                            t -= skip_steps
                            steps_taken += (skip_steps - 1)
                            early_stopped = True
                            stop_reason = f"skip@{steps_taken}(n={num_masks_left},renoise={renoising_count})"
                            continue
                    
                    if t == 0:
                        new_mask = torch.zeros_like(maskable_mask)
                    
                    xt = x0.masked_fill(new_mask, self.tokenizer.mask_token_id)
                    maskable_mask = new_mask
            
            t -= 1
        
        return xt, steps_taken, early_stopped, stop_reason, renoising_count

    def run_battle(self, save_details=None):
        results = {
            "Static":    {"correct": 0, "valid": 0, "steps": 0, "time": 0, "flops": 0},
            "Adaptive": {"correct": 0, "valid": 0, "steps": 0, "time":  0, "flops": 0, 
                        "early_stops": 0, "renoising_total": 0}
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
            
            input_ids = self.tokenizer.encode(quiz_digits) + [self.tokenizer.sep_token_id]
            src_len = len(input_ids)
            
            cutoff_len = 164
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
            input_ids = input_ids[:cutoff_len]
            
            x = torch.tensor([input_ids]).to(self.device)
            src_mask = torch.zeros(cutoff_len, dtype=torch. bool).to(self.device)
            src_mask[:src_len] = True
            src_mask = src_mask.unsqueeze(0)
            
            # STATIC
            flops_counter_s = FLOPsCounter()
            start = time.time()
            out_s, steps_s = self.solve_static(x, src_mask, flops_counter_s)
            time_s = time.time() - start
            
            target_ids_s = out_s[0, src_len: ].cpu().tolist()
            pred_s_decode = self.tokenizer.decode(target_ids_s, skip_special_tokens=True)
            pred_s_str = ''.join(c for c in pred_s_decode if c.isdigit())[:81]
            
            is_correct_s = (pred_s_str == gt_solution)
            is_valid_s = verify_sudoku(pred_s_str)
            
            # ADAPTIVE WITH RE-NOISING
            flops_counter_a = FLOPsCounter()
            start = time.time()
            out_a, steps_a, early_stopped, stop_reason, renoise_count = \
                self.solve_adaptive_with_renoising(x, src_mask, flops_counter_a)
            time_a = time.time() - start
            
            target_ids_a = out_a[0, src_len:]. cpu().tolist()
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
            results["Static"]["flops"] += flops_counter_s.total_flops
            
            if is_correct_a:
                results["Adaptive"]["correct"] += 1
            if is_valid_a:
                results["Adaptive"]["valid"] += 1
            results["Adaptive"]["steps"] += steps_a
            results["Adaptive"]["time"] += time_a
            results["Adaptive"]["flops"] += flops_counter_a.total_flops
            results["Adaptive"]["renoising_total"] += renoise_count
            
            if early_stopped:
                results["Adaptive"]["early_stops"] += 1
                stop_reasons.append(stop_reason)
            
            # Debug first sample
            if total == 1:
                print("\n[DEBUG] First sample:")
                print(f"Quiz: {quiz[: 30]}...")
                print(f"GT:   {gt_solution[:30]}...")
                print(f"Static:     {pred_s_str[:30]}...  | Correct: {is_correct_s}")
                print(f"  FLOPs: {flops_counter_s.total_flops/1e9:.2f} GFLOPs")
                print(f"  Breakdown:  {flops_counter_s. get_summary()['breakdown_gflops']}")
                print(f"Adaptive+Renoise: {pred_a_str[:30]}...  | Correct: {is_correct_a} | Steps: {steps_a}/{steps_s}")
                print(f"  FLOPs:  {flops_counter_a. total_flops/1e9:.2f} GFLOPs")
                print(f"  Breakdown: {flops_counter_a.get_summary()['breakdown_gflops']}")
                print(f"  Re-noising count: {renoise_count}")
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
        avg_time_s = results['Static']['time'] / total
        avg_gflops_s = results['Static']['flops'] / total / 1e9
        print(f"  Accuracy:   {acc_s:.2%} ({results['Static']['correct']}/{total})")
        print(f"  Valid:     {val_s:.2%} ({results['Static']['valid']}/{total})")
        print(f"  Avg Steps: {avg_steps_s:.2f}")
        print(f"  Avg Time:   {avg_time_s:.3f}s")
        print(f"  Avg FLOPs: {avg_gflops_s:.2f} GFLOPs")
        
        print("\n>>> ADAPTIVE + RE-NOISING")
        acc_a = results['Adaptive']['correct'] / total
        val_a = results['Adaptive']['valid'] / total
        avg_steps_a = results['Adaptive']['steps'] / total
        avg_time_a = results['Adaptive']['time'] / total
        avg_gflops_a = results['Adaptive']['flops'] / total / 1e9
        avg_renoise = results['Adaptive']['renoising_total'] / total
        early_stop_rate = results['Adaptive']['early_stops'] / total
        print(f"  Accuracy:    {acc_a:.2%} ({results['Adaptive']['correct']}/{total})")
        print(f"  Valid:       {val_a:.2%} ({results['Adaptive']['valid']}/{total})")
        print(f"  Avg Steps:   {avg_steps_a:.2f}")
        print(f"  Avg Time:    {avg_time_a:.3f}s")
        print(f"  Avg FLOPs:   {avg_gflops_a:.2f} GFLOPs")
        print(f"  Avg Re-noise: {avg_renoise:.2f}")
        print(f"  Early Stops: {early_stop_rate:.1%} ({results['Adaptive']['early_stops']}/{total})")
        
        if stop_reasons: 
            from collections import Counter
            reason_counts = Counter(stop_reasons)
            print(f"\n  Stop reasons:")
            for reason, count in reason_counts.most_common(10):
                print(f"    {reason}: {count}")
        
        print(f"\n>>> COMPARISON")
        speedup = avg_steps_s / avg_steps_a if avg_steps_a > 0 else 0
        time_speedup = avg_time_s / avg_time_a if avg_time_a > 0 else 0
        flops_reduction = (1 - avg_gflops_a / avg_gflops_s) * 100 if avg_gflops_s > 0 else 0
        flops_speedup = avg_gflops_s / avg_gflops_a if avg_gflops_a > 0 else 0
        
        print(f"  Step Speedup:   {speedup:.2f}x ({avg_steps_s:.1f} → {avg_steps_a:.1f} steps)")
        print(f"  Time Speedup:  {time_speedup:.2f}x ({avg_time_s:.3f}s → {avg_time_a:.3f}s)")
        print(f"  FLOPs Speedup: {flops_speedup:.2f}x ({avg_gflops_s:.2f} → {avg_gflops_a:.2f} GFLOPs)")
        print(f"  FLOPs Saved:   {flops_reduction:.1f}%")
        print(f"  Acc Delta:     {acc_a - acc_s:.2%}")
        
        # Efficiency metrics
        print(f"\n>>> EFFICIENCY")
        acc_per_gflop_s = acc_s / avg_gflops_s
        acc_per_gflop_a = acc_a / avg_gflops_a
        print(f"  Accuracy per GFLOPs:")
        print(f"    Static:    {acc_per_gflop_s:.4f}")
        print(f"    Adaptive: {acc_per_gflop_a:.4f} ({acc_per_gflop_a/acc_per_gflop_s:.2f}x)")
        
        gflops_per_correct_s = avg_gflops_s / acc_s if acc_s > 0 else float('inf')
        gflops_per_correct_a = avg_gflops_a / acc_a if acc_a > 0 else float('inf')
        print(f"  GFLOPs per correct solution:")
        print(f"    Static:   {gflops_per_correct_s:.2f}")
        print(f"    Adaptive: {gflops_per_correct_a:.2f} ({gflops_per_correct_s/gflops_per_correct_a:.2f}x better)")
        
        print(f"\n>>> RE-NOISING STATISTICS")
        renoise_rate = (results['Adaptive']['renoising_total'] / total) * 100 / total if total > 0 else 0
        samples_with_renoise = sum(1 for reason in stop_reasons if 'renoise=' in reason and not reason.endswith('renoise=0'))
        print(f"  Samples with re-noising: {samples_with_renoise}/{total} ({samples_with_renoise/total:.1%})")
        print(f"  Total re-noise triggers:   {results['Adaptive']['renoising_total']}")
        print(f"  Avg re-noise per sample:   {avg_renoise:.2f}")
        
        print(f"{'='*60}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output/sudoku/mdm-5m-sudoku")
    parser.add_argument("--intro_path", default="introspection_net.pth")
    parser.add_argument("--csv_path", default="data/sudoku_test.csv")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--intro_thresh", type=float, default=0.7)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--renoising_thresh", type=float, default=0.8,
                       help="Introspection threshold to trigger re-noising (default: 0.8)")
    parser.add_argument("--renoising_steps", type=int, default=3,
                       help="Number of steps to jump back during re-noising (default: 3)")
    parser.add_argument("--save_details", default=None)
    args = parser.parse_args()
    
    arena = Arena(
        args.model_path, args.intro_path, args.csv_path,
        max_samples=args. max_samples,
        conf_thresh=args. conf_thresh,
        intro_thresh=args.intro_thresh,
        diffusion_steps=args.diffusion_steps,
        renoising_thresh=args. renoising_thresh,
        renoising_steps=args. renoising_steps
    )
    
    arena.run_battle(save_details=args.save_details)


if __name__ == "__main__":
    main()