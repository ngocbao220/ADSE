"""
So sánh Static vs Adaptive (Intro + Re-noising) - Side by Side Visualization với Verification
Tạo GIF animation để thấy rõ sự khác biệt, re-noising events, và check kết quả
"""
import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import imageio

sys.path.insert(0, 'src')

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner. hparams import ModelArguments, FinetuningArguments, DiffusionArguments


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
    """Verify sudoku solution"""
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


def decode_sudoku(tokenizer, token_ids):
    """Decode tokens to 81-digit string"""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    digits = ''.join(c for c in text if c.isdigit())
    return digits[: 81] if len(digits) >= 81 else digits


# ==========================================
# COMPARISON VISUALIZER WITH RE-NOISING
# ==========================================
class ComparisonVisualizer:
    def __init__(self, model_path, intro_path, model_config="model_config_tiny", 
                 diffusion_steps=20, decoding_strategy="stochastic0.5-linear",
                 conf_thresh=0.95, intro_thresh=0.3,
                 renoising_thresh=0.8, renoising_steps=3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_steps = diffusion_steps
        self.decoding_strategy = decoding_strategy
        self.conf_thresh = conf_thresh
        self.intro_thresh = intro_thresh
        self.renoising_thresh = renoising_thresh
        self.renoising_steps = renoising_steps
        
        print(f"{'='*60}")
        print("COMPARISON VISUALIZER WITH RE-NOISING")
        print(f"{'='*60}")
        
        # Load Diffusion Model
        print(f"Loading diffusion model...")
        model_args = ModelArguments(
            model_name_or_path=model_config,
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
            decoding_strategy=decoding_strategy
        )
        
        self. model, self.tokenizer = load_model_and_tokenizer(
            model_args, finetuning_args, is_trainable=False,
            diffusion_args=diffusion_args, stage="mdm"
        )
        self.model = self.model.to(self. device).eval()
        print("✓ Diffusion model loaded")
        
        # Load Introspection Net
        print(f"Loading introspection net...")
        ckpt = torch.load(intro_path, map_location=self.device)
        input_dim = ckpt['input_dim']
        self.intro_net = IntrospectionNet(input_dim).to(self.device)
        self.intro_net. load_state_dict(ckpt['state_dict'])
        self.intro_net. eval()
        print(f"✓ Introspection net loaded (dim={input_dim})")
        print(f"✓ Re-noising:  thresh={renoising_thresh}, steps={renoising_steps}")
        print(f"{'='*60}\n")

    def run_static(self, x, src_mask, src_len):
        """Static solver - collect data"""
        init_maskable_mask = maskable_mask = ~src_mask
        
        entropy_hist = []
        mask_hist = []
        confidence_hist = []
        solution_hist = []
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                logits = torch.cat([logits[:,0: 1], logits[:,:-1]], dim=1)
                
                # Entropy
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                # Confidence
                confidence, _ = probs.max(dim=-1)
                
                # Continue diffusion
                scores = torch.log_softmax(logits, dim=-1)
                scores[: ,: ,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                # Decode current solution
                current_solution = decode_sudoku(self.tokenizer, x0[0, src_len:]. cpu().tolist())
                solution_hist. append(current_solution)
                
                # Store target data
                entropy_hist.append(entropy[0, src_len:src_len+81]. cpu().numpy())
                confidence_hist.append(confidence[0, src_len:src_len+81].cpu().numpy())
                mask_hist.append(maskable_mask[0, src_len:src_len+81].cpu().numpy())
                
                if t > 0:
                    # TopK schedule
                    rate = (t / self.diffusion_steps) ** 0.6
                    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                    _scores_for_topk = x0_scores. masked_fill(~init_maskable_mask, 1000.0)
                    
                    noise_scale = 0.5
                    gumbel = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                    _scores = _scores_for_topk + noise_scale * rate * gumbel
                    sorted_idx = _scores.sort(-1)[0]
                    cutoff_val = sorted_idx. gather(dim=-1, index=cutoff_len)
                    lowest_k_mask = _scores < cutoff_val
                    
                    maskable_mask = lowest_k_mask & init_maskable_mask
                    xt = x0.masked_fill(maskable_mask, self. tokenizer.mask_token_id)
                else:
                    xt = x0
                    maskable_mask = torch.zeros_like(maskable_mask)
        
        return {
            'entropy': np.array(entropy_hist),
            'confidence': np.array(confidence_hist),
            'mask': np.array(mask_hist),
            'solutions': solution_hist,
            'steps': self.diffusion_steps,
            'renoising_events': []
        }

    def run_adaptive_with_renoising(self, x, src_mask, src_len):
        """Adaptive solver with FIXED re-noising logic"""
        init_maskable_mask = maskable_mask = ~src_mask
        
        entropy_hist = []
        intro_hist = []
        mask_hist = []
        confidence_hist = []
        solution_hist = []
        renoising_events = []
        
        t = self.diffusion_steps - 1
        steps_taken = 0
        renoising_count = 0
        MIN_STEPS = max(int(self.diffusion_steps * 0.5), 8)
        MAX_RENOISING = 2
        
        renoised_timesteps = set()  # ✅ Track visited timesteps
        MIN_JUMP_DISTANCE = 2
        
        while t >= 0:
            steps_taken += 1
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x. masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
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
                logits = self.model.model. lm_head(hidden)
                logits = torch.cat([logits[:,0: 1], logits[:,:-1]], dim=1)
                
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                confidence, x0_preds = probs.max(dim=-1)
                
                b, s, h = hidden.shape
                hidden_flat = hidden.float().view(-1, h)
                intro_scores = self.intro_net(hidden_flat).view(b, s)
                
                scores = torch.log_softmax(logits, dim=-1)
                scores[: ,: ,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                current_solution = decode_sudoku(self.tokenizer, x0[0, src_len:]. cpu().tolist())
                solution_hist.append(current_solution)
                
                entropy_hist.append(entropy[0, src_len:src_len+81]. cpu().numpy())
                intro_hist.append(intro_scores[0, src_len:src_len+81].cpu().numpy())
                confidence_hist.append(confidence[0, src_len:src_len+81].cpu().numpy())
                mask_hist.append(maskable_mask[0, src_len:src_len+81]. cpu().numpy())
                
                # ===== IMPROVED RE-NOISING =====
                if (renoising_count < MAX_RENOISING and 
                    t < self.diffusion_steps * 0.7 and  # ✅ Narrower window
                    t > self.diffusion_steps * 0.3):
                    
                    target_positions = init_maskable_mask[0]
                    unmasked_positions = ~maskable_mask[0] & target_positions
                    
                    if unmasked_positions.any():
                        unmasked_intro = intro_scores[0, unmasked_positions]
                        unmasked_conf = confidence[0, unmasked_positions]
                        
                        max_difficulty = unmasked_intro.max().item()
                        avg_conf = unmasked_conf.mean().item()
                        
                        # ✅ Dynamic threshold
                        phase_ratio = 1 - (t / self.diffusion_steps)
                        adaptive_thresh = self.renoising_thresh + 0.1 * (1 - phase_ratio)
                        
                        # ✅ Stricter conditions
                        if (max_difficulty > adaptive_thresh and 
                            avg_conf < 0.75):  # ✅ Low confidence required
                            
                            difficult_mask = (intro_scores[0] > adaptive_thresh) & unmasked_positions
                            num_difficult = difficult_mask. sum().item()
                            
                            # ✅ At least 3 difficult tokens
                            if num_difficult >= 3:
                                jump_back = min(self.renoising_steps, t)
                                new_t = t + jump_back
                                
                                # ✅ Check if timestep already visited
                                if new_t not in renoised_timesteps: 
                                    # ✅ Check distance from previous re-noising
                                    too_close = any(abs(new_t - prev_t) < MIN_JUMP_DISTANCE 
                                                  for prev_t in renoised_timesteps)
                                    
                                    if not too_close:
                                        renoised_timesteps.add(new_t)
                                        
                                        renoising_events.append({
                                            'step_idx':  len(entropy_hist) - 1,
                                            'timestep': t,
                                            'new_timestep': new_t,
                                            'num_difficult':  num_difficult,
                                            'max_difficulty': max_difficulty,
                                            'avg_confidence': avg_conf,
                                            'jump_back': jump_back
                                        })
                                        
                                        # Re-mask
                                        new_maskable = maskable_mask.clone()
                                        new_maskable[0, difficult_mask] = True
                                        maskable_mask = new_maskable
                                        
                                        xt = x0.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                                        t = new_t
                                        renoising_count += 1
                                        
                                        print(f"  🔄 Re-noising:  t={t}→{new_t}, "
                                              f"difficult={num_difficult}, conf={avg_conf:.2f}")
                                        
                                        continue
                
                # ...  rest of code (TopK, early stopping) remains same ...
                
                # TopK schedule
                rate = (t / self.diffusion_steps) ** 0.6
                cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                _scores_for_topk = x0_scores. masked_fill(~init_maskable_mask, 1000.0)
                
                noise_scale = 0.5
                gumbel = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                _scores = _scores_for_topk + noise_scale * rate * gumbel
                sorted_idx = _scores.sort(-1)[0]
                cutoff_val = sorted_idx. gather(dim=-1, index=cutoff_len)
                mask_sched = _scores < cutoff_val
                
                relaxed_conf_thresh = max(0.6, self.conf_thresh - 0.3 * (1 - rate))
                relaxed_intro_thresh = self.intro_thresh + 0.3 * (1 - rate)
                
                is_confident = confidence > relaxed_conf_thresh
                is_easy = intro_scores < relaxed_intro_thresh
                should_unmask = is_confident | is_easy
                
                if t > (self.diffusion_steps * 0.75):
                    new_mask = mask_sched & init_maskable_mask
                else:
                    new_mask = mask_sched & (~should_unmask) & init_maskable_mask
                
                if t == 0:
                    new_mask = torch.zeros_like(maskable_mask)
                
                xt = x0.masked_fill(new_mask, self.tokenizer.mask_token_id)
                maskable_mask = new_mask
                
                # Early stopping (same as before)
                if steps_taken >= MIN_STEPS and t > 0:
                    num_masks_left = maskable_mask.sum().item()
                    if num_masks_left == 0:
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
                            continue
            
            t -= 1
        
        return {
            'entropy':  np.array(entropy_hist),
            'intro': np.array(intro_hist),
            'confidence': np.array(confidence_hist),
            'mask': np.array(mask_hist),
            'solutions': solution_hist,
            'steps':  steps_taken,
            'renoising_events': renoising_events
        }

    def create_comparison_gif(self, static_data, adaptive_data, problem_str, 
                              ground_truth, output_gif, duration=1.0):
        """
        Tạo GIF so sánh side-by-side với re-noising visualization
        """
        temp_dir = "temp_frames_compare"
        os.makedirs(temp_dir, exist_ok=True)
        
        max_steps = max(static_data['steps'], adaptive_data['steps'])
        
        print(f"\nCreating comparison frames...")
        print(f"Static steps: {static_data['steps']}, Adaptive steps: {adaptive_data['steps']}")
        print(f"Re-noising events: {len(adaptive_data['renoising_events'])}")
        
        # FIXED FIGURE SIZE
        FIGSIZE = (24, 14)
        DPI = 100
        
        frame_paths = []
        
        for step_idx in range(max_steps):
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            gs = fig.add_gridspec(5, 5, hspace=0.35, wspace=0.3)
            
            # Problem (shared)
            problem_grid = np.array([int(c) for c in problem_str]).reshape(9, 9)
            problem_mask = (problem_grid == 0).astype(float)
            
            ax_prob = fig.add_subplot(gs[0, 0])
            sns.heatmap(problem_mask, annot=problem_grid, fmt='d', cmap='Greys',
                       cbar=False, linewidths=0.5, ax=ax_prob, vmin=0, vmax=1)
            ax_prob.set_title("Input Problem", fontsize=12, fontweight='bold')
            
            # Ground Truth
            if ground_truth: 
                ax_gt = fig. add_subplot(gs[1, 0])
                gt_grid = np.array([int(c) for c in ground_truth]).reshape(9, 9)
                sns.heatmap(np.ones((9, 9)), annot=gt_grid, fmt='d', cmap='Greens',
                           cbar=False, linewidths=0.5, ax=ax_gt, vmin=0, vmax=1, alpha=0.3)
                ax_gt.set_title("Ground Truth ✓", fontsize=12, fontweight='bold', color='green')
            
            # Check if re-noising happened at this step
            renoise_at_step = None
            for event in adaptive_data['renoising_events']:
                if event['step_idx'] == step_idx:
                    renoise_at_step = event
                    break
            
            # ===== STATIC (Left) =====
            if step_idx < static_data['steps']:
                # Entropy
                ax_s_ent = fig.add_subplot(gs[0, 1])
                entropy_grid = static_data['entropy'][step_idx].reshape(9, 9)
                vmax = np.percentile(static_data['entropy'], 95)
                sns.heatmap(entropy_grid, cmap='YlOrRd', cbar=True, linewidths=0.5,
                           ax=ax_s_ent, vmin=0, vmax=vmax)
                ax_s_ent.set_title(f"STATIC:  Entropy (Step {step_idx+1})", 
                                  fontsize=12, fontweight='bold', color='blue')
                
                # Mask
                ax_s_mask = fig.add_subplot(gs[1, 1])
                mask_grid = static_data['mask'][step_idx].reshape(9, 9).astype(float)
                sns.heatmap(mask_grid, cmap='RdYlGn_r', cbar=False, linewidths=0.5,
                           ax=ax_s_mask, vmin=0, vmax=1)
                num_masked = mask_grid.sum()
                ax_s_mask. set_title(f"STATIC: Mask ({int(num_masked)}/81)", fontsize=12)
                
                # Confidence
                ax_s_conf = fig.add_subplot(gs[2, 1])
                conf_grid = static_data['confidence'][step_idx].reshape(9, 9)
                sns.heatmap(conf_grid, cmap='Greens', cbar=True, linewidths=0.5,
                           ax=ax_s_conf, vmin=0, vmax=1)
                ax_s_conf.set_title("STATIC: Confidence", fontsize=12)
                
                # Solution
                ax_s_sol = fig.add_subplot(gs[3, 1])
                solution = static_data['solutions'][step_idx]
                if len(solution) == 81:
                    sol_grid = np.array([int(c) if c. isdigit() else 0 for c in solution]).reshape(9, 9)
                    is_valid = verify_sudoku(solution)
                    is_correct = (solution == ground_truth) if ground_truth else False
                    
                    color = 'green' if is_correct else ('orange' if is_valid else 'red')
                    status = "✅ CORRECT" if is_correct else ("⚠️ VALID" if is_valid else "❌ INVALID")
                    
                    sns.heatmap(np.ones((9, 9)), annot=sol_grid, fmt='d', 
                               cmap='Blues', cbar=False, linewidths=0.5, 
                               ax=ax_s_sol, vmin=0, vmax=1, alpha=0.2)
                    ax_s_sol.set_title(f"STATIC Solution: {status}", 
                                      fontsize=11, fontweight='bold', color=color)
            else:
                # Completed
                for row, gs_pos in [(0, gs[0,1]), (1, gs[1,1]), (2, gs[2,1])]:
                    ax = fig.add_subplot(gs_pos)
                    ax.text(0.5, 0.5, "STATIC\nCOMPLETED", ha='center', va='center',
                           fontsize=16, fontweight='bold', color='blue',
                           transform=ax.transAxes)
                    ax. axis('off')
                
                ax_s_sol = fig.add_subplot(gs[3, 1])
                solution = static_data['solutions'][-1]
                if len(solution) == 81:
                    sol_grid = np.array([int(c) if c.isdigit() else 0 for c in solution]).reshape(9, 9)
                    is_valid = verify_sudoku(solution)
                    is_correct = (solution == ground_truth) if ground_truth else False
                    
                    color = 'green' if is_correct else ('orange' if is_valid else 'red')
                    status = "✅ CORRECT" if is_correct else ("⚠️ VALID" if is_valid else "❌ INVALID")
                    
                    sns. heatmap(np.ones((9, 9)), annot=sol_grid, fmt='d', 
                               cmap='Greens' if is_correct else 'Oranges', 
                               cbar=False, linewidths=0.5, 
                               ax=ax_s_sol, vmin=0, vmax=1, alpha=0.3)
                    ax_s_sol.set_title(f"STATIC FINAL: {status}", 
                                      fontsize=12, fontweight='bold', color=color)
            
            # ===== ADAPTIVE (Right) =====
            if step_idx < adaptive_data['steps']:
                # Entropy
                ax_a_ent = fig.add_subplot(gs[0, 3])
                entropy_grid = adaptive_data['entropy'][step_idx].reshape(9, 9)
                vmax = np.percentile(adaptive_data['entropy'], 95)
                sns.heatmap(entropy_grid, cmap='YlOrRd', cbar=True, linewidths=0.5,
                           ax=ax_a_ent, vmin=0, vmax=vmax)
                
                # Add RE-NOISING indicator
                title = f"ADAPTIVE: Entropy (Step {step_idx+1})"
                if renoise_at_step:
                    title += f"\n🔄 RE-NOISING!  (↩{renoise_at_step['jump_back']} steps)"
                    ax_a_ent.set_title(title, fontsize=11, fontweight='bold', 
                                      color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                else:
                    ax_a_ent.set_title(title, fontsize=12, fontweight='bold', color='red')
                
                # Introspection (highlight difficult tokens if re-noising)
                ax_a_intro = fig.add_subplot(gs[0, 4])
                intro_grid = adaptive_data['intro'][step_idx].reshape(9, 9)
                sns.heatmap(intro_grid, cmap='Blues', cbar=True, linewidths=0.5,
                           ax=ax_a_intro, vmin=0, vmax=1)
                
                if renoise_at_step: 
                    ax_a_intro. set_title(f"Difficulty (Max={renoise_at_step['max_difficulty']:.2f})\n"
                                        f"⚠️ {renoise_at_step['num_difficult']} hard tokens! ",
                                        fontsize=10, fontweight='bold', color='red')
                else:
                    ax_a_intro.set_title("ADAPTIVE: Difficulty", fontsize=12)
                
                # Mask
                ax_a_mask = fig.add_subplot(gs[1, 3])
                mask_grid = adaptive_data['mask'][step_idx].reshape(9, 9).astype(float)
                sns.heatmap(mask_grid, cmap='RdYlGn_r', cbar=False, linewidths=0.5,
                           ax=ax_a_mask, vmin=0, vmax=1)
                num_masked = mask_grid.sum()
                ax_a_mask.set_title(f"ADAPTIVE: Mask ({int(num_masked)}/81)", fontsize=12)
                
                # Confidence
                ax_a_conf = fig. add_subplot(gs[2, 3])
                conf_grid = adaptive_data['confidence'][step_idx].reshape(9, 9)
                sns.heatmap(conf_grid, cmap='Greens', cbar=True, linewidths=0.5,
                           ax=ax_a_conf, vmin=0, vmax=1)
                ax_a_conf.set_title("ADAPTIVE: Confidence", fontsize=12)
                
                # Solution
                ax_a_sol = fig.add_subplot(gs[3, 3])
                solution = adaptive_data['solutions'][step_idx]
                if len(solution) == 81:
                    sol_grid = np.array([int(c) if c.isdigit() else 0 for c in solution]).reshape(9, 9)
                    is_valid = verify_sudoku(solution)
                    is_correct = (solution == ground_truth) if ground_truth else False
                    
                    color = 'green' if is_correct else ('orange' if is_valid else 'red')
                    status = "✅ CORRECT" if is_correct else ("⚠️ VALID" if is_valid else "❌ INVALID")
                    
                    sns.heatmap(np.ones((9, 9)), annot=sol_grid, fmt='d', 
                               cmap='Reds', cbar=False, linewidths=0.5, 
                               ax=ax_a_sol, vmin=0, vmax=1, alpha=0.2)
                    ax_a_sol.set_title(f"ADAPTIVE Solution: {status}", 
                                      fontsize=11, fontweight='bold', color=color)
            else:
                # Completed
                for col, gs_pos in [(3, gs[0,3]), (4, gs[0,4]), (3, gs[1,3]), (3, gs[2,3])]:
                    ax = fig.add_subplot(gs_pos)
                    if col == 3 and gs_pos == gs[0,3]:
                        ax. text(0.5, 0.5, "ADAPTIVE\nCOMPLETED\n✅ EARLY STOP", 
                               ha='center', va='center', fontsize=16, 
                               fontweight='bold', color='green',
                               transform=ax.transAxes)
                    ax.axis('off')
                
                ax_a_sol = fig.add_subplot(gs[3, 3])
                solution = adaptive_data['solutions'][-1]
                if len(solution) == 81:
                    sol_grid = np.array([int(c) if c.isdigit() else 0 for c in solution]).reshape(9, 9)
                    is_valid = verify_sudoku(solution)
                    is_correct = (solution == ground_truth) if ground_truth else False
                    
                    color = 'green' if is_correct else ('orange' if is_valid else 'red')
                    status = "✅ CORRECT" if is_correct else ("⚠️ VALID" if is_valid else "❌ INVALID")
                    
                    sns.heatmap(np.ones((9, 9)), annot=sol_grid, fmt='d', 
                               cmap='Greens' if is_correct else 'Oranges', 
                               cbar=False, linewidths=0.5, 
                               ax=ax_a_sol, vmin=0, vmax=1, alpha=0.3)
                    ax_a_sol. set_title(f"ADAPTIVE FINAL: {status}", 
                                      fontsize=12, fontweight='bold', color=color)
            
            # ===== RE-NOISING TIMELINE =====
            ax_timeline = fig.add_subplot(gs[4, : ])
            ax_timeline.set_xlim(0, max_steps)
            ax_timeline.set_ylim(0, 1)
            ax_timeline.set_xlabel("Step", fontsize=12)
            ax_timeline.set_title("🔄 Re-noising Timeline", fontsize=12, fontweight='bold')
            
            # Plot re-noising events
            for event in adaptive_data['renoising_events']:
                if event['step_idx'] <= step_idx:
                    ax_timeline.axvline(event['step_idx'], color='red', linestyle='--', linewidth=2, alpha=0.7)
                    ax_timeline.text(event['step_idx'], 0.5, f"↩{event['jump_back']}", 
                                    ha='center', fontsize=10, color='red', fontweight='bold')
            
            # Current position
            ax_timeline.axvline(step_idx, color='blue', linestyle='-', linewidth=3, alpha=0.5)
            ax_timeline.set_yticks([])
            
            # Title
            speedup = static_data['steps'] / adaptive_data['steps']
            renoise_info = f" | Re-noise: {len(adaptive_data['renoising_events'])} events" if adaptive_data['renoising_events'] else ""
            fig.suptitle(f"Static vs Adaptive (Introspection + Re-noising) - Step {step_idx+1}/{max_steps}\n"
                        f"Total:  Static={static_data['steps']} steps, Adaptive={adaptive_data['steps']} steps "
                        f"(Speedup:  {speedup:.2f}x){renoise_info}", 
                        fontsize=16, fontweight='bold')
            
            # Save frame
            frame_path = f"{temp_dir}/frame_{step_idx: 03d}.png"
            fig. savefig(frame_path, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            
            frame_paths.append(frame_path)
            
            if (step_idx + 1) % 5 == 0:
                print(f"  Generated frame {step_idx + 1}/{max_steps}")
        
        # Create GIF
        print(f"\nCreating GIF...  (duration={duration}s per frame)")
        
        images = [imageio.imread(fpath) for fpath in frame_paths]
        
        # Hold last frame
        images. extend([images[-1]] * 5)
        
        # Save GIF
        imageio.mimsave(output_gif, images, duration=duration, loop=0)
        
        print(f"✅ GIF saved:  {output_gif}")
        print(f"   Total frames: {len(frame_paths)}")

    def run_comparison(self, problem, ground_truth=None, output_gif="comparison_renoising.gif", 
                      duration=1.0, cutoff_len=164):
        """Main comparison pipeline"""
        numbers = ''.join(c for c in problem if c.isdigit())
        if len(numbers) != 81:
            raise ValueError(f"Need 81 digits, got {len(numbers)}")
        
        print(f"Problem: {numbers[: 30]}...")
        if ground_truth:
            print(f"Ground Truth: {ground_truth[:30]}...")
        
        # Encode
        src_ids = self.tokenizer.encode(numbers) + [self.tokenizer.sep_token_id]
        src_len = len(src_ids)
        
        input_ids = src_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
        input_ids = input_ids[: cutoff_len]
        
        src_mask = torch.zeros(cutoff_len, dtype=torch. bool).to(self.device)
        src_mask[: src_len] = True
        
        x = torch.tensor([input_ids]).to(self.device)
        src_mask = src_mask.unsqueeze(0)
        
        # Run both solvers
        print("\nRunning STATIC solver...")
        static_data = self.run_static(x, src_mask, src_len)
        
        print("Running ADAPTIVE solver with RE-NOISING...")
        adaptive_data = self.run_adaptive_with_renoising(x, src_mask, src_len)
        
        # Verify final solutions
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        static_final = static_data['solutions'][-1]
        adaptive_final = adaptive_data['solutions'][-1]
        
        static_valid = verify_sudoku(static_final)
        adaptive_valid = verify_sudoku(adaptive_final)
        
        static_correct = (static_final == ground_truth) if ground_truth else False
        adaptive_correct = (adaptive_final == ground_truth) if ground_truth else False
        
        print(f"Static:    {'✅ CORRECT' if static_correct else ('⚠️ VALID' if static_valid else '❌ INVALID')}")
        print(f"  Solution: {static_final[: 30]}...")
        print(f"\nAdaptive: {'✅ CORRECT' if adaptive_correct else ('⚠️ VALID' if adaptive_valid else '❌ INVALID')}")
        print(f"  Solution:  {adaptive_final[:30]}...")
        print(f"  Re-noising events: {len(adaptive_data['renoising_events'])}")
        
        for i, event in enumerate(adaptive_data['renoising_events']):
            print(f"    Event {i+1}: Step {event['step_idx']}, t={event['timestep']}, "
                  f"difficult={event['num_difficult']}, max_score={event['max_difficulty']:.2f}")
        
        if ground_truth:
            print(f"\nGround Truth: {ground_truth[:30]}...")
        
        # Create visualization
        self.create_comparison_gif(static_data, adaptive_data, numbers, 
                                   ground_truth, output_gif, duration)
        
        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE!")
        print(f"Output:  {output_gif}")
        print(f"Static:  {static_data['steps']} steps")
        print(f"Adaptive: {adaptive_data['steps']} steps")
        print(f"Speedup: {static_data['steps']/adaptive_data['steps']:.2f}x")
        print(f"Re-noising: {len(adaptive_data['renoising_events'])} events")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare Static vs Adaptive with Re-noising")
    parser.add_argument("--model_path", default="output/sudoku/mdm-5m-sudoku", help="Path to diffusion model checkpoint")
    parser.add_argument("--intro_path", default="introspection_net.pth", help="Path to introspection net")
    parser.add_argument("--output_gif", default="static_vs_adaptive_renoising.gif", help="Output GIF filename")
    parser.add_argument("--problem", default=None, help="Sudoku problem (81 digits)")
    parser.add_argument("--ground_truth", default=None, help="Ground truth solution (81 digits)")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration per frame in seconds")
    parser.add_argument("--conf_thresh", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--intro_thresh", type=float, default=0.5, help="Introspection threshold")
    parser.add_argument("--renoising_thresh", type=float, default=0.5, help="Re-noising trigger threshold")
    parser.add_argument("--renoising_steps", type=int, default=3, help="Steps to jump back during re-noising")
    parser.add_argument("--diffusion_steps", type=int, default=20, help="Number of diffusion steps")
    
    args = parser.parse_args()
    
    # Default problem
    problem = args.problem or "000000000060500004000000000020005070150008000000300000000050007800000000506100040"
    ground_truth = args.ground_truth or "715834926962571384348296751623915478159748632487362195234659817871423569596187243"
    
    visualizer = ComparisonVisualizer(
        model_path=args.model_path,
        intro_path=args.intro_path,
        conf_thresh=args.conf_thresh,
        intro_thresh=args.intro_thresh,
        diffusion_steps=args.diffusion_steps,
        renoising_thresh=args.renoising_thresh,
        renoising_steps=args.renoising_steps
    )
    
    visualizer.run_comparison(
        problem=problem,
        ground_truth=ground_truth,
        output_gif=args.output_gif,
        duration=args.duration
    )


if __name__ == "__main__":
    main()