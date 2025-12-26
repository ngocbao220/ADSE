"""
So sánh Static vs Adaptive (Intro) - Side by Side Visualization
Tạo GIF animation để thấy rõ sự khác biệt
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


def topk_decoding(x0, x0_scores, decoding_strategy, init_maskable_mask, t, max_step, noise):
    topk_mode, schedule = decoding_strategy. split("-")
    if schedule == "linear":
        rate = t / max_step
    elif schedule == "cosine":  
        rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
    else:  
        raise NotImplementedError
    
    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

    if topk_mode. startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic":
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError

    masked_to_noise = lowest_k_mask
    if isinstance(noise, torch.Tensor):
        xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
    elif isinstance(noise, (int, float)):
        xt = x0.masked_fill(masked_to_noise, noise)
    else:
        raise NotImplementedError
    return xt


# ==========================================
# COMPARISON VISUALIZER
# ==========================================
class ComparisonVisualizer:
    def __init__(self, model_path, intro_path, model_config="model_config_tiny", 
                 diffusion_steps=20, decoding_strategy="stochastic0.5-linear",
                 conf_thresh=0.95, intro_thresh=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_steps = diffusion_steps
        self.decoding_strategy = decoding_strategy
        self.conf_thresh = conf_thresh
        self.intro_thresh = intro_thresh
        
        print(f"{'='*60}")
        print("COMPARISON VISUALIZER INITIALIZATION")
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
        self. intro_net = IntrospectionNet(input_dim).to(self.device)
        self.intro_net. load_state_dict(ckpt['state_dict'])
        self.intro_net. eval()
        print(f"✓ Introspection net loaded (dim={input_dim})")
        print(f"{'='*60}\n")

    def run_static(self, x, src_mask, src_len):
        """Static solver - collect data"""
        init_maskable_mask = maskable_mask = ~src_mask
        
        entropy_hist = []
        mask_hist = []
        confidence_hist = []
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Entropy
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                # Confidence
                confidence, _ = probs.max(dim=-1)
                
                # Store target data
                entropy_hist.append(entropy[0, src_len:src_len+81].cpu().numpy())
                confidence_hist.append(confidence[0, src_len:src_len+81].cpu().numpy())
                mask_hist.append(maskable_mask[0, src_len:src_len+81]. cpu().numpy())
                
                # Continue diffusion
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,: ,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    # TopK schedule
                    rate = (t / self.diffusion_steps) ** 0.6
                    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
                    
                    noise_scale = 0.5
                    gumbel = -torch.log(-torch.log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                    _scores = _scores_for_topk + noise_scale * rate * gumbel
                    sorted_idx = _scores.sort(-1)[0]
                    cutoff_val = sorted_idx.gather(dim=-1, index=cutoff_len)
                    lowest_k_mask = _scores < cutoff_val
                    
                    xt = x0.masked_fill(lowest_k_mask, self.tokenizer.mask_token_id)
                else:
                    xt = x0
        
        return {
            'entropy': np.array(entropy_hist),
            'confidence': np.array(confidence_hist),
            'mask': np.array(mask_hist),
            'steps': self.diffusion_steps
        }

    def run_adaptive(self, x, src_mask, src_len):
        """Adaptive solver with intro - collect data"""
        init_maskable_mask = maskable_mask = ~src_mask
        
        entropy_hist = []
        intro_hist = []
        mask_hist = []
        confidence_hist = []
        
        t = self.diffusion_steps - 1
        steps_taken = 0
        MIN_STEPS = max(int(self.diffusion_steps * 0.5), 8)
        
        while t >= 0:
            steps_taken += 1
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self. tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                attention_mask = torch.ones_like(xt)
                
                # Get hidden + logits
                outputs = self.model. model. transformer(
                    inputs_embeds=self.model.model.transformer.wte(xt),
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )
                hidden = outputs. last_hidden_state
                logits = self.model.model. lm_head(hidden)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Entropy
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch. log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                # Confidence
                confidence, x0_preds = probs.max(dim=-1)
                
                # Introspection
                b, s, h = hidden.shape
                hidden_flat = hidden.float().view(-1, h)
                intro_scores = self.intro_net(hidden_flat).view(b, s)
                
                # Store data
                entropy_hist.append(entropy[0, src_len: src_len+81].cpu().numpy())
                intro_hist. append(intro_scores[0, src_len:src_len+81].cpu().numpy())
                confidence_hist.append(confidence[0, src_len:src_len+81].cpu().numpy())
                mask_hist.append(maskable_mask[0, src_len:src_len+81]. cpu().numpy())
                
                # Continue adaptive diffusion
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,:,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores. max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                # Aggressive TopK
                rate = (t / self.diffusion_steps) ** 0.6
                cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
                _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
                
                noise_scale = 0.5
                gumbel = -torch.log(-torch. log(torch.rand_like(_scores_for_topk) + 1e-8) + 1e-8)
                _scores = _scores_for_topk + noise_scale * rate * gumbel
                sorted_idx = _scores.sort(-1)[0]
                cutoff_val = sorted_idx.gather(dim=-1, index=cutoff_len)
                mask_sched = _scores < cutoff_val
                
                # Hybrid masking with intro
                relaxed_conf_thresh = max(0.6, self.conf_thresh - 0.3 * (1 - rate))
                relaxed_intro_thresh = self.intro_thresh + 0.3 * (1 - rate)
                
                is_confident = confidence > relaxed_conf_thresh
                is_easy = intro_scores < relaxed_intro_thresh
                should_unmask = is_confident | is_easy
                
                if t > (self.diffusion_steps * 0.75):
                    new_mask = mask_sched & init_maskable_mask
                else:
                    new_mask = mask_sched & (~should_unmask) & init_maskable_mask
                
                xt = x0.masked_fill(new_mask, self.tokenizer.mask_token_id)
                maskable_mask = new_mask
                
                # Early stopping
                if steps_taken >= MIN_STEPS:
                    num_masks_left = maskable_mask.sum().item()
                    if num_masks_left == 0:
                        break
                    
                    if num_masks_left < 8 and t > 2:
                        target_conf = confidence[0, init_maskable_mask[0]]
                        target_intro = intro_scores[0, init_maskable_mask[0]]
                        avg_conf = target_conf.mean().item()
                        max_intro = target_intro.max().item()
                        
                        if avg_conf > 0.92 and max_intro < 0.2:
                            skip_steps = max(1, min(t - 1, 3))
                            t -= skip_steps
                            steps_taken += (skip_steps - 1)
                            continue
            
            t -= 1
        
        return {
            'entropy': np.array(entropy_hist),
            'intro': np.array(intro_hist),
            'confidence': np. array(confidence_hist),
            'mask': np.array(mask_hist),
            'steps':  steps_taken
        }

    def create_comparison_gif(self, static_data, adaptive_data, problem_str, 
                              output_gif, duration=1.0):
        """
        Tạo GIF so sánh side-by-side - FIXED SIZE
        """
        temp_dir = "temp_frames_compare"
        os.makedirs(temp_dir, exist_ok=True)
        
        max_steps = max(static_data['steps'], adaptive_data['steps'])
        
        print(f"\nCreating comparison frames...")
        print(f"Static steps: {static_data['steps']}, Adaptive steps: {adaptive_data['steps']}")
        
        # FIXED FIGURE SIZE
        FIGSIZE = (22, 10)
        DPI = 100
        
        frame_paths = []
        
        for step_idx in range(max_steps):
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
            
            # Problem (shared)
            problem_grid = np.array([int(c) for c in problem_str]).reshape(9, 9)
            problem_mask = (problem_grid == 0).astype(float)
            
            ax_prob = fig.add_subplot(gs[0, 0])
            sns.heatmap(problem_mask, annot=problem_grid, fmt='d', cmap='Greys',
                       cbar=False, linewidths=0.5, ax=ax_prob, vmin=0, vmax=1)
            ax_prob.set_title("Input Problem", fontsize=12, fontweight='bold')
            
            # ===== STATIC (Left) =====
            if step_idx < static_data['steps']: 
                ax_s_ent = fig. add_subplot(gs[0, 1])
                entropy_grid = static_data['entropy'][step_idx].reshape(9, 9)
                vmax = np.percentile(static_data['entropy'], 95)
                sns.heatmap(entropy_grid, cmap='YlOrRd', cbar=True, linewidths=0.5,
                           ax=ax_s_ent, vmin=0, vmax=vmax)
                ax_s_ent.set_title(f"STATIC:  Entropy (Step {step_idx+1})", 
                                  fontsize=12, fontweight='bold', color='blue')
                
                ax_s_mask = fig.add_subplot(gs[1, 1])
                mask_grid = static_data['mask'][step_idx].reshape(9, 9).astype(float)
                sns.heatmap(mask_grid, cmap='RdYlGn_r', cbar=False, linewidths=0.5,
                           ax=ax_s_mask, vmin=0, vmax=1)
                num_masked = mask_grid.sum()
                ax_s_mask.set_title(f"STATIC: Mask ({int(num_masked)}/81)", fontsize=12)
                
                ax_s_conf = fig.add_subplot(gs[2, 1])
                conf_grid = static_data['confidence'][step_idx].reshape(9, 9)
                sns. heatmap(conf_grid, cmap='Greens', cbar=True, linewidths=0.5,
                           ax=ax_s_conf, vmin=0, vmax=1)
                ax_s_conf.set_title("STATIC:  Confidence", fontsize=12)
            else:
                ax_s_ent = fig.add_subplot(gs[0, 1])
                ax_s_ent.text(0.5, 0.5, "STATIC\nCOMPLETED", ha='center', va='center',
                            fontsize=20, fontweight='bold', color='blue',
                            transform=ax_s_ent. transAxes)
                ax_s_ent.axis('off')
                ax_s_mask = fig.add_subplot(gs[1, 1])
                ax_s_mask.axis('off')
                ax_s_conf = fig. add_subplot(gs[2, 1])
                ax_s_conf.axis('off')
            
            # ===== ADAPTIVE (Right) =====
            if step_idx < adaptive_data['steps']:
                ax_a_ent = fig.add_subplot(gs[0, 3])
                entropy_grid = adaptive_data['entropy'][step_idx].reshape(9, 9)
                vmax = np.percentile(adaptive_data['entropy'], 95)
                sns.heatmap(entropy_grid, cmap='YlOrRd', cbar=True, linewidths=0.5,
                           ax=ax_a_ent, vmin=0, vmax=vmax)
                ax_a_ent.set_title(f"ADAPTIVE: Entropy (Step {step_idx+1})", 
                                  fontsize=12, fontweight='bold', color='red')
                
                ax_a_intro = fig.add_subplot(gs[0, 4])
                intro_grid = adaptive_data['intro'][step_idx].reshape(9, 9)
                sns.heatmap(intro_grid, cmap='Blues', cbar=True, linewidths=0.5,
                           ax=ax_a_intro, vmin=0, vmax=1)
                ax_a_intro.set_title("ADAPTIVE: Difficulty", fontsize=12)
                
                ax_a_mask = fig.add_subplot(gs[1, 3])
                mask_grid = adaptive_data['mask'][step_idx].reshape(9, 9).astype(float)
                sns.heatmap(mask_grid, cmap='RdYlGn_r', cbar=False, linewidths=0.5,
                           ax=ax_a_mask, vmin=0, vmax=1)
                num_masked = mask_grid.sum()
                ax_a_mask.set_title(f"ADAPTIVE:  Mask ({int(num_masked)}/81)", fontsize=12)
                
                ax_a_conf = fig.add_subplot(gs[2, 3])
                conf_grid = adaptive_data['confidence'][step_idx].reshape(9, 9)
                sns.heatmap(conf_grid, cmap='Greens', cbar=True, linewidths=0.5,
                           ax=ax_a_conf, vmin=0, vmax=1)
                ax_a_conf.set_title("ADAPTIVE: Confidence", fontsize=12)
            else:
                ax_a_ent = fig.add_subplot(gs[0, 3])
                ax_a_ent.text(0.5, 0.5, "ADAPTIVE\nCOMPLETED\n✅ EARLY STOP", 
                            ha='center', va='center',
                            fontsize=20, fontweight='bold', color='green',
                            transform=ax_a_ent.transAxes)
                ax_a_ent.axis('off')
                ax_a_intro = fig.add_subplot(gs[0, 4])
                ax_a_intro.axis('off')
                ax_a_mask = fig. add_subplot(gs[1, 3])
                ax_a_mask.axis('off')
                ax_a_conf = fig.add_subplot(gs[2, 3])
                ax_a_conf.axis('off')
            
            # ===== DIFFERENCE =====
            if step_idx < min(static_data['steps'], adaptive_data['steps']):
                ax_diff = fig.add_subplot(gs[1, 2])
                static_mask = static_data['mask'][step_idx].reshape(9, 9)
                adaptive_mask = adaptive_data['mask'][step_idx].reshape(9, 9)
                diff = static_mask. astype(float) - adaptive_mask.astype(float)
                
                sns.heatmap(diff, cmap='RdBu_r', center=0, cbar=True, linewidths=0.5,
                           ax=ax_diff, vmin=-1, vmax=1)
                ax_diff.set_title("DIFFERENCE\n(Blue=Adaptive Faster)", 
                                 fontsize=11, fontweight='bold')
            else:
                ax_diff = fig.add_subplot(gs[1, 2])
                ax_diff.axis('off')
            
            # Title
            speedup = static_data['steps'] / adaptive_data['steps']
            fig.suptitle(f"Static vs Adaptive (Introspection) - Step {step_idx+1}/{max_steps}\n"
                        f"Total:  Static={static_data['steps']} steps, Adaptive={adaptive_data['steps']} steps "
                        f"(Speedup:  {speedup:.2f}x)", 
                        fontsize=16, fontweight='bold')
            
            # Save to buffer with FIXED size
            frame_path = f"{temp_dir}/frame_{step_idx:03d}.png"
            
            # CRITICAL: Use fixed size, NO bbox_inches='tight'
            fig.savefig(frame_path, dpi=DPI, format='png')
            plt.close(fig)
            
            frame_paths.append(frame_path)
            
            if (step_idx + 1) % 5 == 0:
                print(f"  Generated frame {step_idx + 1}/{max_steps}")
        
        # Create GIF
        print(f"\nCreating GIF...  (duration={duration}s per frame)")
        
        # Load all frames as numpy arrays
        images = []
        for fpath in frame_paths:
            img = imageio.imread(fpath)
            images.append(img)
        
        # Verify all same shape
        shapes = [img.shape for img in images]
        print(f"  Frame shapes: {set(shapes)}")
        
        if len(set(shapes)) > 1:
            print("  WARNING: Different shapes detected, resizing...")
            # Get max dimensions
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            
            resized = []
            for img in images:
                if img.shape[: 2] != (max_h, max_w):
                    # Pad to max size
                    padded = np.ones((max_h, max_w, img.shape[2]), dtype=img.dtype) * 255
                    h, w = img.shape[:2]
                    padded[:h, :w] = img
                    resized.append(padded)
                else:
                    resized.append(img)
            images = resized
        
        # Hold last frame
        images.extend([images[-1]] * 5)
        
        # Save GIF
        imageio.mimsave(output_gif, images, duration=duration, loop=0)
        
        print(f"✅ GIF saved: {output_gif}")
        print(f"   Total frames: {len(frame_paths)}, Final size: {images[0].shape}")
        

    def run_comparison(self, problem, output_gif="comparison.gif", 
                      duration=1.0, cutoff_len=164):
        """Main comparison pipeline"""
        numbers = ''. join(c for c in problem if c.isdigit())
        if len(numbers) != 81:
            raise ValueError(f"Need 81 digits, got {len(numbers)}")
        
        print(f"Problem: {numbers[: 30]}...")
        
        # Encode
        src_ids = self.tokenizer. encode(numbers) + [self.tokenizer.sep_token_id]
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
        
        print("Running ADAPTIVE solver...")
        adaptive_data = self.run_adaptive(x, src_mask, src_len)
        
        # Create visualization
        self.create_comparison_gif(static_data, adaptive_data, numbers, 
                                   output_gif, duration)
        
        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE!")
        print(f"Output:  {output_gif}")
        print(f"Static: {static_data['steps']} steps")
        print(f"Adaptive: {adaptive_data['steps']} steps")
        print(f"Speedup:  {static_data['steps']/adaptive_data['steps']:.2f}x")
        print(f"{'='*60}\n")


# Trong main():

def main():
    parser = argparse.ArgumentParser(description="Compare Static vs Adaptive")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--intro_path", required=True)
    parser.add_argument("--output_gif", default="static_vs_adaptive.gif")
    parser.add_argument("--problem", default=None)
    parser.add_argument("--duration", type=float, default=1.0)
    
    # ADD THESE:
    parser.add_argument("--conf_thresh", type=float, default=0.7,
                       help="Confidence threshold (lower = easier unmask)")
    parser.add_argument("--intro_thresh", type=float, default=0.5,
                       help="Introspection threshold (higher = fewer hard tokens)")
    
    args = parser.parse_args()
    
    problem = args.problem or "000000000000003085001020000000507000004000100090000000500000073002010000000040009"
    
    visualizer = ComparisonVisualizer(
        model_path=args.model_path,
        intro_path=args.intro_path,
        conf_thresh=args.conf_thresh,      # <<<< Pass from args
        intro_thresh=args.intro_thresh     # <<<< Pass from args
    )
    
    visualizer.run_comparison(
        problem=problem,
        output_gif=args.output_gif,
        duration=args.duration
    )


if __name__ == "__main__":
    main()