"""
Visualize Entropy Evolution + Introspection Net Impact trong Diffusion Process
Tạo GIF animation để thấy rõ model tập trung vào các token khó
"""
import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imageio

sys.path.insert(0, 'src')

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments


# ==========================================
# INTROSPECTION NET (Same as arena)
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
    cutoff = sorted_index.  gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def topk_decoding(x0, x0_scores, decoding_strategy, init_maskable_mask, t, max_step, noise):
    topk_mode, schedule = decoding_strategy.split("-")
    if schedule == "linear":
        rate = t / max_step
    elif schedule == "cosine":  
        rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
    else:  
        raise NotImplementedError
    
    cutoff_len = (init_maskable_mask. sum(1, keepdim=True) * rate).long()
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
# VISUALIZER CLASS
# ==========================================
class EntropyVisualizer: 
    def __init__(self, model_path, intro_path=None, model_config="model_config_tiny", 
                 diffusion_steps=20, decoding_strategy="stochastic0.5-linear"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_steps = diffusion_steps
        self.decoding_strategy = decoding_strategy
        
        print(f"{'='*60}")
        print("ENTROPY VISUALIZER INITIALIZATION")
        print(f"{'='*60}")
        print(f"Loading model from {model_path}...")
        
        # Load Diffusion Model
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
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_args, finetuning_args, is_trainable=False,
            diffusion_args=diffusion_args, stage="mdm"
        )
        
        self.model = self.model.to(self.device).eval()
        print("✓ Diffusion model loaded")
        
        # Load Introspection Net (Optional)
        self.intro_net = None
        if intro_path and os.path.exists(intro_path):
            print(f"\nLoading Introspection Net from {intro_path}...")
            ckpt = torch.load(intro_path, map_location=self. device)
            input_dim = ckpt['input_dim']
            self.intro_net = IntrospectionNet(input_dim).to(self.device)
            self.intro_net. load_state_dict(ckpt['state_dict'])
            self.intro_net.eval()
            print(f"✓ Introspection net loaded (hidden_dim={input_dim})")
        else:
            print("\n⚠️  No Introspection Net provided (will only show entropy)")
        
        print(f"{'='*60}\n")

    def collect_data(self, x, src_mask, src_len):
        """
        Chạy diffusion và thu thập entropy + introspection scores
        Returns:   
            entropy_history: [Steps, 81]
            intro_history:   [Steps, 81] (or None)
            mask_history:  [Steps, 81] (boolean)
        """
        self. model.eval()
        attention_mask = torch.ones_like(x)
        batch_size = x.size(0)
        init_maskable_mask = maskable_mask = ~src_mask
        
        entropy_history = []
        intro_history = [] if self.intro_net else None
        mask_history = []

        print(f"Collecting data through {self.diffusion_steps} steps...")
        
        for step_idx, t in enumerate(range(self. diffusion_steps - 1, -1, -1)):
            with torch.no_grad():
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((batch_size,), t, device=x.device)
                
                # Get hidden states (for introspection)
                if self.intro_net:
                    outputs = self.model. model. transformer(
                        inputs_embeds=self.model.model.transformer.wte(xt),
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True
                    )
                    hidden = outputs.last_hidden_state
                    logits = self.model.model.lm_head(hidden)
                    logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                    
                    # Introspection scores
                    b, s, h = hidden.shape
                    hidden_flat = hidden.float().view(-1, h)
                    intro_scores = self.intro_net(hidden_flat).view(b, s)
                    target_intro = intro_scores[0, src_len: src_len+81]. cpu().numpy()
                    intro_history.append(target_intro)
                else:
                    logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                    logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Compute Entropy
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy_full = -torch.sum(probs * log_probs, dim=-1)  # [1, Seq_Len]
                
                target_entropy = entropy_full[0, src_len:src_len+81]. cpu().numpy()
                entropy_history.append(target_entropy)
                
                # Track mask positions
                target_mask = maskable_mask[0, src_len:src_len+81].cpu().numpy()
                mask_history.append(target_mask)
                
                # Continue diffusion process
                scores = torch.log_softmax(logits, dim=-1)
                scores[:, :, self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    xt = topk_decoding(x0, x0_scores, self.decoding_strategy,
                                     init_maskable_mask, t, self.diffusion_steps,
                                     self.tokenizer.mask_token_id)
                else:
                    xt = x0
            
            if (step_idx + 1) % 5 == 0:
                print(f"  Processed step {step_idx + 1}/{self.diffusion_steps}")
        
        return (np.array(entropy_history), 
                np.array(intro_history) if intro_history else None,
                np.array(mask_history))

    def create_gif(self, entropy_matrix, intro_matrix, mask_matrix, output_gif, problem_str):
        """
        Tạo GIF animation cho từng timestep
        """
        frames = []
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"\nCreating animation frames...")
        
        for step_idx in range(self.diffusion_steps):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Reshape data thành 9x9 grid
            entropy_grid = entropy_matrix[step_idx].reshape(9, 9)
            mask_grid = mask_matrix[step_idx].reshape(9, 9).astype(float)
            
            # Problem grid (input)
            problem_grid = np. array([int(c) for c in problem_str]).reshape(9, 9)
            problem_mask = (problem_grid == 0).astype(float)
            
            # ===== Plot 1: Problem =====
            ax1 = axes[0, 0]
            sns.heatmap(problem_mask, annot=problem_grid, fmt='d', cmap='Greys', 
                       cbar=False, linewidths=0.5, ax=ax1, vmin=0, vmax=1)
            ax1.set_title(f"Input Problem (0 = Unknown)", fontsize=14, fontweight='bold')
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            
            # ===== Plot 2:   Entropy Heatmap =====
            ax2 = axes[0, 1]
            vmax = np.percentile(entropy_matrix, 95)
            sns.heatmap(entropy_grid, cmap='YlOrRd', cbar=True, linewidths=0.5, 
                       ax=ax2, vmin=0, vmax=vmax)
            ax2.set_title(f"Entropy (Step {step_idx+1}/{self.diffusion_steps})", 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel("High Entropy = High Uncertainty")
            
            # ===== Plot 3: Introspection (if available) =====
            ax3 = axes[1, 0]
            if intro_matrix is not None:
                intro_grid = intro_matrix[step_idx].reshape(9, 9)
                sns.heatmap(intro_grid, cmap='Blues', cbar=True, linewidths=0.5, 
                           ax=ax3, vmin=0, vmax=1)
                ax3.set_title(f"Introspection Difficulty Score", fontsize=14, fontweight='bold')
                ax3.set_xlabel("High Score = Difficult Token")
            else:
                ax3.text(0.5, 0.5, "No Introspection Net", ha='center', va='center', 
                        fontsize=16, transform=ax3.transAxes)
                ax3.set_title("Introspection (N/A)", fontsize=14)
                ax3.axis('off')
            
            # ===== Plot 4:  Mask Status =====
            ax4 = axes[1, 1]
            sns.heatmap(mask_grid, cmap='RdYlGn_r', cbar=False, linewidths=0.5, 
                       ax=ax4, vmin=0, vmax=1)
            ax4.set_title(f"Mask Status (Red = Masked)", fontsize=14, fontweight='bold')
            num_masked = mask_grid.sum()
            ax4.set_xlabel(f"Masked Tokens:  {int(num_masked)}/81")
            
            plt.suptitle(f"Diffusion Denoising Process - Step {step_idx+1}/{self.diffusion_steps}", 
                        fontsize=18, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            # Save frame
            frame_path = f"{temp_dir}/frame_{step_idx: 03d}.png"
            plt. savefig(frame_path, dpi=100, bbox_inches='tight')
            frames.append(frame_path)
            plt.close(fig)
            
            if (step_idx + 1) % 5 == 0:
                print(f"  Generated frame {step_idx + 1}/{self.diffusion_steps}")
        
        # Create GIF
        print(f"\nCreating GIF animation...")
        images = [Image.open(frame) for frame in frames]
        
        # Add final frame pause
        images.extend([images[-1]] * 5)  # Hold last frame for 5 extra frames
        
        imageio.mimsave(output_gif, images, duration=0.5, loop=0)
        
        # Cleanup
        for frame in frames: 
            os.remove(frame)
        os.rmdir(temp_dir)
        
        print(f"✅ GIF saved to:  {output_gif}")

    def create_static_summary(self, entropy_matrix, intro_matrix, output_path):
        """
        Tạo summary image (không phải GIF)
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Entropy Evolution
        ax1 = axes[0]
        sns.heatmap(entropy_matrix, cmap="YlOrRd", cbar=True, ax=ax1,
                   vmin=0, vmax=np.percentile(entropy_matrix, 95))
        ax1.set_title("Entropy Evolution (Time → Position)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Token Position (0-80)")
        ax1.set_ylabel("Diffusion Step (Start → End)")
        
        # Plot 2: Average Entropy per position
        ax2 = axes[1]
        avg_entropy = entropy_matrix.mean(axis=0).reshape(9, 9)
        sns.heatmap(avg_entropy, cmap="Reds", cbar=True, linewidths=0.5, ax=ax2,
                   annot=True, fmt='.2f')
        ax2.set_title("Average Entropy (9x9 Grid)", fontsize=14, fontweight='bold')
        
        # Plot 3: Introspection if available
        ax3 = axes[2]
        if intro_matrix is not None: 
            avg_intro = intro_matrix.mean(axis=0).reshape(9, 9)
            sns.heatmap(avg_intro, cmap="Blues", cbar=True, linewidths=0.5, ax=ax3,
                       annot=True, fmt='.2f', vmin=0, vmax=1)
            ax3.set_title("Average Difficulty (Introspection)", fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, "No Introspection Data", ha='center', va='center',
                    fontsize=16, transform=ax3.transAxes)
            ax3.axis('off')
        
        plt.suptitle("Summary: Entropy & Introspection Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Summary image saved to: {output_path}")
        plt.close()

    def run_visualization(self, problem, output_gif="entropy_animation.gif", 
                         output_summary="entropy_summary.png", cutoff_len=164):
        """
        Main visualization pipeline
        """
        # Prepare data
        numbers = ''. join(c for c in problem if c.isdigit())
        if len(numbers) != 81:
            raise ValueError(f"Input must be 81 digits, got {len(numbers)}")
        
        print(f"Problem: {numbers[: 30]}...")
        
        # Encode
        src_ids = self.tokenizer.encode(numbers) + [self.tokenizer.sep_token_id]
        src_len = len(src_ids)
        
        input_ids = src_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
        input_ids = input_ids[: cutoff_len]
        
        src_mask = torch.zeros(cutoff_len, dtype=torch.bool).to(self.device)
        src_mask[: src_len] = True
        
        x = torch.tensor([input_ids]).to(self.device)
        src_mask = src_mask.unsqueeze(0)
        
        # Collect data
        entropy_matrix, intro_matrix, mask_matrix = self.collect_data(x, src_mask, src_len)
        
        # Create visualizations
        self.create_gif(entropy_matrix, intro_matrix, mask_matrix, output_gif, numbers)
        self.create_static_summary(entropy_matrix, intro_matrix, output_summary)
        
        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"GIF:      {output_gif}")
        print(f"Summary: {output_summary}")
        print(f"{'='*60}\n")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Visualize Entropy + Introspection")
    parser.add_argument("--model_path", required=True, help="Path to diffusion model")
    parser.add_argument("--intro_path", default=None, help="Path to introspection net (optional)")
    parser.add_argument("--output_gif", default="entropy_animation.gif", help="Output GIF path")
    parser.add_argument("--output_summary", default="entropy_summary.png", help="Summary image")
    parser.add_argument("--problem", default=None, help="81-digit sudoku string")
    
    args = parser.parse_args()
    
    # Hard Sudoku example (or use custom)
    HARD_SUDOKU = args.problem or "000000000000003085001020000000507000004000100090000000500000073002010000000040009"
    
    visualizer = EntropyVisualizer(
        model_path=args.model_path,
        intro_path=args.intro_path
    )
    
    visualizer.run_visualization(
        problem=HARD_SUDOKU,
        output_gif=args. output_gif,
        output_summary=args.output_summary
    )


if __name__ == "__main__":
    main()