import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Thêm đường dẫn src để import được llmtuner
sys.path.insert(0, 'src')

from llmtuner.tuner. core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments

# ==========================================
# 1. CÁC HÀM PHỤ TRỢ
# ==========================================
def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    if stochastic:   
        gumbel_noise = -torch.log(-torch. log(torch.rand_like(scores) + 1e-8) + 1e-8)
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
    
    cutoff_len = (init_maskable_mask. sum(1, keepdim=True) * rate).long()
    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

    if topk_mode.startswith("stochastic"):
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
# 2. ENTROPY VISUALIZER CLASS
# ==========================================
class EntropyVisualizer:
    def __init__(self, model_path, model_config="model_config_tiny", 
                 diffusion_steps=20, decoding_strategy="stochastic0.5-linear"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_steps = diffusion_steps
        self.decoding_strategy = decoding_strategy
        
        print(f"Loading model from {model_path}...")
        
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
        
        self.model = self. model.to(self.device).eval()
        print("Model loaded successfully!\n")

    def collect_entropy(self, x, src_mask, src_len):
        """
        Chạy inference và thu thập entropy từng bước denoise
        Returns:  numpy array [Steps, 81]
        """
        self.model.eval()
        attention_mask = torch.ones_like(x)
        batch_size = x.size(0)
        init_maskable_mask = maskable_mask = ~src_mask
        
        entropy_history = []

        print(f"Collecting entropy across {self.diffusion_steps} denoising steps...")
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                # Initialize noise at first step
                if t == self. diffusion_steps - 1:
                    xt = x. masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                # Forward pass
                t_tensor = torch.full((batch_size,), t, device=x.device)
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                
                # Shift logits
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Calculate entropy:  H(X) = -Σ p(x) log p(x)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                entropy_full = -torch.sum(probs * log_probs, dim=-1)  # [1, Seq_Len]
                
                # Extract target entropy (81 Sudoku cells)
                target_entropy = entropy_full[0, src_len :  src_len + 81]
                entropy_history.append(target_entropy.cpu().numpy())

                # Continue denoising process
                scores = log_probs. clone()
                scores[:, : , self.tokenizer.vocab_size: ] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    xt = topk_decoding(x0, x0_scores, self. decoding_strategy,
                                     init_maskable_mask, t, self.diffusion_steps,
                                     self.tokenizer.mask_token_id)
                else:
                    xt = x0
        
        return np.array(entropy_history)  # [Steps, 81]

    def plot_entropy_with_sudoku(self, entropy_matrix, sudoku_string, output_path):
        """
        Vẽ heatmap entropy với bảng Sudoku dẹt ngang bên dưới
        
        Args: 
            entropy_matrix: [Steps, 81] array
            sudoku_string:  Chuỗi 81 ký tự Sudoku
            output_path: Đường dẫn lưu file
        """
        # Parse Sudoku
        numbers = ''.join(c for c in sudoku_string if c.isdigit())
        sudoku_array = np.array([int(c) for c in numbers])
        
        # Create figure với 2 subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Tạo grid layout:  heatmap chiếm 80%, sudoku chiếm 20%
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        
        # ============= SUBPLOT 1: ENTROPY HEATMAP =============
        ax1 = fig.add_subplot(gs[0])
        
        im = ax1.imshow(entropy_matrix, cmap="YlOrRd", aspect='auto', 
                       vmin=0, vmax=np.percentile(entropy_matrix, 98))
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        
        # Trục
        ax1.set_xlabel("Position", fontsize=12)
        ax1.set_ylabel("Step", fontsize=12)
        
        # ============= SUBPLOT 2: SUDOKU STRIP =============
        ax2 = fig.add_subplot(gs[1])
        
        ax2.set_xlim(0, 81)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('auto')
        
        # Vẽ 81 ô Sudoku dẹt ngang
        for i in range(81):
            # Tính toán màu dựa trên block 3x3 (để dễ phân biệt)
            row = i // 9
            col = i % 9
            block_row = row // 3
            block_col = col // 3
            
            # Alternate colors cho các block 3x3
            if (block_row + block_col) % 2 == 0:
                color = '#f0f0f0'
            else:
                color = 'white'
            
            rect = Rectangle((i, 0), 1, 1, linewidth=0.5, 
                           edgecolor='gray', facecolor=color)
            ax2.add_patch(rect)
            
            # Vẽ số (hoặc để trống nếu là 0)
            num = sudoku_array[i]
            if num != 0:
                ax2.text(i + 0.5, 0.5, str(num),
                       ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')
        
        # Vẽ đường kẻ đậm cho các block 3x3 (mỗi 9 ô)
        for i in range(0, 82, 9):
            ax2.axvline(i, color='black', linewidth=2)
        
        # Viền ngoài
        ax2.axhline(0, color='black', linewidth=2)
        ax2.axhline(1, color='black', linewidth=2)
        
        # Remove ticks
        ax2.set_xticks(range(0, 81, 9))
        ax2.set_xticklabels([f"{i}-{i+8}" for i in range(0, 81, 9)], fontsize=9)
        ax2.set_yticks([])
        ax2.set_ylabel("Sudoku", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu entropy heatmap + sudoku tại: {output_path}")
        plt.close()

    def run_visualization(self, problem, output_path="entropy_heatmap. png", cutoff_len=164):
        """
        Chạy visualization entropy heatmap với Sudoku strip
        """
        # Prepare data
        numbers = ''.join(c for c in problem if c.isdigit())
        if len(numbers) != 81:
            raise ValueError("Input must be 81 digits")
        
        src_ids = self.tokenizer.encode(numbers) + [self.tokenizer.sep_token_id]
        src_len = len(src_ids)
        
        input_ids = src_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
        input_ids = input_ids[:cutoff_len]
        
        src_mask = torch.zeros(cutoff_len, dtype=torch.bool).to(self.device)
        src_mask[: src_len] = True
        
        x = torch.tensor([input_ids]).to(self.device)
        src_mask = src_mask.unsqueeze(0)
        
        # Collect entropy
        entropy_matrix = self.collect_entropy(x, src_mask, src_len)
        
        # Plot entropy + sudoku
        self.plot_entropy_with_sudoku(entropy_matrix, problem, output_path)
        
        print(f"✅ HOÀN THÀNH!  Đã lưu tại: {output_path}")

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse. ArgumentParser(description="Visualize Entropy Heatmap with Sudoku")
    parser.add_argument("--model_path", required=True, help="Path to checkpoint folder")
    parser.add_argument("--output", default="entropy_heatmap.png", help="Output file path")
    parser.add_argument("--puzzle", default=None, help="81-digit Sudoku string (0=empty)")
    args = parser.parse_args()

    # Hard Sudoku example
    HARD_SUDOKU = "000000000000003085001020000000507000004000100090000000500000073002010000000040009"
    
    puzzle = args. puzzle if args.puzzle else HARD_SUDOKU

    print("🎯 Starting Entropy Heatmap Visualization...")
    print(f"Puzzle:  {puzzle}\n")

    visualizer = EntropyVisualizer(args.model_path)
    visualizer.run_visualization(puzzle, args.output)