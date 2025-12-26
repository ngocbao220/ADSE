"""
FIXED VERSION - Visualization cho báo cáo
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Dict, Tuple
import sys
import os

sys.path.insert(0, 'src')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# ============================================
# Utility functions (same as before)
# ============================================

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """Helper function for topk decoding"""
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
    """Top-k decoding with various scheduling strategies"""
    topk_mode, schedule = decoding_strategy.split("-")
    
    if schedule == "linear":
        rate = t / max_step
    elif schedule == "cosine":
        rate = np.cos((max_step - t) / max_step * np.pi * 0.5)
    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")
    
    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
    
    if topk_mode. startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic":
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError(f"Unknown topk_mode: {topk_mode}")
    
    masked_to_noise = lowest_k_mask
    if isinstance(noise, torch.Tensor):
        xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
    elif isinstance(noise, (int, float)):
        xt = x0.masked_fill(masked_to_noise, noise)
    else:
        raise NotImplementedError(f"Unknown noise type: {type(noise)}")
    
    return xt


def verify_sudoku(problem, solution):
    """Verify sudoku solution"""
    prob = ''.join(c for c in problem if c.isdigit())
    sol = ''.join(c for c in solution if c.isdigit())
    
    if len(prob) != 81 or len(sol) != 81:
        return False
    
    # Check that solution matches problem where problem has digits
    for i in range(81):
        if prob[i] != '0' and prob[i] != sol[i]: 
            return False
    
    # Check rows, columns, and 3x3 boxes
    for i in range(9):
        row = [int(sol[i*9+j]) for j in range(9)]
        if len(set(row)) != 9 or sum(row) != 45:
            return False
        col = [int(sol[j*9+i]) for j in range(9)]
        if len(set(col)) != 9 or sum(col) != 45:
            return False
        br, bc = (i//3)*3, (i%3)*3
        box = [int(sol[r*9+c]) for r in range(br,br+3) for c in range(bc,bc+3)]
        if len(set(box)) != 9 or sum(box) != 45:
            return False
    return True


# ============================================
# Visualizer Class
# ============================================

class SudokuVisualizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def create_comprehensive_figure(
        self,
        problem: str,
        solution: str,
        entropy_history: np.ndarray,
        attention_history: List[np.ndarray],
        steps_used: int,
        save_path: str = "sudoku_analysis.png"
    ):
        """Create comprehensive 3-panel figure"""
        fig = plt.figure(figsize=(18, 6), constrained_layout=True)  # Use constrained_layout instead of tight_layout
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # 1. Sudoku Puzzle
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_sudoku_grid(ax1, problem, solution)
        
        # 2. Entropy Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_entropy_heatmap(ax2, entropy_history, steps_used)
        
        # 3. Adaptive Attention
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_attention_focus(ax3, problem, attention_history, entropy_history)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
        
        # Also create individual high-res versions
        self._save_individual_figures(
            problem, solution, entropy_history, 
            attention_history, steps_used
        )
    
    def _plot_sudoku_grid(self, ax, problem:  str, solution: str):
        """Figure 1: Sudoku grid"""
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title('Sudoku Puzzle & Solution', fontsize=14, fontweight='bold', pad=20)
        
        # Draw grid lines
        for i in range(10):
            lw = 3 if i % 3 == 0 else 1
            ax.plot([i, i], [0, 9], 'k-', linewidth=lw)
            ax.plot([0, 9], [i, i], 'k-', linewidth=lw)
        
        # Fill cells
        for idx in range(81):
            row, col = idx // 9, idx % 9
            given = problem[idx] if idx < len(problem) else '0'
            pred = solution[idx] if idx < len(solution) else '0'
            
            # Background color for predicted cells
            if given == '0':
                rect = patches.Rectangle(
                    (col, row), 1, 1,
                    linewidth=0,
                    facecolor='#E8F4F8',
                    alpha=0.5
                )
                ax.add_patch(rect)
            
            # Text
            if given != '0':
                ax.text(
                    col + 0.5, row + 0.5, given,
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='black'
                )
            elif pred != '0':
                ax.text(
                    col + 0.5, row + 0.5, pred,
                    ha='center', va='center',
                    fontsize=16,
                    color='#2E86AB',
                    fontweight='normal'
                )
        
        # Legend
        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='black', label='Given'),
            patches.Patch(facecolor='#E8F4F8', edgecolor='black', label='Predicted')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05), ncol=2)
    
    def _plot_entropy_heatmap(self, ax, entropy_history: np.ndarray, steps_used: int):
        """Figure 2: Entropy heatmap"""
        entropy_data = entropy_history[: steps_used, :]
        
        colors = ['#FFFFFF', '#FFF4E6', '#FFE5B4', '#FFB347', '#FF6B35', '#C1121F', '#590D22']
        cmap = LinearSegmentedColormap.from_list('confusion', colors, N=100)
        
        im = ax.imshow(
            entropy_data. T,
            aspect='auto',
            cmap=cmap,
            interpolation='nearest',
            vmin=0,
            vmax=np.percentile(entropy_data, 95)  # Use 95th percentile to avoid outliers
        )
        
        ax.set_xlabel('Diffusion Timestep (Start → End)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token Position (0-80)', fontsize=12, fontweight='bold')
        ax.set_title('Entropy Evolution:  Confusion Zones', fontsize=14, fontweight='bold', pad=15)
        
        xticks = np.linspace(0, steps_used-1, min(5, steps_used)).astype(int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        
        ax.set_yticks(range(0, 81, 10))
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Entropy (Uncertainty)', rotation=270, labelpad=20, fontweight='bold')
        
        ax.text(
            0.5, -0.15,
            'High Entropy = Confusion Zone (Model uncertain)',
            transform=ax.transAxes,
            ha='center', fontsize=10, style='italic', color='#C1121F'
        )
    
    def _plot_attention_focus(self, ax, problem:  str, attention_history: List[np.ndarray],
                             entropy_history: np.ndarray):
        """Figure 3: Adaptive attention focus"""
        num_steps = len(attention_history)
        frozen_count_per_step = [mask.sum() for mask in attention_history]
        
        given_mask = np.array([c != '0' for c in problem])
        
        # Plot frozen tokens
        ax_main = ax
        ax_main.plot(
            range(num_steps),
            frozen_count_per_step,
            linewidth=3,
            color='#2E86AB',
            marker='o',
            markersize=6,
            label='Frozen Tokens'
        )
        
        ax_main.fill_between(
            range(num_steps),
            frozen_count_per_step,
            alpha=0.3,
            color='#2E86AB'
        )
        
        ax_main.set_xlabel('Diffusion Step', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('# Frozen Tokens', fontsize=12, fontweight='bold', color='#2E86AB')
        ax_main.tick_params(axis='y', labelcolor='#2E86AB')
        ax_main.set_title('Adaptive Attention Focus', fontsize=14, fontweight='bold', pad=15)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Twin axis for entropy
        ax_twin = ax_main.twinx()
        
        active_entropy_per_step = []
        for step, frozen_mask in enumerate(attention_history):
            if step < len(entropy_history):
                active_mask = ~frozen_mask & ~given_mask
                if active_mask.sum() > 0:
                    avg_entropy = entropy_history[step, active_mask].mean()
                else:
                    avg_entropy = 0
                active_entropy_per_step.append(avg_entropy)
        
        ax_twin.plot(
            range(len(active_entropy_per_step)),
            active_entropy_per_step,
            linewidth=2.5,
            color='#C1121F',
            marker='s',
            markersize=5,
            linestyle='--',
            label='Avg Entropy (Active)'
        )
        
        ax_twin.set_ylabel('Avg Entropy (Active Tokens)', fontsize=12, fontweight='bold', color='#C1121F')
        ax_twin.tick_params(axis='y', labelcolor='#C1121F')
        
        lines1, labels1 = ax_main.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_main.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
        
        ax_main.text(
            0.5, -0.15,
            'Model freezes high-confidence tokens first, focuses on high-entropy regions',
            transform=ax_main.transAxes,
            ha='center', fontsize=10, style='italic', color='#555555'
        )
    
    def _save_individual_figures(self, problem, solution, entropy_history,
                                attention_history, steps_used):
        """Save individual high-res figures"""
        
        # Figure 1
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        self._plot_sudoku_grid(ax1, problem, solution)
        plt.savefig('fig1_sudoku_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ fig1_sudoku_grid.png")
        
        # Figure 2
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        self._plot_entropy_heatmap(ax2, entropy_history, steps_used)
        plt.savefig('fig2_entropy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ fig2_entropy_heatmap.png")
        
        # Figure 3
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        self._plot_attention_focus(ax3, problem, attention_history, entropy_history)
        plt.savefig('fig3_adaptive_attention.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ fig3_adaptive_attention.png")


# ============================================
# Generation with tracking
# ============================================

def generate_samples_with_tracking(tester, x, src_mask, intro_thresh=0.7, conf_thresh=0.5):
    """Generate samples with entropy and attention tracking"""
    tester.model.eval()
    attention_mask = torch.ones_like(x)
    batch_size = x.size(0)
    init_maskable_mask = maskable_mask = ~src_mask
    frozen_mask = torch.zeros_like(maskable_mask, dtype=torch. bool)
    
    entropy_history = []
    attention_history = []
    
    for t in range(tester.diffusion_steps - 1, -1, -1):
        with torch.no_grad():
            if t == tester.diffusion_steps - 1:
                xt = x.masked_fill(maskable_mask, tester.tokenizer.mask_token_id)
            
            t_tensor = torch.full((batch_size,), t, device=x.device)
            logits = tester.model(xt, t_tensor, attention_mask=attention_mask)
            logits = torch.cat([logits[:,0: 1], logits[:,:-1]], dim=1)
            
            # Calculate entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            target_entropy = entropy[0, : ].cpu().numpy()
            entropy_history.append(target_entropy)
            attention_history.append(frozen_mask[0]. cpu().numpy().copy())
            
            # Adaptive logic
            max_probs, x0 = probs.max(-1)
            high_conf_mask = (max_probs > intro_thresh) & maskable_mask & ~frozen_mask
            frozen_mask |= high_conf_mask
            
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
            
            # Early stop check
            active_mask = maskable_mask & ~frozen_mask
            if active_mask.sum() == 0:
                # Pad remaining steps with final state
                for _ in range(t):
                    entropy_history.append(target_entropy)
                    attention_history. append(frozen_mask[0]. cpu().numpy().copy())
                xt = x0
                break
            
            if t > 0:
                scores = torch.log(max_probs + 1e-8)
                scores = scores.masked_fill(frozen_mask, 1000.0)
                xt = topk_decoding(
                    x0, scores, tester.decoding_strategy,
                    active_mask, t, tester.diffusion_steps,
                    tester.tokenizer. mask_token_id
                )
            else:
                xt = x0
    
    entropy_history = np.array(entropy_history)
    return xt, entropy_history, attention_history


# ============================================
# Main visualization function
# ============================================

def create_paper_visualization(tester, problem, save_dir="paper_figures",
                               intro_thresh=0.7, conf_thresh=0.5):
    """Create visualization for paper"""
    os.makedirs(save_dir, exist_ok=True)
    
    numbers = ''. join(c for c in problem if c.isdigit())
    if len(numbers) != 81:
        raise ValueError(f"Problem must have 81 digits, got {len(numbers)}")
    
    # Encode
    src_ids = tester.tokenizer.encode(numbers) + [tester.tokenizer.sep_token_id]
    src_len = len(src_ids)
    cutoff_len = 164
    
    input_ids = src_ids + [tester.tokenizer.pad_token_id] * (cutoff_len - src_len)
    input_ids = input_ids[: cutoff_len]
    
    src_mask = torch.zeros(cutoff_len, dtype=torch. bool).to(tester.device)
    src_mask[: src_len] = True
    
    x = torch.tensor([input_ids]).to(tester.device)
    src_mask = src_mask.unsqueeze(0)
    
    print(f"\n{'='*60}")
    print(f"Generating solution with tracking...")
    print(f"  Problem: {numbers[: 30]}...")
    print(f"  Source length: {src_len} tokens")
    print(f"  intro_thresh={intro_thresh}, conf_thresh={conf_thresh}")
    print(f"{'='*60}\n")
    
    # Generate
    xt, entropy_history, attention_history = generate_samples_with_tracking(
        tester, x, src_mask, intro_thresh=intro_thresh, conf_thresh=conf_thresh
    )
    
    # Decode - FIXED: Only decode target portion
    target_ids = xt[0, src_len: ].cpu().tolist()
    target_decode = tester.tokenizer.decode(target_ids, skip_special_tokens=True)
    
    # Extract EXACTLY 81 digits
    solution_digits = ''.join(c for c in target_decode if c.isdigit())
    solution = solution_digits[:81]  # Take first 81 digits
    
    # Pad if needed
    if len(solution) < 81:
        print(f"⚠️  Warning: Only got {len(solution)} digits, padding with 0s")
        solution = solution + '0' * (81 - len(solution))
    
    print(f"✅ Solution generated:  {len(solution)} digits")
    print(f"   First 30: {solution[:30]}...")
    print(f"   Entropy history: {entropy_history. shape}")
    print(f"   Steps used: {len(entropy_history)}")
    
    # Verify
    is_valid = verify_sudoku(numbers, solution)
    print(f"   Valid: {is_valid} {'✅' if is_valid else '❌'}")
    
    # Extract target token entropy (81 positions)
    target_entropy = entropy_history[: , : 81]
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    visualizer = SudokuVisualizer(tester. tokenizer)
    
    save_path = f"{save_dir}/comprehensive_analysis.png"
    visualizer. create_comprehensive_figure(
        problem=numbers,
        solution=solution,
        entropy_history=target_entropy,
        attention_history=[mask[: 81] for mask in attention_history],
        steps_used=len(entropy_history),
        save_path=save_path
    )
    
    print(f"\n{'='*60}")
    print(f"✅ All figures saved in:  {save_dir}/")
    print(f"{'='*60}\n")
    
    return solution, entropy_history, attention_history


# ============================================
# Main execution
# ============================================

if __name__ == "__main__":
    from llmtuner.tuner. core import load_model_and_tokenizer
    from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments
    
    print("="*60)
    print("PAPER VISUALIZATION GENERATOR")
    print("="*60)
    
    # Configuration
    model_path = "output/sudoku/mdm-5m-sudoku"
    model_config = "model_config_tiny"
    diffusion_steps = 20
    
    print(f"\nLoading model from: {model_path}")
    
    # Load model
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
        decoding_strategy="stochastic0. 5-linear"
    )
    
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, is_trainable=False,
        diffusion_args=diffusion_args, stage="mdm"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"✅ Model loaded on {device}")
    
    # Create tester wrapper
    class TesterWrapper:
        def __init__(self, model, tokenizer, device, diffusion_steps, decoding_strategy):
            self.model = model
            self.tokenizer = tokenizer
            self. device = device
            self.diffusion_steps = diffusion_steps
            self.decoding_strategy = decoding_strategy
    
    tester = TesterWrapper(model, tokenizer, device, diffusion_steps, "stochastic0.5-linear")
    
    # Example sudoku
    hard_sudoku = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
    
    solution, entropy_hist, attention_hist = create_paper_visualization(
        tester, hard_sudoku, save_dir="paper_figures",
        intro_thresh=0.7, conf_thresh=0.5
    )