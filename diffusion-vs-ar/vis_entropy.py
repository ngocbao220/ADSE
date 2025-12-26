import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm đường dẫn src để import được llmtuner
sys.path.insert(0, 'src')

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments

# ==========================================
# 1. CÁC HÀM PHỤ TRỢ (GIỮ NGUYÊN TỪ CODE CỦA BẠN)
# ==========================================
def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    if stochastic:  
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:  
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
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
    
    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
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
# 2. CLASS VISUALIZER (CẢI TIẾN TỪ SUDOKU TESTER)
# ==========================================
class EntropyVisualizer:
    def __init__(self, model_path, model_config="model_config_tiny", 
                 diffusion_steps=20, decoding_strategy="stochastic0.5-linear"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_steps = diffusion_steps
        self.decoding_strategy = decoding_strategy
        
        print(f"Loading model from {model_path}...")
        
        # Cấu hình y hệt code chạy thành công của bạn
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
        print("Model loaded successfully for Visualization!\n")

    def collect_entropy(self, x, src_mask, src_len):
        """
        Chạy inference và thu thập entropy từng bước
        """
        self.model.eval()
        attention_mask = torch.ones_like(x)
        batch_size = x.size(0)
        init_maskable_mask = maskable_mask = ~src_mask
        
        # Mảng lưu entropy: [Steps, 81] (Chỉ lưu 81 ô kết quả)
        entropy_history = []

        print(f"Running Diffusion to collect Entropy ({self.diffusion_steps} steps)...")
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                # 1. Tạo nhiễu ban đầu
                if t == self.diffusion_steps - 1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                # 2. Forward Pass
                t_tensor = torch.full((batch_size,), t, device=x.device)
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                
                # 3. Shift Logits (Quan trọng: Logic giống hệt code gốc)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # 4. Tính Entropy
                # Logits shape: [1, Seq_Len, Vocab]
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Entropy = - sum(p * log_p)
                # Xử lý nan/inf bằng cách clamp nhẹ hoặc mask, nhưng softmax thường ổn
                entropy_full = -torch.sum(probs * log_probs, dim=-1) # [1, Seq_Len]
                
                # 5. Cắt lấy phần TARGET (81 ô Sudoku cần giải)
                # Input structure: [Source] [SEP] [Target...] [Pad]
                # Target bắt đầu từ src_len
                target_entropy = entropy_full[0, src_len : src_len + 81]
                entropy_history.append(target_entropy.cpu().numpy())

                # 6. Tiếp tục quy trình giải (Để trajectory đúng thực tế)
                scores = log_probs.clone()
                scores[:, :, self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    xt = topk_decoding(x0, x0_scores, self.decoding_strategy,
                                     init_maskable_mask, t, self.diffusion_steps,
                                     self.tokenizer.mask_token_id)
                else:
                    xt = x0
        
        return np.array(entropy_history) # [Steps, 81]

    def run_visualization(self, problem, output_img="entropy_heatmap.png", cutoff_len=164):
        # Prepare Data (Y hệt hàm solve)
        numbers = ''.join(c for c in problem if c.isdigit())
        if len(numbers) != 81: raise ValueError("Input must be 81 digits")
        
        # Encode Source
        src_ids = self.tokenizer.encode(numbers) + [self.tokenizer.sep_token_id]
        src_len = len(src_ids)
        
        # Pad Input
        input_ids = src_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
        input_ids = input_ids[:cutoff_len]
        
        src_mask = torch.zeros(cutoff_len, dtype=torch.bool).to(self.device)
        src_mask[:src_len] = True
        
        x = torch.tensor([input_ids]).to(self.device)
        src_mask = src_mask.unsqueeze(0)
        
        # Collect Entropy
        entropy_matrix = self.collect_entropy(x, src_mask, src_len)
        
        # Vẽ Heatmap
        self.plot_heatmap(entropy_matrix, output_img)
        
    def plot_heatmap(self, entropy_matrix, output_path):
        plt.figure(figsize=(18, 8))
        
        # Vẽ Heatmap
        # Trục Y: Time Steps (Từ T=20 về 0), Trục X: 81 ô Sudoku
        sns.heatmap(entropy_matrix, cmap="Reds", vmin=0, vmax=np.percentile(entropy_matrix, 99))
        
        plt.title("Dynamic Confusion Zones: Entropy Evolution during Inference", fontsize=16)
        plt.xlabel("Sudoku Token Position (0-80)", fontsize=12)
        plt.ylabel("Diffusion Timestep (Start -> End)", fontsize=12)
        
        # Highlight vùng khó (Vùng đỏ đậm)
        plt.text(40, -1, "High Entropy = Confusion Zone", color='red', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"✅ Đã lưu biểu đồ Entropy tại: {output_path}")

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to checkpoint folder")
    parser.add_argument("--output", default="entropy_heatmap.png")
    args = parser.parse_args()

    # Mẫu Hard Sudoku (Cần mẫu khó để thấy rõ vùng đỏ)
    HARD_SUDOKU = "000000000000003085001020000000507000004000100090000000500000073002010000000040009"

    visualizer = EntropyVisualizer(args.model_path)
    visualizer.run_visualization(HARD_SUDOKU, args.output)