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
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments

# ==========================================
# 1. INTROSPECTION NET (Giữ nguyên)
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
# 2. ARENA CLASS (V3 - FIXED)
# ==========================================
class Arena:
    def __init__(self, model_path, intro_path, csv_path):
        self.device = "cuda"
        
        print("Loading Main Diffusion Model...")
        model_args = ModelArguments(model_name_or_path="gpt2", checkpoint_dir=model_path)
        finetuning_args = FinetuningArguments(stage="mdm", finetuning_type="full")
        diffusion_args = DiffusionArguments(diffusion_steps=20, topk_decoding=True, 
                                          token_reweighting=True, time_reweighting="linear", 
                                          alpha=0.25, gamma=1)
        
        self.model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args, is_trainable=False, diffusion_args=diffusion_args, stage="mdm")
        self.model = self.model.to(self.device).eval()
        
        # --- CHECK MASK TOKEN ---
        # Đảm bảo mask token hợp lệ. Nếu tokenizer không có, ta gán thủ công (thường là 50256 hoặc 10 tùy lúc train)
        if self.tokenizer.mask_token_id is None:
            print("[WARN] Tokenizer has no mask_token_id. Defaulting to 10 (Sudoku custom) or 50256 (GPT2).")
            # Bạn có thể sửa cứng ở đây nếu biết ID lúc train, ví dụ:
            # self.tokenizer.mask_token_id = 10 
        
        print(f"Using Mask Token ID: {self.tokenizer.mask_token_id}")

        print(f"Loading Introspection Net from {intro_path}...")
        ckpt = torch.load(intro_path)
        input_dim = ckpt['input_dim']
        self.intro_net = IntrospectionNet(input_dim).to(self.device)
        self.intro_net.load_state_dict(ckpt['state_dict'])
        self.intro_net.eval()
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sample(min(200, len(self.df)), random_state=42)
        print(f"Ready to battle on {len(self.df)} samples!")

    def encode_sudoku(self, quiz_str):
        """
        Custom Encoder: Biến '0' hoặc '.' thành mask_token_id
        """
        input_ids = []
        mask_id = self.tokenizer.mask_token_id
        
        for c in quiz_str:
            if c in ['0', '.']:
                input_ids.append(mask_id)
            else:
                # Encode số bình thường (trả về list id, lấy phần tử đầu)
                # Lưu ý: Tokenizer có thể encode ra [id], cần lấy int
                t_ids = self.tokenizer.encode(c, add_special_tokens=False)
                if len(t_ids) > 0:
                    input_ids.append(t_ids[0])
                else:
                    input_ids.append(mask_id) # Fallback
                    
        return torch.tensor([input_ids]).to(self.device)

    def decode_sudoku(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        digits = ''.join(c for c in text if c.isdigit())
        return digits[:81]

    # ----------------------------------------------------------------
    # STATIC SOLVER
    # ----------------------------------------------------------------
    def solve_static(self, x, src_mask):
        init_mask = ~src_mask
        maskable_mask = init_mask.clone()
        
        for t in range(19, -1, -1):
            with torch.no_grad():
                if t == 19:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                logits = self.model(xt, t_tensor)
                
                # --- FIX: REMOVE SHIFT ---
                # Với Diffusion Bidirectional, h_i dự đoán x_i. Không shift!
                # logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1) # <--- DELETE THIS
                
                # TopK Schedule
                probs = torch.softmax(logits, dim=-1)
                x0_preds = probs.argmax(dim=-1)
                
                rate = t / 20.0
                cutoff_len = (init_mask.sum(1, keepdim=True) * rate).long()
                
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,:,self.tokenizer.vocab_size:] = -1000
                x0_scores, _ = scores.max(-1)
                
                _scores_for_topk = x0_scores.masked_fill(~init_mask, 1000.0)
                sorted_idx = _scores_for_topk.sort(-1)[0]
                cutoff_val = sorted_idx.gather(dim=-1, index=cutoff_len)
                
                to_mask = (_scores_for_topk < cutoff_val) & init_mask
                
                xt = x0_preds.clone()
                xt[to_mask] = self.tokenizer.mask_token_id
        
        return xt, 20

    # ----------------------------------------------------------------
    # ADAPTIVE SOLVER
    # ----------------------------------------------------------------
    def solve_adaptive(self, x, src_mask):
        steps_taken = 0
        init_mask = ~src_mask
        maskable_mask = init_mask.clone()
        
        CONF_THRESH = 0.95   
        INTRO_THRESH = 0.3 # IntroNet > 0.3 là Khó
        
        t = 19
        while t >= 0:
            steps_taken += 1
            with torch.no_grad():
                if t == 19:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                
                # Get Hidden & Logits
                backbone = self.model.model
                outputs = backbone.transformer(
                    inputs_embeds=backbone.transformer.wte(xt),
                    return_dict=True, output_hidden_states=True
                )
                hidden = outputs.last_hidden_state
                logits = backbone.lm_head(hidden)
                # --- FIX: REMOVE SHIFT ---
                # logits = torch.cat(...) <--- DELETE
                
                probs = torch.softmax(logits, dim=-1)
                confidence, x0_preds = probs.max(dim=-1)
                
                # Introspection
                b, s, h = hidden.shape
                hidden_flat = hidden.float().view(-1, h)
                intro_scores = self.intro_net(hidden_flat).view(b, s)
                
                # Hybrid Update
                rate = t / 20.0
                cutoff_len = (init_mask.sum(1, keepdim=True) * rate).long()
                
                scores = torch.log_softmax(logits, dim=-1)
                scores[:,:,self.tokenizer.vocab_size:] = -1000
                x0_scores, _ = scores.max(-1)
                _scores_for_topk = x0_scores.masked_fill(~init_mask, 1000.0)
                
                sorted_idx = _scores_for_topk.sort(-1)[0]
                cutoff_val = sorted_idx.gather(dim=-1, index=cutoff_len)
                
                mask_sched = (_scores_for_topk < cutoff_val)
                mask_conf = (intro_scores > INTRO_THRESH) | (confidence < CONF_THRESH)
                
                # --- SAFETY RAIL ---
                # 5 bước đầu (19->15) bắt buộc tuân thủ TopK để tạo khung
                if t > 14:
                    new_mask = mask_sched & init_mask
                else:
                    new_mask = mask_sched & mask_conf & init_mask
                
                xt = x0_preds.clone()
                xt[new_mask] = self.tokenizer.mask_token_id
                maskable_mask = new_mask
                
                if maskable_mask.sum() == 0:
                    break
            t -= 1 
            
        return xt, steps_taken

    def run_battle(self):
        results = {"Static": {"correct": 0, "steps": 0, "time": 0},
                   "Adaptive": {"correct": 0, "steps": 0, "time": 0}}
        total = 0
        debug_printed = False 

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            if 'quizzes' in row: quiz, sol = row['quizzes'], row['solutions']
            else: quiz, sol = row[0], row[1]
            
            # --- FIX: Custom Encode ---
            x = self.encode_sudoku(quiz)
            if x.size(1) != 81: continue # Skip bad data
            
            # Mask Correctly
            src_mask = (x != self.tokenizer.mask_token_id)
            
            # --- ROUND 1 ---
            start = time.time()
            out_s, steps_s = self.solve_static(x, src_mask)
            time_s = time.time() - start
            
            # --- ROUND 2 ---
            start = time.time()
            out_a, steps_a = self.solve_adaptive(x, src_mask)
            time_a = time.time() - start
            
            # --- COMPARE ---
            pred_s_str = self.decode_sudoku(out_s[0])
            pred_a_str = self.decode_sudoku(out_a[0])
            gt_str = sol[:81]
            
            is_correct_s = (pred_s_str == gt_str)
            is_correct_a = (pred_a_str == gt_str)
            
            if not debug_printed:
                print(f"\n[DEBUG] MaskID: {self.tokenizer.mask_token_id}")
                print(f"Quiz  : {quiz[:15]}... (Zeros count: {quiz.count('0')})")
                print(f"Masked: {(x == self.tokenizer.mask_token_id).sum().item()} tokens")
                print(f"GT    : {gt_str[:15]}...")
                print(f"Static: {pred_s_str[:15]}... (Correct: {is_correct_s})")
                print(f"Adapt : {pred_a_str[:15]}... (Steps: {steps_a})")
                debug_printed = True

            total += 1
            if is_correct_s: results["Static"]["correct"] += 1
            results["Static"]["steps"] += steps_s
            results["Static"]["time"] += time_s
            
            if is_correct_a: results["Adaptive"]["correct"] += 1
            results["Adaptive"]["steps"] += steps_a
            results["Adaptive"]["time"] += time_a

        print("\n" + "="*50)
        print(" FINAL BATTLE RESULTS (V3) ")
        print("="*50)
        print(f"Total: {total}")
        
        acc_s = results['Static']['correct']/total
        acc_a = results['Adaptive']['correct']/total
        
        print(f"\n>>> STATIC: Acc {acc_s:.2%} | Steps {results['Static']['steps']/total:.1f}")
        print(f">>> ADAPT : Acc {acc_a:.2%} | Steps {results['Adaptive']['steps']/total:.1f}")
        
        if results['Adaptive']['steps'] > 0:
            speedup = results['Static']['steps'] / results['Adaptive']['steps']
            print(f"\nSPEEDUP: {speedup:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--intro_path", default="introspection_net.pth")
    parser.add_argument("--csv_path", required=True)
    args = parser.parse_args()
    
    arena = Arena(args.model_path, args.intro_path, args.csv_path)
    arena.run_battle()