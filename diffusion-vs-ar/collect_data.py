import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# Thêm đường dẫn src để import library của bạn
sys.path.insert(0, 'src')

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments

def collect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to checkpoint folder")
    parser.add_argument("--csv_path", required=True, help="Path to sudoku_train.csv")
    parser.add_argument("--save_path", default="introspection_data.pt")
    parser.add_argument("--num_samples", type=int, default=10000, help="Số lượng mẫu train")
    args = parser.parse_args()

    # ==========================================
    # 1. LOAD MODEL (Chuẩn logic dự án)
    # ==========================================
    print(f"Loading Model from {args.model_path}...")
    model_args = ModelArguments(model_name_or_path="gpt2", checkpoint_dir=args.model_path)
    # Tắt training, chỉ load để infer
    finetuning_args = FinetuningArguments(stage="mdm", finetuning_type="full")
    diffusion_args = DiffusionArguments(diffusion_steps=20, topk_decoding=True, 
                                      token_reweighting=True, time_reweighting="linear", 
                                      alpha=0.25, gamma=1)
    
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, is_trainable=False, diffusion_args=diffusion_args, stage="mdm")
    model = model.cuda().eval()
    
    # Lấy config hidden size tự động (tránh hardcode 384 hay 768)
    hidden_dim = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
    print(f"Detected Hidden Dimension: {hidden_dim}")

    # ==========================================
    # 2. CHUẨN BỊ DATA
    # ==========================================
    print(f"Reading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Nếu file csv dùng tên cột khác, hãy sửa ở đây (ví dụ 'puzzle', 'solution')
    if 'quizzes' not in df.columns:
        print("Cảnh báo: Không thấy cột 'quizzes', thử tìm cột đầu tiên...")
        quiz_col = df.columns[0]
        sol_col = df.columns[1]
    else:
        quiz_col, sol_col = 'quizzes', 'solutions'

    # Lấy mẫu ngẫu nhiên
    if len(df) > args.num_samples:
        df = df.sample(args.num_samples, random_state=42)
    
    data_buffer_x = []
    data_buffer_y = []

    # ==========================================
    # 3. VÒNG LẶP THU THẬP
    # ==========================================
    print("Start Collecting Data...")
    mask_token_id = tokenizer.mask_token_id
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sol_str = row[sol_col]
        
        # Encode Solution (Ground Truth)
        # Lưu ý: Tokenizer của bạn có thể add special tokens, ta cần lấy đúng 81 số
        sol_ids = tokenizer.encode(sol_str)
        
        # Lọc chỉ lấy các token số (bỏ qua BOS/EOS nếu có)
        # Giả sử tokenizer encode ra list int, ta cần convert sang tensor
        # Logic an toàn: Pad hoặc cắt về đúng độ dài model cần (thường là seq_len)
        # Ở đây ta giả lập batch size = 1
        
        # Tạo Tensor x0 (Đáp án chuẩn)
        x0 = torch.tensor([sol_ids]).cuda() # [1, L]
        
        # Chọn ngẫu nhiên 1 thời điểm t (từ 1 đến 19)
        # Ta không lấy t=0 (đã xong) và t=20 (nhiễu hoàn toàn)
        t = np.random.randint(1, 20)
        
        # Tạo nhiễu giả lập (Re-noising)
        # Mask rate phụ thuộc vào t: t càng lớn (gần 20) mask càng nhiều
        mask_prob = t / 20.0 
        mask = torch.rand_like(x0.float()) < mask_prob
        xt = x0.masked_fill(mask, mask_token_id)
        
        # Chỉ học trên những ô bị mask (những ô đã hiện số rồi thì dễ quá)
        target_mask = (xt == mask_token_id)
        
        if target_mask.sum() == 0: continue # Không có gì để học

        with torch.no_grad():
            # Forward Pass lấy Hidden State
            # Gọi trực tiếp backbone GPT2 bên trong DiffusionWrapper
            # model là DiffusionModel -> model.model là GPT2LMHeadModel -> model.model.transformer
            
            # Cách an toàn nhất để lấy hidden state từ output wrapper
            # Input của model wrapper: (input_ids, t, ...)
            t_tensor = torch.full((1,), t, device=x0.device)
            
            # Ta cần can thiệp để lấy hidden_states. 
            # Đa số model HuggingFace hỗ trợ output_hidden_states=True
            # Nhưng hàm forward của wrapper có thể không truyền arg này vào trong.
            # Nên ta gọi thẳng component bên trong:
            
            backbone = model.model # GPT2LMHeadModel
            outputs = backbone.transformer(
                inputs_embeds=backbone.transformer.wte(xt),
                return_dict=True,
                output_hidden_states=True
            )
            
            # Lấy layer cuối cùng
            last_hidden = outputs.last_hidden_state # [1, L, Hidden]
            
            # Tính Logits để xem model đoán gì
            logits = backbone.lm_head(last_hidden)
            preds = logits.argmax(dim=-1)
            
            # GÁN NHÃN (LABELING)
            # 1 = Model đoán SAI (Khó/Cần nghĩ thêm)
            # 0 = Model đoán ĐÚNG (Dễ/Tự tin)
            is_wrong = (preds != x0).long()
            
            # Chỉ lấy data tại các vị trí bị mask
            selected_hidden = last_hidden[target_mask] # [N, Hidden]
            selected_labels = is_wrong[target_mask]    # [N]
            
            data_buffer_x.append(selected_hidden.cpu().half()) # Lưu FP16 cho nhẹ RAM
            data_buffer_y.append(selected_labels.cpu())

    # ==========================================
    # 4. LƯU FILE
    # ==========================================
    print("Concatenating data...")
    X = torch.cat(data_buffer_x, dim=0)
    Y = torch.cat(data_buffer_y, dim=0)
    
    print(f"Dataset Created!")
    print(f" - Shape: {X.shape}")
    print(f" - Error Rate (Label 1 ratio): {Y.float().mean():.2%}")
    print(f" - Saving to {args.save_path}...")
    
    torch.save({"x": X, "y": Y, "hidden_dim": hidden_dim}, args.save_path)
    print("DONE.")

if __name__ == "__main__":
    collect()