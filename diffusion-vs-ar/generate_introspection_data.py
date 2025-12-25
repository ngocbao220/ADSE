import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH
# ==========================================
CHECKPOINT_PATH = "output/sudoku/mdm-5m-sudoku"
CSV_DATA_PATH = "data/sudoku_train.csv" # Dùng file train gốc
OUTPUT_FILE = "mind_dataset_train.pt"
NUM_SAMPLES = 2000 # Số lượng mẫu dùng để sinh data (càng nhiều càng tốt, nhưng 2k là đủ demo)
DIFFUSION_STEPS = 20

# ==========================================
# 2. MODEL & TOKENIZER (Giữ nguyên)
# ==========================================
class DiffusionModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = model.config
        self.embed_tokens = self.model.transformer.wte 
        self.denoise_model = self.model.transformer
        for gpt2block in self.model.transformer.h:
            gpt2block.attn.bias.fill_(True)
        self.lm_head = self.model.lm_head
    
    def forward(self, input_ids, t, output_hidden_states=False):
        # Quan trọng: Cần bật output_hidden_states=True
        outputs = self.denoise_model(
            inputs_embeds=self.embed_tokens(input_ids), 
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        x = outputs.last_hidden_state
        logits = self.lm_head(x)
        
        if output_hidden_states:
            return logits, x # Trả về cả Logits và Hidden State
        return logits

class SudokuTokenizer:
    def __init__(self):
        self.vocab = {str(i): i for i in range(10)} 
        self.vocab["*"] = 10
        self.mask_token_id = 10 
    
    def encode(self, text, is_input=False):
        ids = []
        for c in text:
            if c == '0' and is_input: ids.append(10)
            else: ids.append(self.vocab.get(c, 0))
        return ids

def load_model(path):
    config = GPT2Config.from_json_file(os.path.join(path, "config.json"))
    state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    for k in state.keys():
        if "wte.weight" in k: config.vocab_size = state[k].shape[0]; break
    
    backbone = GPT2LMHeadModel(config)
    model = DiffusionModel(backbone, config)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    return model.cuda().eval()

# ==========================================
# 3. HÀM THU THẬP DỮ LIỆU (DATA COLLECTOR)
# ==========================================
def collect_data():
    # Load Model
    model = load_model(CHECKPOINT_PATH)
    tokenizer = SudokuTokenizer()
    
    # Load Data
    print(f"Đang đọc dữ liệu từ {CSV_DATA_PATH}...")
    df = pd.read_csv(CSV_DATA_PATH)
    # Lấy mẫu ngẫu nhiên
    if len(df) > NUM_SAMPLES:
        df = df.sample(NUM_SAMPLES, random_state=42)
    
    data_buffer = {
        "hidden_states": [], # X (Features)
        "labels": []         # Y (Targets)
    }
    
    print(f"Bắt đầu thu thập dữ liệu MIND từ {NUM_SAMPLES} mẫu...")
    print("Mỗi mẫu chạy 20 bước -> Tổng số mẫu training cho MIND sẽ rất lớn!")
    
    for _, row in tqdm(df.iterrows(), total=NUM_SAMPLES):
        quiz, sol = row['quizzes'], row['solutions']
        
        # Prepare Tensors
        in_ids = torch.tensor([tokenizer.encode(quiz, is_input=True)]).cuda() # [1, 81]
        gt_ids = torch.tensor([tokenizer.encode(sol, is_input=False)]).cuda() # [1, 81]
        
        # Thêm BOS token để khớp logic inference
        bos = torch.tensor([[tokenizer.mask_token_id]]).cuda()
        x = torch.cat([bos, in_ids], dim=1)         # [1, 82]
        gt_x = torch.cat([bos, gt_ids], dim=1)      # [1, 82]
        
        # Mask
        src_mask = (x != tokenizer.mask_token_id)
        maskable_mask = ~src_mask
        maskable_mask[:, 0] = False # Không xét BOS
        
        # --- STATIC INFERENCE LOOP ---
        # Chúng ta chạy lại quá trình suy luận để xem model "nghĩ gì"
        for t in range(DIFFUSION_STEPS - 1, -1, -1):
            with torch.no_grad():
                # 1. Tạo input nhiễu cho bước t
                if t == DIFFUSION_STEPS - 1:
                    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
                
                # 2. Forward để lấy Hidden States
                t_tensor = torch.full((1, ), t, device=x.device)
                logits, hidden = model(xt, t_tensor, output_hidden_states=True)
                
                # Shift Logits & Hidden (Do cơ chế causal của GPT)
                # Hidden gốc: [1, 82, 768] -> Shift để token i nhìn thấy i
                # Lưu ý: Với Diffusion non-causal (attention mask full), hidden[i] chứa thông tin ngữ cảnh
                # Nhưng output logits đã được shift, nên ta shift hidden tương ứng
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # 3. Dự đoán
                preds = logits.argmax(dim=-1) # [1, 82]
                
                # 4. Tạo Nhãn (Correctness)
                # So sánh Preds với Ground Truth
                is_correct = (preds == gt_x).long() # 1=Đúng, 0=Sai
                
                # 5. Lọc dữ liệu: CHỈ LẤY CÁC Ô ĐANG BỊ MASK
                # Vì các ô đề bài (src_mask) thì luôn đúng, học làm gì cho nhiễu
                # Ta chỉ quan tâm những ô model phải "suy nghĩ"
                
                # maskable_mask: [1, 82] (True ở những ô cần điền)
                # Ta chỉ lấy data tại các vị trí này
                
                active_hidden = hidden[maskable_mask]      # [N_masked, Hidden_Dim]
                active_labels = is_correct[maskable_mask]  # [N_masked]
                
                # 6. Lưu vào buffer (Chuyển về CPU để tiết kiệm VRAM)
                data_buffer["hidden_states"].append(active_hidden.cpu().half()) # Lưu float16 cho nhẹ
                data_buffer["labels"].append(active_labels.cpu())
                
                # 7. Update xt cho bước sau (Theo logic Static)
                # Cần update xt để mô phỏng đúng quá trình suy luận
                # (Ở đây ta dùng update kiểu Greedy đơn giản để lấy trajectory)
                x0_preds = preds
                xt = xt.masked_scatter(maskable_mask, x0_preds[maskable_mask])
                
                # Random re-mask (giả lập quá trình static diffusion)
                if t > 0:
                     unmask_prob = 1 / (t + 1)
                     mask_keep = torch.rand(xt.shape, device=x.device) < unmask_prob
                     mask_keep = mask_keep & maskable_mask
                     xt[mask_keep] = x0_preds[mask_keep]
                     maskable_mask[mask_keep] = False
                else:
                    xt = x0_preds

    # Gộp data lại
    print("Đang ghép dữ liệu...")
    all_hidden = torch.cat(data_buffer["hidden_states"], dim=0) # [Total_Tokens, Hidden_Dim]
    all_labels = torch.cat(data_buffer["labels"], dim=0)        # [Total_Tokens]
    
    print(f"Hoàn tất! Kích thước dataset: {all_hidden.shape}")
    print(f"Tỷ lệ nhãn đúng (1): {all_labels.float().mean():.2%}")
    
    torch.save({"x": all_hidden, "y": all_labels}, OUTPUT_FILE)
    print(f"Đã lưu dataset tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    collect_data()