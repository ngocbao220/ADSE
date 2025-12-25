import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DATA_FILE = "mind_dataset_train.pt"
OUTPUT_MODEL = "mind_model.pth"
INPUT_DIM = 384 
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

# ==========================================
# 2. ĐỊNH NGHĨA MẠNG NỘI QUAN (MIND)
# ==========================================
class IntrospectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output: 0.0 -> 1.0 (Xác suất tự tin)
        )
    
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. DATASET
# ==========================================
class MindDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        # Chuyển về float32 để train
        self.x = data["x"].float() 
        self.y = data["y"].float().unsqueeze(1) # [N, 1]
        
        print(f"Loaded Data: {self.x.shape}")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    # 1. Prepare Data
    if not torch.cuda.is_available():
        print("WARNING: Đang chạy trên CPU, sẽ hơi chậm!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = MindDataset(DATA_FILE)
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # 2. Init Model
    model = IntrospectionHead(INPUT_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss() # Binary Cross Entropy (Đúng/Sai)
    
    # 3. Train
    print(f"\nStart Training MIND on {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == y_val).sum().item()
                total += y_val.size(0)
        
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2%}")

    # 4. Save
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\n>>> Đã lưu model tại: {OUTPUT_MODEL}")
    print("Mạng MIND đã sẵn sàng để tích hợp vào Inference!")

if __name__ == "__main__":
    train()