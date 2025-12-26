import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

class IntrospectionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Mạng MLP đơn giản: Input -> 256 -> 64 -> 1 (Sigmoid)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256), # Giúp train ổn định hơn
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="introspection_data.pt")
    parser.add_argument("--save_path", default="introspection_net.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192) # Batch to vì H200 mạnh
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Không tìm thấy file {args.data_path}. Hãy chạy collect_data.py trước!")
        return

    # 1. LOAD DATA
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path)
    X = data['x'].float() # Convert lại sang float32 để train
    Y = data['y'].float().unsqueeze(1)
    
    # Tự động lấy hidden dim từ file save
    input_dim = data.get("hidden_dim", X.shape[1])
    print(f"Input Dimension: {input_dim}")
    print(f"Total Samples: {len(X)}")

    # Split Train/Val (90/10)
    split = int(len(X) * 0.9)
    # Shuffle nhẹ
    perm = torch.randperm(len(X))
    train_X, val_X = X[perm[:split]], X[perm[split:]]
    train_Y, val_Y = Y[perm[:split]], Y[perm[split:]]

    # DataLoader
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=args.batch_size, shuffle=True)
    
    # 2. INIT MODEL
    model = IntrospectionNet(input_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss() # Binary Cross Entropy

    # 3. TRAINING LOOP
    print("\nStart Training Introspection Network...")
    print(f"{'Epoch':^5} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^10}")
    print("-" * 45)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for bx, by in train_loader:
            bx, by = bx.cuda(), by.cuda()
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.cuda())
            v_loss = criterion(val_pred, val_Y.cuda()).item()
            # Accuracy: Nếu > 0.5 coi là class 1
            predicted_class = (val_pred > 0.5).float()
            acc = (predicted_class == val_Y.cuda()).float().mean().item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"{epoch+1:^5} | {avg_train_loss:^12.4f} | {v_loss:^10.4f} | {acc:^10.2%}")

    # 4. SAVE
    # Lưu cả weight và config input_dim để lúc load inference không bị nhầm
    save_dict = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim
    }
    torch.save(save_dict, args.save_path)
    print(f"\nTraining Complete! Model saved to {args.save_path}")

if __name__ == "__main__":
    train()