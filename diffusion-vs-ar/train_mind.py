import torch
import torch. nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import argparse
import os
from sklearn.metrics import roc_auc_score, average_precision_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

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

def compute_metrics(pred, target, threshold=0.5):
    """Compute comprehensive metrics"""
    pred_class = (pred > threshold).float()
    
    tp = ((pred_class == 1) & (target == 1)).sum().item()
    fp = ((pred_class == 1) & (target == 0)).sum().item()
    fn = ((pred_class == 0) & (target == 1)).sum().item()
    tn = ((pred_class == 0) & (target == 0)).sum().item()
    
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC-ROC and AUC-PR
    try:
        auc_roc = roc_auc_score(target.cpu(), pred.cpu())
        auc_pr = average_precision_score(target.cpu(), pred.cpu())
    except:
        auc_roc = auc_pr = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }

def find_optimal_threshold(pred, target):
    """Find threshold that maximizes F1"""
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in torch.linspace(0.1, 0.9, 50):
        metrics = compute_metrics(pred, target, threshold=thresh)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_thresh = thresh. item()
    
    return best_thresh, best_f1

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="introspection_data.pt")
    parser.add_argument("--save_path", default="introspection_net.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error:  Không tìm thấy file {args.data_path}")
        return

    # 1. LOAD DATA
    print(f"Loading data from {args. data_path}...")
    data = torch.load(args.data_path)
    X = data['x']. float()
    Y = data['y'].float().unsqueeze(1)
    
    input_dim = data. get("hidden_dim", X.shape[1])
    print(f"Input Dimension: {input_dim}")
    print(f"Total Samples: {len(X)}")
    
    # Check class balance
    n_wrong = (Y == 1).sum().item()
    n_correct = (Y == 0).sum().item()
    print(f"\nClass Distribution:")
    print(f"  Correct (0): {n_correct} ({n_correct/len(Y):.1%})")
    print(f"  Wrong (1):   {n_wrong} ({n_wrong/len(Y):.1%})")
    print(f"  Imbalance Ratio: {n_correct/n_wrong:.1f}: 1")

    # Split Train/Val
    split = int(len(X) * 0.9)
    perm = torch.randperm(len(X))
    train_X, val_X = X[perm[:split]], X[perm[split:]]
    train_Y, val_Y = Y[perm[:split]], Y[perm[split:]]

    # Weighted sampling for imbalanced data
    class_counts = torch.bincount(train_Y. long().squeeze())
    class_weights = 1.0 / class_counts. float()
    sample_weights = class_weights[train_Y.long().squeeze()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        TensorDataset(train_X, train_Y), 
        batch_size=args.batch_size, 
        sampler=sampler
    )

    # 2. INIT MODEL
    model = IntrospectionNet(input_dim).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim. lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2)
        print("Using Focal Loss")
    else:
        pos_weight = (train_Y == 0).sum() / (train_Y == 1).sum()
        criterion = nn.BCELoss()
        print(f"Using BCE Loss (pos_weight={pos_weight:.1f})")

    # 3. TRAINING LOOP
    print(f"\n{'='*80}")
    print("Start Training Introspection Network")
    print(f"{'='*80}")
    print(f"{'Epoch': >5} | {'Train Loss':>11} | {'Val F1':>8} | {'Val AUC': >8} | {'Recall':>7} | {'Prec':>7}")
    print("-" * 80)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(args. epochs):
        # Training
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
            val_pred = model(val_X. cuda())
            v_loss = criterion(val_pred, val_Y.cuda()).item()
            
            # Compute metrics
            metrics = compute_metrics(val_pred, val_Y.cuda(), threshold=0.5)
        
        avg_train_loss = total_loss / len(train_loader)
        
        print(f"{epoch+1:>5} | {avg_train_loss:>11.4f} | {metrics['f1']:>8.2%} | "
              f"{metrics['auc_roc']:>8.3f} | {metrics['recall']: >7.2%} | {metrics['precision']:>7.2%}")
        
        # Scheduler step
        scheduler.step(metrics['f1'])
        
        # Early stopping based on F1
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            
            # Find optimal threshold
            opt_thresh, opt_f1 = find_optimal_threshold(val_pred, val_Y. cuda())
            
            save_dict = {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "optimal_threshold": opt_thresh,
                "best_f1": best_f1,
                "metrics": metrics
            }
            torch.save(save_dict, args.save_path)
            patience_counter = 0
            
            if epoch > 5:   # Don't print in early epochs
                print(f"  → New best!  Optimal***
