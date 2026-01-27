import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import joblib

# ================= ⚙️ CONFIGURATION =================
CSV_FILE = 'output_dataset.csv' # หรือ output_dataset.csv
SEQ_LEN = 60        
BATCH_SIZE = 32
MAX_EPOCHS = 100    
LEARNING_RATE = 0.001
# ====================================================

FEATURE_COLS = [
    'd_nose_x', 'd_nose_y', 'd_nose_z',
    'd_leye_x', 'd_leye_y', 'd_leye_z',
    'd_reye_x', 'd_reye_y', 'd_reye_z',
    'd_lear_x', 'd_lear_y', 'd_lear_z',
    'd_rear_x', 'd_rear_y', 'd_rear_z',
    'bg_flow_x', 'bg_flow_y', 'fg_flow_x', 'fg_flow_y',
    'gyro_x', 'gyro_y', 'gyro_z',
    'accel_x', 'accel_y', 'accel_z'
]

# --- 1. Dataset Class with Augmentation ---
class KYCDataset(Dataset):
    def __init__(self, sequences, labels, training=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.training = training 

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # 🔥 Augmentation: ใส่ Noise เฉพาะตอน Train
        if self.training:
            noise = torch.randn_like(seq) * 0.02
            seq = seq + noise
            
        return seq, label

# --- 2. Model Architecture ---
class AntiDeepfakeModelPro(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gru_vision = nn.GRU(19, 48, batch_first=True, bidirectional=True)
        self.gru_sensor = nn.GRU(6, 48, batch_first=True, bidirectional=True)
        
        self.bn = nn.BatchNorm1d(192) 
        
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, 1)
        )

    def forward(self, x):
        vis = x[:, :, :19]  
        sen = x[:, :, 19:]  
        
        _, h_v = self.gru_vision(vis)   
        _, h_s = self.gru_sensor(sen)   
        
        feat_v = torch.cat((h_v[-2], h_v[-1]), dim=1) 
        feat_s = torch.cat((h_s[-2], h_s[-1]), dim=1) 
        
        combined = torch.cat((feat_v, feat_s), dim=1)
        return self.classifier(self.bn(combined))

# --- 3. Padding Function ---
def pad_sequence(data, max_len):
    length = len(data)
    if length >= max_len:
        return data[:max_len]
    else:
        padding = np.zeros((max_len - length, data.shape[1]))
        return np.vstack((data, padding))

# --- 4. Data Loading ---
def load_and_prep_data():
    if not os.path.exists(CSV_FILE):
        print(f"❌ {CSV_FILE} not found!"); return None, None, None
    
    df = pd.read_csv(CSV_FILE)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}"); return None, None, None

    X_list, y_list, groups_list = [], [], []
    grouped = df.groupby('session')
    
    for session_name, group in grouped:
        group = group.sort_values('timestamp')
        data = group[FEATURE_COLS].values
        label = group['label'].iloc[0]
        
        if len(data) >= 3: 
            padded_data = pad_sequence(data, SEQ_LEN)
            X_list.append(padded_data)
            y_list.append(label)
            groups_list.append(session_name)
            
    return np.array(X_list), np.array(y_list), np.array(groups_list)

# --- 5. Main Execution ---
def main():
    print("📂 Loading Data...")
    X, y, groups = load_and_prep_data()
    if X is None: return
    
    # Check data balance
    print(f"✅ Loaded {len(X)} sequences. Shape: {X.shape}")
    print(f"   Real: {sum(y==1)} | Fake: {sum(y==0)}")

    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    N, S, F = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N, S, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], S, F)
    
    joblib.dump(scaler, 'scaler_v2.pkl')
    
    train_loader = DataLoader(KYCDataset(X_train, y_train, training=True), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(KYCDataset(X_test, y_test, training=False), batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training on {device}")
    
    model = AntiDeepfakeModelPro().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 🔥 FIX: ลบ verbose=True ออก (เพราะ PyTorch ใหม่เลิกใช้)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_acc = 0.0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100 * correct / total
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        test_acc = 100 * correct / total
        
        # Update Scheduler
        scheduler.step(test_acc)
        
        # 🔥 Manual Print Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | LR: {current_lr:.6f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "liveness_model_final.pth")
            
    print(f"🏆 Best Accuracy: {best_acc:.2f}%")
    print("💾 Model saved to liveness_model_final.pth")

if __name__ == "__main__":
    main()