import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from model.unet import UNet

#M1 GPU kontrolü (Metal backend)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple M1 GPU (MPS) kullanılacak")
else:
    device = torch.device("cpu")
    print("MPS desteklenmiyor, CPU kullanılacak")

# Dataset sınıfı
class ChangeDetectionDataset(Dataset):
    def __init__(self, a_paths, b_paths, label_paths):
        self.a_paths = a_paths
        self.b_paths = b_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.a_paths)

    def __getitem__(self, idx):
        a = cv2.imread(self.a_paths[idx])
        b = cv2.imread(self.b_paths[idx])
        label = cv2.imread(self.label_paths[idx], 0)

        # Resize
        a = cv2.resize(a, (256, 256))
        b = cv2.resize(b, (256, 256))
        label = cv2.resize(label, (256, 256))

        # Normalize
        a = a / 255.0
        b = b / 255.0
        label = label / 255.0

        # Kanal birleştirme ve tensöre çevirme
        x = np.concatenate([a, b], axis=2).transpose(2, 0, 1)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return x, y

# Veri yolları
def get_paths(split="train"):
    base = f"data/levircd/{split}"
    A = sorted(glob(os.path.join(base, "A", "*.png")))
    B = sorted(glob(os.path.join(base, "B", "*.png")))
    L = sorted(glob(os.path.join(base, "label", "*.png")))
    return A, B, L

# Veriyi yükle
train_A, train_B, train_L = get_paths("train")
train_ds = ChangeDetectionDataset(train_A, train_B, train_L)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# Model + Loss + Optimizasyon
model = UNet(in_channels=6, out_channels=1).to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Eğitim
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)

        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Kaydet
torch.save(model.state_dict(), "unet_levircd.pth")
print("Model başarıyla kaydedildi.")