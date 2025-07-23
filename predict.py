import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNet

#Apple M1 (Metal) için MPS kontrolü
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Kullanılan cihaz: {device}")

#Modeli yükle
model = UNet(in_channels=6, out_channels=1).to(device)
model.load_state_dict(torch.load("unet_levircd.pth", map_location=device))
model.eval()

#Tahmin için örnek görsel seç
img_name = "train_638.png"
path_a = f"data/levircd/test/A/{img_name}"
path_b = f"data/levircd/test/B/{img_name}"

#Görselleri yükle ve hazırla
a = cv2.imread(path_a)
b = cv2.imread(path_b)

a = cv2.resize(a, (256, 256)) / 255.0
b = cv2.resize(b, (256, 256)) / 255.0

input_img = np.concatenate([a, b], axis=2).transpose(2, 0, 1)
input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).to(device)

#Tahmin et
with torch.no_grad():
    pred = model(input_tensor)
    pred_mask = pred.squeeze().cpu().numpy()

#Maske eşikleme (0.5'ten büyükse değişim var kabul et)
pred_binary = (pred_mask > 0.5).astype(np.uint8) * 255

#Görselleştir
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Önceki")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title("Sonraki")

plt.subplot(1, 3, 3)
plt.imshow(pred_binary, cmap="gray")
plt.title("Tahmin Edilen Değişiklik")

plt.tight_layout()
plt.show()