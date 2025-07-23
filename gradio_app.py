import gradio as gr
import torch
import numpy as np
import cv2
from model.unet import UNet
from torchvision import transforms

# Modeli yükle
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("unet_levircd.pth", map_location=device))
model.to(device)
model.eval()

# Görseli hazırla
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict_changes(image_a, image_b):
    # Görselleri yeniden boyutlandır
    image_a = cv2.resize(image_a, (256, 256)) / 255.0
    image_b = cv2.resize(image_b, (256, 256)) / 255.0

    # Tensor hale getir
    a_tensor = transform(image_a).unsqueeze(0).to(device)
    b_tensor = transform(image_b).unsqueeze(0).to(device)

    # Tahmin et
    with torch.no_grad():
        output = model(a_tensor, b_tensor)
        prediction = torch.sigmoid(output)
        mask = prediction.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

    return mask

# Gradio Arayüzü
with gr.Blocks() as demo:
    gr.Markdown("## 🛰️ Uydu Görüntüsü Değişiklik Tespiti")
    gr.Markdown("Aşağıya iki uydu görüntüsü yükleyin: önceki ve sonraki")

    with gr.Row():
        image_a = gr.Image(type="numpy", label="Önceki Görüntü")
        image_b = gr.Image(type="numpy", label="Sonraki Görüntü")

    predict_button = gr.Button("Değişiklikleri Tahmin Et")

    output_mask = gr.Image(type="numpy", label="Tahmin Edilen Değişiklik Maskesi")

    predict_button.click(fn=predict_changes, inputs=[image_a, image_b], outputs=output_mask)

# Çalıştır
demo.launch()