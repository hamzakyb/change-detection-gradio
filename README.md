# 🛰️ AI Change Detection - Gradio App

Bu proje, **uydu görüntülerinde zaman içindeki değişiklikleri tespit etmek** amacıyla geliştirilmiş bir yapay zeka uygulamasıdır. Gradio arayüzüyle entegre edilmiştir ve Hugging Face Spaces üzerinde çalışan bir demo sunar.

---

## 📌 Proje Özellikleri

- 🔍 Uydu görüntülerinde değişiklik tespiti (ör. yapılaşma, afet sonrası hasar, doğa tahribatı vs.)
- 🧠 Eğitilmiş UNet tabanlı segmentasyon modeli (`unet_levircd.pth`)
- 🖼️ İki farklı zaman dilimine ait görselleri karşılaştırarak değişiklik alanlarını vurgulama
- ⚡ Gradio ile web arayüzü
- ☁️ Hugging Face Spaces üzerinde canlı demo

---

## 🧠 Kullanılan Yöntem ve Yaklaşım

Bu projede, görüntü segmentasyonu için derin öğrenmeye dayalı **UNet mimarisi** kullanılmıştır. Amaç, ** iki farklı zamanda çekilmiş uydu görüntüsü ** arasındaki farkları **piksel düzeyinde tespit etmek** ve bunları segmentasyon maskesi olarak kullanıcıya sunmaktır.

### 🧩 Adımlar:

1. ** Giriş Verisi **: Kullanıcı iki adet uydu görüntüsü yükler (örneğin 2020 ve 2023).
2. ** Veri İşleme **: Görseller normalize edilir ve modelin anlayabileceği boyutlara dönüştürülür.
3. ** Model Tahmini **: Eğitilmiş `UNet` modeli, bu iki görüntü arasındaki farkları öğrenerek bir "değişim maskesi" üretir.
4. ** Çıktı **: Üretilen maskede sadece değişiklik içeren pikseller vurgulanır.

### 🔍 Kullanılan Veri Seti

Model, ** LEVIR-CD ** adlı yüksek çözünürlüklü bir değişiklik tespiti veri seti ile eğitilmiştir. Bu veri seti:

- 637 çift uydu görüntüsünden oluşur.
- Her bir çiftte, önceki ve sonraki zaman dilimlerine ait görseller vardır.
- Gerçek etiketli değişim maskeleri içerir.

---

## 🚀 Canlı Demo ve Bağlantılar

| Platform | Link |
|----------|------|
| 🔴 Canlı Demo (Hugging Face) | [change-detection-gradio](https://huggingface.co/spaces/Hamzakoybasi/change-detection-gradio) |
| 📁 Hugging Face Dosyalar | [Tree view](https://huggingface.co/spaces/Hamzakoybasi/change-detection-gradio/tree/main) |
| 🧑‍💻 GitHub Kaynağı | [GitHub Repository](https://github.com/hamzakyb/change-detection-gradio) |

---

## 🧰 Kullanılan Teknolojiler

- Python
- PyTorch
- Gradio
- Hugging Face Spaces
- UNet Model
- LEVIR-CD Veri Seti

---

## ⚙️ Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için şu adımları takip edebilirsiniz:

```bash
git clone https://github.com/hamzakyb/change-detection-gradio.git
cd change-detection-gradio

# Sanal ortam (isteğe bağlı)
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Uygulamayı çalıştır
python app.py

⸻

📂 Proje Yapısı

📦 change-detection-gradio
 ┣ 📁 model/
 ┣ 📜 app.py
 ┣ 📜 utils.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📜 unet_levircd.pth


⸻

👨‍💻 Geliştirici

Hamza Köybaşı
📫 hamzakybsi@gmail.com
🌍 [LinkedIn Profilim](https://www.linkedin.com/in/hamzakybsi/)

⸻

📄 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.

⸻

⭐ Destek Ol

Beğendiyseniz GitHub reposuna ⭐ bırakabilirsiniz.
