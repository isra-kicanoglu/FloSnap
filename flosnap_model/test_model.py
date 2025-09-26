import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
import numpy as np
import numpy as np

# 1. MODELİ VE İŞLEMCİYİ YÜKLE
# --------------------------------
model_name = "./finetuned_clip_model"
try:
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. ÖZELLİKLERİ YÜKLE (Torch ile kaydedilen yapı)
# ------------------------------------------------
try:
    saved = torch.load("extracted_features.pkl")
    features = saved["features"]  # torch.Tensor [N, 512], normalize
    metadata = saved["metadata"]
    print(f"\nVeri tabanında {features.size(0)} adet görsel bulundu.")
except FileNotFoundError:
    print("\nHata: 'extracted_features.pkl' dosyası bulunamadı.")
    print("Lütfen önce 'feature_extractor.py' dosyasını çalıştırın.")
    exit()

# 3. TEST İÇİN HEDEF GÖRSELİ YÜKLE
# --------------------------------
target_image_path = "target.jpg" 
try:
    target_image = Image.open(target_image_path).convert("RGB")
    print(f"Hedef görsel başarıyla yüklendi: {target_image_path}")
except FileNotFoundError:
    print("\nHata: `target.jpg` adlı görsel bulunamadı.")
    print("Lütfen aynı klasöre test etmek istediğiniz görseli `target.jpg` adıyla kaydedin.")
    exit()

# 4. HEDEF GÖRSELİN ÖZELLİKLERİNİ AL VE BENZERLİKLERİ HESAPLA
# --------------------------------------------------------
inputs = processor(images=target_image, return_tensors="pt").to(device)
with torch.no_grad():
    target_embedding = model.get_image_features(**inputs).squeeze().cpu().numpy()

# L2 normalize target (indeksteki vektörler normalize)
target_embedding = target_embedding / (np.linalg.norm(target_embedding) + 1e-12)

# Cosine similarity = dot product since both sides normalized
all_features = features.cpu().numpy()  # [N, 512]
similarities = all_features @ target_embedding  # [N]

# 5. EN YÜKSEK SKORLARI SIRALA VE BENZER ÜRÜNLERİ LİSTELE
# ---------------------------------------------------
top_k = 5
top_indices = np.argsort(-similarities)[:top_k]

print("\n--- Hedef Görsele En Çok Benzeyen Benzersiz Ürünler ---")
for rank, idx in enumerate(top_indices, start=1):
    m = metadata[idx]
    score = similarities[idx]
    print(f"{rank}. SIRA: Ürün ID: {m.get('product_id', 'bilinmiyor')} | En İyi Benzerlik Skoru: {score:.4f} | Görsel Yolu: {m.get('image_path', '')}")