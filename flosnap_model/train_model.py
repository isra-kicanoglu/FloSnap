import torch
import os
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from itertools import chain

# Sabitler
MODEL_PATH = "./finetuned_clip_model"
DATASET_FILE = "ayakkabilar.jsonl"
IMAGES_FOLDER = os.path.join("..", "ayakkabi_scraper", "ayakkabi_resimleri")
SAVE_PATH = "./finetuned_clip_with_logos"

# Ayakkabı Veri Seti Sınıfı
# Bu sınıf, ayakkabilar.jsonl dosyasını okuyarak veri setini hazırlar.
class AyakkabiDataset(Dataset):
    def __init__(self, file_path, processor, images_folder):
        self.processor = processor
        self.images_folder = images_folder
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_urls = item.get('image_urls', [])
        brand = item.get('brand')
        
        processed_images = []
        
        if image_urls and brand:
            for image_relative_path in image_urls:
                # Ürün ID'sini ve dosya adını al
                product_id = os.path.basename(os.path.dirname(image_relative_path))
                image_name = os.path.basename(image_relative_path)
                
                # Doğru dosya yolunu oluştur
                image_path = os.path.join(self.images_folder, product_id, image_name)
                
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path).convert("RGB")
                        processed_images.append(image)
                    except Exception as e:
                        print(f"Resim yüklenirken hata oluştu: {image_path}. Hata: {e}")
            
            if processed_images:
                # Hem görsel hem de metin (marka) için veriyi hazırla
                inputs = self.processor(
                    text=brand,
                    images=processed_images,
                    return_tensors="pt",
                    padding=True
                )
                inputs['input_ids'] = inputs['input_ids'].squeeze(0)
                # Resim vektörlerinin ortalamasını al
                inputs['pixel_values'] = inputs['pixel_values'].mean(dim=0).unsqueeze(0)
                return inputs

        return None

# Toplu İşlem Fonksiyonu
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    inputs = {key: [d[key] for d in batch] for key in batch[0]}
    
    for key in inputs:
        # Metin girdileri için yığınlama
        if key == 'input_ids':
            inputs[key] = torch.stack(inputs[key])
        # Resim girdileri için boyutlandırma ve yığınlama
        else:
            inputs[key] = torch.cat(inputs[key], dim=0)

    return inputs

# Model eğitimi
def train_model():
    print("Mevcut model yükleniyor...")
    try:
        model = CLIPModel.from_pretrained(MODEL_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        dataset = AyakkabiDataset(DATASET_FILE, processor, IMAGES_FOLDER)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        print("Modelin markaları öğrenmesi için ince ayar (fine-tuning) başlatılıyor...")
        
        num_epochs = 1
        model.train()
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                if batch is None:
                    continue
                
                optimizer.zero_grad()
                
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                print(f"Epoch: {epoch+1}, Kayıp (Loss): {loss.item():.4f}")

        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        
        # Hem modeli hem de işlemciyi kaydet
        model.save_pretrained(SAVE_PATH)
        processor.save_pretrained(SAVE_PATH)
        
        print("\nModel başarıyla eğitildi ve yeni klasöre kaydedildi! 🎉")
        print(f"Yeni modelin konumu: {SAVE_PATH}")

    except Exception as e:
        print(f"Model eğitimi sırasında bir hata oluştu: {e}")

train_model()
