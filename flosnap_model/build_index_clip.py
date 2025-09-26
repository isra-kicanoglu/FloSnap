import torch
import os
import faiss
import numpy as np
import json
from transformers import CLIPModel, CLIPProcessor
import re
from urllib.parse import urlparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Sabitler
MODEL_PATH = os.path.abspath("./finetuned_clip_with_logos")  # Eğitilmiş modelin konumu
DATASET_FILE = "ayakkabilar.jsonl"
IMAGES_FOLDER = os.path.join("..", "ayakkabi_scraper", "ayakkabi_resimleri")
INDEX_PATH = "./flo_index.faiss"

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
        
        processed_images = []
        
        if not image_urls:
            return None

        for image_relative_path in image_urls:
            # URL ya da yol gelebilir → önce olası query'leri ve klasörlerini ayıkla
            parsed_path = urlparse(image_relative_path).path  # sadece path kısmı
            image_name = os.path.basename(parsed_path)

            # product_id: dosya adının başındaki sayısal blok (örn: 101978338_d1.jpg → 101978338)
            match = re.match(r"(\d+)", image_name)
            if not match:
                continue
            product_id = match.group(1)

            # Doğru yerel dosya yolu: images_folder/product_id/image_name
            image_path = os.path.join(self.images_folder, product_id, image_name)
            
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    processed_images.append(image)
                except Exception as e:
                    print(f"Resim yüklenirken hata oluştu: {image_path}. Hata: {e}")
            else:
                print(f"Uyarı: Resim dosyası bulunamadı: {image_path}")
        
        if processed_images:
            # Sadece görsel veriyi hazırla
            inputs = self.processor(
                images=processed_images,
                return_tensors="pt"
            )
            return inputs

        return None

# Toplu İşlem Fonksiyonu
def collate_fn(batch):
    # None olan öğeleri çıkar
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # inputs sözlüğünü listelere dönüştür
    inputs = {key: [d[key] for d in batch] for key in batch[0]}
    
    # Tensorları yığınla
    for key in inputs:
        # Tek bir tensör olacak şekilde birleştir
        inputs[key] = torch.cat(inputs[key], dim=0)
    
    return inputs

# Vektörleri Oluşturma ve İndeksleme
def build_index():
    print("Eğitilmiş model yükleniyor...")
    try:
        # Model klasörünün varlığını kontrol et
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Eğitilmiş model klasörü bulunamadı: {MODEL_PATH}. Lütfen 'train_model.py' dosyasını çalıştırdığınızdan emin olun.")
        
        model = CLIPModel.from_pretrained(MODEL_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval() # Modeli değerlendirme moduna al

        # Görsel klasörünün varlığını kontrol et
        if not os.path.exists(IMAGES_FOLDER):
            raise FileNotFoundError(f"Görsel klasörü bulunamadı: {IMAGES_FOLDER}. Lütfen 'ayakkabi_scraper' klasörünün ve resimlerin doğru konumda olduğundan emin olun.")
        
        dataset = AyakkabiDataset(DATASET_FILE, processor, IMAGES_FOLDER)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

        all_embeddings = []

        print("Görsellerden vektörler (embeddingler) çıkarılıyor...")
        
        with torch.no_grad(): # Gradient hesaplamalarını devre dışı bırak
            for i, batch in enumerate(dataloader):
                if batch is None:
                    continue
                
                # Sadece resim verilerini al ve cihaza taşı
                pixel_values = batch['pixel_values'].to(device)
                
                # Resim vektörlerini çıkar
                image_features = model.get_image_features(pixel_values=pixel_values)
                
                # L2 normalizasyonu uygula
                normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)

                all_embeddings.append(normalized_features.cpu().numpy())
                
                print(f"Parti {i+1} işlendi.")
        
        if not all_embeddings:
            print("\n")
            print("--- HATA AÇIKLAMASI ---")
            print("Hata: Hiçbir geçerli vektör oluşturulamadı. Bu, 'ayakkabilar.jsonl' dosyasındaki resim yollarının diskinizde bulunamadığı anlamına gelir.")
            print("Lütfen aşağıdaki klasörün varlığını ve içinde resimler olduğundan emin olun:")
            print(f"  -> {os.path.abspath(IMAGES_FOLDER)}")
            print("---")
            return

        # Tüm vektörleri birleştir
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # FAISS indeksini oluştur
        d = all_embeddings.shape[1] # Vektör boyutu
        index = faiss.IndexFlatIP(d) # İç çarpım (cosine benzerliği için L2 normalizasyonu gerekli)
        index.add(all_embeddings)

        # İndeksi kaydet
        faiss.write_index(index, INDEX_PATH)
        
        print("\nFAISS indeksi başarıyla oluşturuldu ve kaydedildi! 🎉")
        print(f"Yeni indeksin konumu: {INDEX_PATH}")
        print(f"Toplam vektör sayısı: {index.ntotal}")

    except Exception as e:
        print(f"İndeks oluşturulurken bir hata oluştu: {e}")

build_index()

