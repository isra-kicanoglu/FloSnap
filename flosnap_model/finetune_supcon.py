import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
from SupCon.loss import SupConLoss
from tqdm import tqdm
import jsonlines
import torch.nn as nn
import torch.nn.functional as F

# 1. Sabitler ve Ayarlar
DATA_ROOT = "../ayakkabi_scraper/ayakkabi_resimleri"
MODEL_PATH = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
LEARNING_RATE = 1e-6  #öğrenme hızı
NUM_EPOCHS = 10
WARMUP_STEPS = 500

# 2. CUDA Desteği
device = "cuda" if torch.cuda.is_available() else "cpu"  #gpu varsa kullan
print(f"Kullanılan Cihaz: {device}")

# 3. Ayakkabı Veri Kümesi (Dataset)
class ShoesDataset(Dataset):
    def __init__(self, root_dir, jsonl_path):
        self.root_dir = root_dir   #atama işlemü
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),   #makine öğrenmesine ayarlama
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        
        # Ürün ID'lerini ve ilgili görsel yollarını topla
        self.product_data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            reader = jsonlines.Reader(f)
            metadata = list(reader)
        
        # Product ID'lere göre görsel yollarını grupla
        product_images = {}
        for item in metadata:
            product_id = str(item.get("product_id"))
            product_folder = os.path.join(self.root_dir, product_id)
            if os.path.isdir(product_folder):
                images_in_folder = glob.glob(os.path.join(product_folder, "*.jpg")) + \
                                   glob.glob(os.path.join(product_folder, "*.JPG")) + \
                                   glob.glob(os.path.join(product_folder, "*.jpeg")) + \
                                   glob.glob(os.path.join(product_folder, "*.JPEG")) + \
                                   glob.glob(os.path.join(product_folder, "*.png")) + \
                                   glob.glob(os.path.join(product_folder, "*.PNG"))
                if len(images_in_folder) >= 2:  #eğer 2 veya daha fazla görsel varsa onları al
                    if product_id not in product_images:
                        product_images[product_id] = []
                    product_images[product_id].extend(images_in_folder)
        
        # Veri setini oluştur
        for product_id, image_paths in product_images.items():
            for img_path in image_paths:
                self.product_data.append({'image_path': img_path, 'label': product_id})
        
        self.labels = [item['label'] for item in self.product_data]
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_map = {label: i for i, label in enumerate(self.unique_labels)}
        
        print(f"Toplam {len(self.product_data)} görsel ve {len(self.unique_labels)} benzersiz ürün bulundu.")

    def __len__(self):
        return len(self.product_data)  #kaç tane verid örnek var döndür

    def __getitem__(self, idx):  #idx parametre modelin
        item = self.product_data[idx]
        img_path = item['image_path']
        label = item['label']
        
        image = Image.open(img_path).convert("RGB")  #insan gözünün aldığı temel renkler olduğu için rgb
        image = self.transform(image)
        
        # `label_map` ile product_id'yi sayısal bir değere çevir
        label_id = self.label_map[label]
        return image, label_id, label

# 4. Projeksiyon Başı (Projection Head)
class ImageProjHead(torch.nn.Module):
    # Bu kısmı 512 olarak düzelttik
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.ReLU(inplace=True),   #negatif değerleri 0 a çekiyo 
            torch.nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)  #modelin tahmin yaparken izlediği yol

if __name__ == '__main__':  #dosya doğrudan çalıştırılıyorsa çalışıyo
    # 5. Model ve Bileşenlerin Yüklenmesi
    print("CLIP modeli ve tokenizer yükleniyor...")
    model = CLIPModel.from_pretrained(MODEL_PATH)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH)

    # Sadece görsel encoder'ı kullanacağımız için text encoder'ı dondur
    for p in model.text_model.parameters():
        p.requires_grad = False   #true yaparsam model textide alır 
    model.vision_model.requires_grad_(True)
    model.to(device)

    # Projeksiyon başını 128 boyutlu çıktı için oluştur
    proj_head = ImageProjHead(in_dim=512, out_dim=128).to(device) #512 yap tekrar eğit sonra

    # 6. Veri Yükleyici (DataLoader)
    dataset = ShoesDataset(DATA_ROOT, os.path.join(os.getcwd(), "ayakkabilar.jsonl"))

    # Aynı etikete sahip örneklerin aynı batch'te olmasını sağlayan sampler
    # Not: Bu, SupConLoss'un çalışması için kritik
    from torch.utils.data.sampler import BatchSampler, RandomSampler
    sampler = BatchSampler( #veri kümesindeki ürünleri batchliyo , randomsampler :rastgele seçiyo overfitting azaltıyo
        RandomSampler(dataset, replacement=False),
        batch_size=BATCH_SIZE,
        drop_last=False
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0) #ana işlemde yapıyo veri yüklemeyi 0 sa , sayı fazlaysa işçide fazla

    # 7. Optimizer ve Kayıp Fonksiyonu
    optimizer = AdamW(  #öğrenme hızını ayarlıyo genelleme sağlıyo overfitting yapmaz model hızlı ve etkili eğitim
        list(model.vision_model.parameters()) + list(proj_head.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = SupConLoss(temperature=0.07) #benzerlikleri daha hassas ayırt ediyo

    # 8. Eğitim Döngüsü
    print("Eğitim başlıyor...")
    for epoch in range(NUM_EPOCHS):  #epoch modelin eğitim kümesini baştan sonra görmesi
        model.train()
        proj_head.train()
        total_loss = 0  #kaybı takip edebilmek
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")  #pbariçin kütüphane eğitimmsürecini daha rahat gözlemlemek için tqdm 
        for i, (images, labels, _) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():  #automatic mix pression eğitim süreci hızlanır
                image_features = model.get_image_features(pixel_values=images)
                image_features = proj_head(image_features)
                
                # `SupConLoss` için normalize et
                image_features = torch.nn.functional.normalize(image_features, dim=1)
                
                loss = criterion(image_features, labels)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (i+1)})   #ilerleme sürecinin sonuna ek bilgiler getiriyo
            # Yeni eklenen kısım: En iyi modeli kaydetme kontrolü
        if epoch == 2:  # 3. epoch bittiğinde (epoch'lar 0'dan başladığı için)
            print("\n3. epoch bitti, en iyi modeli kaydediyor...")
            model.save_pretrained("./finetuned_clip_model")
            torch.save(proj_head.state_dict(), "./proj_head.pth")
        
        print(f"\nEpoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    print("\nEğitim tamamlandı! Modeller kaydediliyor...")
    model.save_pretrained("./finetuned_clip_model")
    torch.save(proj_head.state_dict(), "./proj_head.pth")
    print("Modeller 'finetuned_clip_model' ve 'proj_head.pth' olarak kaydedildi. ✅")
