import torch
from torchvision import transforms
from PIL import Image
import os
import glob
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 1. Sabitler ve Ayarlar
DATA_ROOT = "../ayakkabi_scraper/ayakkabi_resimleri"
MODEL_SAVE_PATH = "./finetuned_clip_model"
PROJ_HEAD_PATH = "./proj_head.pth"
EMBEDDINGS_FILE = "embeddings.pt"

# 2. Model Bileşenleri
class ImageProjHead(torch.nn.Module):
    """
    Eğitilmiş Projeksiyon Başı.
    Bu sınıf, modelin eğitildiği dosyadakiyle aynı olmalıdır.
    """
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# 3. Model Yükleme ve Ayarlar
def load_model_and_processor():
    """Eğitilmiş modeli ve projeksiyon başını yükler."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Kullanılan Cihaz: {device}")
        
        # Eğitilmiş CLIP modelini yükle
        model = CLIPModel.from_pretrained(MODEL_SAVE_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_SAVE_PATH)
        
        # Projeksiyon başını yükle
        proj_head = ImageProjHead(in_dim=512, out_dim=128)
        proj_head.load_state_dict(torch.load(PROJ_HEAD_PATH, map_location=device))
        
        model.to(device)
        proj_head.to(device)
        model.eval()  #model tahmin yapmaya odaklanır
        proj_head.eval() 

        print("Eğitilmiş model ve projeksiyon başı başarıyla yüklendi.")
        return model, proj_head, processor, device
    except Exception as e:
        print(f"Hata: Modeller yüklenemedi. 'finetuned_clip_model' ve 'proj_head.pth' dosyalarını kontrol edin. Hata: {e}")
        return None, None, None, None

# 4. Görsel Özelliklerini (Embedding) Çıkarma
def get_image_embedding(model, proj_head, processor, image_path, device):
    """Verilen görselin özelliklerini (embedding) çıkarır."""
    if not os.path.exists(image_path):
        print(f"Hata: {image_path} bulunamadı.")
        return None
    
    image = Image.open(image_path).convert("RGB")
    
    # Girdi resminin normalleştirilmesi
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  #görüntüyü pytorcha dönüştürür
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    image = transform(image).unsqueeze(0).to(device) #unsqueeze 0 ile tensrun basına boyut ekleme
    
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=image)
        image_features = proj_head(image_features)
        
    return image_features / image_features.norm(dim=-1, keepdim=True)

# 5. Görsel Arama Fonksiyonu
def find_similar_images(query_embedding, all_embeddings, image_paths, top_k=5):
    """
    Sorgu görseline en benzer görselleri bulur.
    En benzeri bulmak için kosinüs benzerliği kullanılır.
    """
    # Kosinüs benzerliği (vektörlerin nokta çarpımı)
    similarities = (query_embedding @ all_embeddings.T).squeeze(0)
    
    # En yüksek benzerlik skorlarına sahip indeksleri al
    top_indices = torch.topk(similarities, k=top_k, dim=-1).indices
    
    # Benzer görsellerin yollarını ve skorlarını al
    similar_images = []
    for idx in top_indices:
        similar_images.append({
            'path': image_paths[idx],
            'similarity': similarities[idx].item()
        })
    
    return similar_images

def create_embeddings_database(model, proj_head, processor, device):
    """Tüm görsellerin özelliklerini hesaplar ve bir dosyaya kaydeder."""
    print("Veri setindeki tüm görsellerin özelliklerini çıkarıyor ve kaydediyor...")
    #recursive ile alt klasörlere de arama yap
    all_image_paths = glob.glob(os.path.join(DATA_ROOT, "**", "*.jpg"), recursive=True) + \  
                      glob.glob(os.path.join(DATA_ROOT, "**", "*.JPG"), recursive=True) + \
                      glob.glob(os.path.join(DATA_ROOT, "**", "*.jpeg"), recursive=True) + \
                      glob.glob(os.path.join(DATA_ROOT, "**", "*.JPEG"), recursive=True) + \
                      glob.glob(os.path.join(DATA_ROOT, "**", "*.png"), recursive=True) + \
                      glob.glob(os.path.join(DATA_ROOT, "**", "*.PNG"), recursive=True)
    
    if not all_image_paths:
        print("Hata: Veri setinde hiç görsel bulunamadı. Lütfen veri yolu doğru mu kontrol edin.")
        return None, None

    all_embeddings = []
    for img_path in tqdm(all_image_paths, desc="Görsel özellikleri hesaplanıyor"):
        emb = get_image_embedding(model, proj_head, processor, img_path, device)
        if emb is not None:
            all_embeddings.append(emb.squeeze(0))
    
    all_embeddings = torch.stack(all_embeddings).to(device)
    torch.save((all_embeddings, all_image_paths), EMBEDDINGS_FILE)
    print(f"Özellikler başarıyla '{EMBEDDINGS_FILE}' dosyasına kaydedildi. ✅")
    return all_embeddings, all_image_paths

if __name__ == '__main__':
    model, proj_head, processor, device = load_model_and_processor()
    if not model:
        exit()

    # Eğer özellikler dosyası yoksa oluştur
    if not os.path.exists(EMBEDDINGS_FILE):
        all_embeddings, all_image_paths = create_embeddings_database(model, proj_head, processor, device)
    else:
        print(f"'{EMBEDDINGS_FILE}' dosyası bulundu. Yükleniyor...")
        all_embeddings, all_image_paths = torch.load(EMBEDDINGS_FILE, map_location=device)
        print("Özellikler başarıyla yüklendi.")

    if all_embeddings is None or not all_image_paths:
        exit()
        
    print("Görsel arama başlıyor...")
    
    # Kullanıcıdan bir sorgu görseli seçmesini iste
    query_image_path = input("Aramak istediğiniz görselin yolunu girin (ör: ../ayakkabi_scraper/ayakkabi_resimleri/121855/121855_0_31.jpg): ")
    
    if not os.path.exists(query_image_path):
        print(f"Hata: Girilen görsel yolu bulunamadı: {query_image_path}")
    else:
        query_embedding = get_image_embedding(model, proj_head, processor, query_image_path, device)
        
        if query_embedding is not None:
            similar_images = find_similar_images(query_embedding, all_embeddings, all_image_paths, top_k=5)
            
            print("\n--- En Benzer Görseller ---")
            for i, img in enumerate(similar_images):
                print(f"{i+1}. Görsel:")
                print(f"  Yol: {img['path']}")
                print(f"  Benzerlik Skoru: {img['similarity']:.4f}\n")
