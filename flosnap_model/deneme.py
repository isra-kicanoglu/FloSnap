import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F

# -- DOSYA YOLLARINI DÜZENLEYİN --
# Lütfen aşağıdaki dosya yollarını kendi projenizdeki yollarla değiştirin
model_path = "finetuned_clip_model"
image_path_1 = "target.jpg"  # Örneğin, bir araba resmi
image_path_2 = "hakiki-deri-siyah-kadin-cizme-e107.1zsyhe04.jpg"  # Örneğin, bir ev resmi

# -- 1. MODEL VE PROCESSOR'I YÜKLEME --
try:
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    print("Model ve işlemci başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    exit()

# -- 2. GÖRSELLERİ YÜKLEME --
try:
    image1 = Image.open(image_path_1)
    image2 = Image.open(image_path_2)
    print("Görseller başarıyla yüklendi.")
except Exception as e:
    print(f"Görseller yüklenirken bir hata oluştu: {e}")
    exit()

# -- 3. GÖRSELLERİ İŞLEME VE EMBEDDING OLUŞTURMA --
# Modelden geçen her iki görsel için vektörleri alıyoruz
inputs1 = processor(images=image1, return_tensors="pt")
with torch.no_grad():
    embedding1 = model.get_image_features(**inputs1)

inputs2 = processor(images=image2, return_tensors="pt")
with torch.no_grad():
    embedding2 = model.get_image_features(**inputs2)

# -- 4. VEKTÖRLERİN BENZERLİĞİNİ HESAPLAMA --
# Kosinüs benzerliği ile iki vektörün ne kadar benzer olduğunu buluyoruz
# Değer 1'e ne kadar yakınsa, o kadar benzerdir.
similarity = F.cosine_similarity(embedding1, embedding2, dim=1).item()

print(f"\nİki görsel arasındaki benzerlik skoru: {similarity:.4f}")

# -- 5. SONUCU YORUMLAMA --
if similarity > 0.95:
    print("\nSonuç: Benzerlik skoru çok yüksek! 🚨")
    print("Bu, modelinizin her iki farklı görsel için neredeyse aynı vektörü ürettiği anlamına gelir.")
    print("Sorun muhtemelen **modelin kendisinde** veya eğitim aşamasındadır (model çökmesi).")
    print("Şimdi **`finetune_model.py`** dosyanızı ve eğitim hiperparametrelerini (özellikle öğrenme hızını) kontrol edin.")
elif similarity < 0.5:
    print("\nSonuç: Benzerlik skoru düşük. ✅")
    print("Bu, modelinizin görselleri doğru bir şekilde ayırt edebildiği anlamına gelir.")
    print("Sorun muhtemelen **`predict.py`** dosyanızdaki arama veya sıralama mantığındadır.")
else:
    print("\nSonuç: Benzerlik skoru orta seviyede.")
    print("Bu, modelin tam olarak doğru çalışmadığını ancak tamamen çökmediğini gösterir.")
    print("Yine de **`predict.py`** dosyanızdaki mantığı kontrol etmek iyi bir başlangıç olabilir.")