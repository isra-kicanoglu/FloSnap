# FloSnap

**Türkçe:**  
Bu proje, kullanıcı tarafından yüklenen bir ayakkabıyı sistemdeki katalogdaki en benzer ürünlerle eşleştiren bir uygulamadır.  

**English:**  
This project matches a user-uploaded shoe with the most similar products in the catalog.

---

## İçerik / Content

- `ayakkabi_scraper/` → Ayakkabı verilerini çekmek için scraper kodları / Scraper code to collect shoe data  
- `flosnap_model/` → Model eğitimi ve inferans için Python kodları / Python code for model training and inference  

---

## Önemli Dosyalar / Important Files

Bazı dosyalar GitHub’a eklenmemiştir çünkü çok büyük veya özel dosyalardır.  
Some files are not included in GitHub due to large size or privacy.

| Dosya/Klasör / File/Folder | Sebep / Reason | Ne Yapılmalı / Action Needed |
|-----------------------------|----------------|-----------------------------|
| `flosnap_model/finetuned_clip_model/model.safetensors` | Model dosyası (577 MB) / Model file (577 MB) | [Drive linkinden indirip](https://drive.google.com/file/d/1plNAlfUOFHUiv21nZqoglLW7nwkaPq9l/view?usp=sharing) `flosnap_model/finetuned_clip_model/` klasörüne koyun / Place it in the folder |
| `.venv/` | Sanal ortam / Virtual environment | Kendi bilgisayarında `python -m venv .venv` ile oluşturun / Create it on your own computer |
| `ayakkabi_scraper/ayakkabi_resimleri/` | Büyük resim dosyaları / Large image files | Kullanıcı verileri kendisi çekmeli / Users should download images themselves |
| `flosnap_model/large_model_files/` | Büyük model dosyaları / Large model files | Kullanıcı temin etmeli / Users should obtain them |

---

## Kurulum ve Çalıştırma / Installation & Usage

1. Repo’yu klonlayın / Clone the repository:  
```bash
git clone https://github.com/isra-kicanoglu/FloSnap.git
