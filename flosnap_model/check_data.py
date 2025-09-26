import pandas as pd

try:
    # Veri paketini yükle
    df = pd.read_pickle("extracted_features.pkl")
    
    # Dosya hakkında özet bilgi ver
    print("--- Dosya Bilgisi ---")
    print(f"Toplam {len(df)} adet ürün verisi bulundu.")
    print(f"Veri setinin sütunları: {list(df.columns)}")
    print("-" * 20)
    
    # İlk ürünün verisini göster
    print("--- İlk Ürün Örneği ---")
    first_product = df.iloc[0]
    
    print(f"Product ID: {first_product['product_id']}")
    print(f"Brand: {first_product['brand']}")
    print(f"Name: {first_product['name']}")
    
    # 512 sayılık parmak izinin boyutunu göster
    if 'feature_vector' in df.columns:
        print(f"Özellik Vektörü Boyutu: {first_product['feature_vector'].shape}")
    
except FileNotFoundError:
    print("Hata: 'extracted_features.pkl' dosyası bulunamadı. Lütfen feature_extractor.py dosyasını çalıştırdığından emin ol.")
except KeyError:
    print("Hata: Dosyada beklenilen 'product_id' veya 'feature_vector' verisi yok.")