import pandas as pd
import json
import os
import numpy as np
from PIL import Image

# Veri ve resim klasörlerinin yolunu belirt
data_path = os.path.join('..', 'ayakkabi_scraper', 'ayakkabilar.jsonl')
images_folder = os.path.join('..', 'ayakkabi_scraper', 'ayakkabi_resimleri')

def load_data_and_images(data_path, images_folder):
    """
    JSONL verisini Pandas'a yükler ve her ürüne ait resimleri işler.
    """
    try:
        df = pd.read_json(data_path, lines=True)
    except FileNotFoundError:
        print(f"Hata: {data_path} dosyası bulunamadı.")
        return None, None

    image_arrays = []
    product_data = []

    for index, row in df.iterrows():
        product_id = str(row['product_id'])
        product_folder = os.path.join(images_folder, product_id)
        
        if not os.path.exists(product_folder):
            continue
            
        for img_name in os.listdir(product_folder):
            if img_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG')):
                try:
                    img_path = os.path.join(product_folder, img_name)
                    img = Image.open(img_path).convert('RGB')
                    
                    img = img.resize((224, 224))
                    
                    img_array = np.array(img)
                    
                    image_arrays.append(img_array)
                    product_data.append(row.to_dict())
                
                except Exception as e:
                    print(f"Hata: {img_path} resmi işlenemedi. Hata: {e}")

    return np.array(image_arrays), pd.DataFrame(product_data)

# Ana fonksiyonu çağır ve veriyi yükle
images_data, products_df = load_data_and_images(data_path, images_folder)

if images_data is not None:
    print("Veri başarıyla yüklendi!")
    print(f"Yüklenen toplam resim sayısı: {len(images_data)}")
    print(f"Tüm resim verisinin boyutu: {images_data.shape}")
    print("\nEşleştirilen metin verisi:")
    print(products_df.head())