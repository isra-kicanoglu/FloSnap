import json
import os

DATASET_FILE = "ayakkabilar.jsonl"
IMAGES_FOLDER = os.path.join("..", "ayakkabi_scraper", "ayakkabi_resimleri")

def check_images_exist():
    """
    JSONL dosyasındaki her ürün için görsel dosyalarının diskte varlığını kontrol eder.
    """
    total_products = 0
    products_with_missing_images = 0
    
    print("Görsel dosyaları kontrol ediliyor...")
    print("-----------------------------------")
    
    if not os.path.exists(DATASET_FILE):
        print(f"HATA: '{DATASET_FILE}' dosyası bulunamadı.")
        return

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_products += 1
            try:
                item = json.loads(line)
                product_id = item.get('product_id')
                image_urls = item.get('image_urls', [])

                if not image_urls:
                    print(f"Ürün ID: {product_id} için görsel URL'si yok.")
                    products_with_missing_images += 1
                    continue

                first_image_url = image_urls[0]
                image_name = os.path.basename(first_image_url)
                local_image_path = os.path.join(IMAGES_FOLDER, product_id, image_name)
                
                if not os.path.exists(local_image_path):
                    print(f"EKSİK: Ürün ID: {product_id} için görsel '{local_image_path}' bulunamadı.")
                    products_with_missing_images += 1
            
            except json.JSONDecodeError:
                print(f"HATA: Satır JSON formatında değil.")
            except Exception as e:
                print(f"HATA: İşlem sırasında bir sorun oluştu: {e}")

    print("\n-----------------------------------")
    print(f"Toplam ürün sayısı: {total_products}")
    print(f"Görseli eksik olan ürün sayısı: {products_with_missing_images}")
    print(f"Görseli eksik olmayan ürün sayısı: {total_products - products_with_missing_images}")
    
    if products_with_missing_images > 0:
        print("\nÇÖZÜM: 'ayakkabi_scraper' projenizdeki indirme betiğini kontrol edin ve eksik görselleri indirin.")
        print("İndirme işlemi bittikten sonra, FAISS indeksini yeniden oluşturmayı unutmayın.")
    else:
        print("\nTebrikler! Tüm görseller diskte bulunuyor.")
        print("Model veya FAISS indeksi ile ilgili bir sorun olabilir. Onları yeniden oluşturmayı deneyelim.")


if __name__ == "__main__":
    check_images_exist()