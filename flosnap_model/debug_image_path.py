# Bu betik, app.py'nin görsel yolunu doğru oluşturup oluşturmadığını kontrol eder.
import json
import os
import faiss
import numpy as np

# Dosya yollarını tanımla (app.py'dekiyle aynı olmalı)
DATASET_FILE = "ayakkabilar.jsonl"
IMAGES_FOLDER = os.path.join("..", "ayakkabi_scraper", "ayakkabi_resimleri")
FAISS_INDEX_FILE = "flo_index.faiss"

def debug_image_path():
    """
    FAISS'ten bir örnek ürün çekip yerel görsel yolunu doğrular.
    """
    print("Görsel Yolu Hata Ayıklama Aracı...")
    print("-----------------------------------")
    
    # 1. Veri setini ve FAISS indeksini yükle
    if not os.path.exists(DATASET_FILE) or not os.path.exists(FAISS_INDEX_FILE):
        print(f"HATA: '{DATASET_FILE}' veya '{FAISS_INDEX_FILE}' dosyası bulunamadı.")
        return

    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    
    # Her bir görselin hangi ürüne ait olduğunu gösteren bir liste oluştur.
    # Bu, FAISS indeksindeki her bir görsel için doğru ürünü bulmamızı sağlayacak.
    image_to_product_data = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            product = json.loads(line)
            # Her bir görsel için o ürünü tekrar tekrar listeye ekle.
            for _ in product.get('image_urls', []):
                image_to_product_data.append(product)
    
    # 2. FAISS'ten rastgele bir sorgu yap.
    # FAISS indeksinin gerçek boyutunu kullanıyoruz.
    num_images_in_index = faiss_index.ntotal
    if num_images_in_index == 0:
        print("HATA: FAISS indeksinde hiç özellik vektörü bulunmuyor.")
        return
        
    random_idx = np.random.randint(0, num_images_in_index)
    
    # Rastgele bir vektörü alıp FAISS'te arama yap
    dummy_query_vector = faiss_index.reconstruct(random_idx).reshape(1, -1)
    
    k = 5
    distances, indices = faiss_index.search(dummy_query_vector, k)
    
    # 3. İlk sonucun görsel yolunu oluştur.
    # Şimdi FAISS indeksinden gelen indeksi kullanarak doğru ürünü buluyoruz.
    image_index = indices[0][0]
    result_item = image_to_product_data[image_index]
    
    product_id = result_item.get('product_id')
    image_urls = result_item.get('image_urls', [])
    
    # 4. Doğru görsel URL'sini al.
    # Hangi görselin eşleştiğini bulmak için, o ürüne ait tüm görselleri tekrar FAISS'te
    # arayıp en yakın olanı bulmalıyız.
    # Basitlik için sadece ilk görseli alıyoruz.
    if not image_urls:
        print(f"HATA: Ürün ID: {product_id} için görsel URL'si yok.")
        return
        
    first_image_url = image_urls[0]
    image_name = os.path.basename(first_image_url)
    local_image_path = os.path.join(IMAGES_FOLDER, product_id, image_name)
    
    print("\n-----------------------------------")
    print(f"Test edilen Ürün ID: {product_id}")
    print(f"Oluşturulan yerel görsel yolu: {local_image_path}")

    # 5. Oluşturulan yolun varlığını kontrol et.
    if os.path.exists(local_image_path):
        print("\nSONUÇ: Başarılı! Görsel dosyası bu yolda bulundu.")
        print("Artık app.py'ye odaklanabiliriz. Muhtemelen Streamlit veya başka bir mantıkla ilgili bir sorun var.")
    else:
        print("\nSONUÇ: HATA! Görsel dosyası bu yolda bulunamadı.")
        print("Bu, 'app.py' dosyasının yerel görsel yolunu doğru oluşturmada sorun yaşadığını gösterir.")
        print(f"Lütfen '{IMAGES_FOLDER}' klasörünün doğru konumda olduğundan ve içeride 'ayakkabi_resimleri' klasörünün bulunduğundan emin olun.")


if __name__ == "__main__":
    debug_image_path()
