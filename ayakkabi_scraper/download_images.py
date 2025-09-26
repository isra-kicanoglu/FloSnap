import json
import requests
import os

def download_images(jsonl_file, output_folder="ayakkabi_resimleri"):
    """
    JSONL dosyasındaki URL'lerden resimleri indirir ve düzenler.
    Sadece daha önce indirilmemiş olanları indirmeyi sağlar.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' klasörü oluşturuldu.")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                product_id = item.get("product_id")
                image_urls = item.get("image_urls", [])

                if not product_id or not image_urls:
                    continue

                product_folder = os.path.join(output_folder, product_id)
                
                # Klasör zaten varsa, bu ürünün resimleri daha önce indirilmiştir.
                if os.path.exists(product_folder):
                    print(f"Ürün {product_id} zaten indirilmiş, atlanıyor.")
                    continue

                os.makedirs(product_folder)
                print(f"Ürün {product_id} için {len(image_urls)} resim indiriliyor...")

                for url in image_urls:
                    try:
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        
                        file_name = os.path.basename(url)
                        file_path = os.path.join(product_folder, file_name)
                        
                        with open(file_path, 'wb') as img_file:
                            img_file.write(response.content)
                        
                    except requests.exceptions.RequestException as e:
                        print(f"  -> Hata: {url} indirilemedi. Hata: {e}")

            except json.JSONDecodeError as e:
                print(f"JSON okuma hatası: {e} - Hatalı satır: {line.strip()}")

# Ana işlem
jsonl_file_name = "ayakkabilar.jsonl"
download_images(jsonl_file_name)