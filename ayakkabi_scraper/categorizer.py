import jsonlines
import os
import re

def categorize_product(name, keywords):
    """
    Ürün adını anahtar kelimelerle eşleştirerek kategori atar.
    """
    if name is None:
        return "diğer"
    
    name = name.lower()
    for category, word_list in keywords.items():
        for word in word_list:
            if re.search(r'\b' + re.escape(word) + r'\b', name):
                return category
    return "diğer"

def process_data(input_path, output_path, keywords):
    """
    Tüm veri setini okur, her ürünü kategorize eder ve yeni bir dosya olarak kaydeder.
    """
    categorized_count = 0
    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode='w') as writer:
        for item in reader:
            name = item.get('name', '')
            category = categorize_product(name, keywords)
            item['category'] = category
            writer.write(item)
            categorized_count += 1
    return categorized_count

if __name__ == "__main__":
    input_file = "ayakkabilar.jsonl"
    output_file = "ayakkabilar_categorized.jsonl"

    # Kategorize etmek için anahtar kelimeler
    category_keywords = {
        "sandalet": ["sandalet", "sandal"],
        "terlik": ["terlik", "babet", "plaj"],
        "spor_ayakkabi": ["ayakkabi", "sneaker", "spor", "koşu", "yürüyüş", "training"],
        "bot": ["bot", "çizme", "postal"],
        "topuklu": ["topuklu"],
        "günlük": ["günlük"]
    }

    print("Ürünler otomatik olarak kategorize ediliyor...")
    
    # Dosya yollarını doğru ayarla
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_file)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)

    processed_items = process_data(input_path, output_path, category_keywords)
    
    if processed_items > 0:
        print(f"Başarıyla {processed_items} ürün kategorize edildi ve '{output_file}' dosyasına kaydedildi.")
    else:
        print("Hata: Hiçbir ürün bulunamadı veya işlenemedi.")