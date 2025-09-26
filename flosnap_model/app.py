import streamlit as st
import torch
import os
import numpy as np
import json
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch.nn as nn

# 1. Sabitler ve Ayarlar
MODEL_PATH = "./finetuned_clip_model"
PROJ_HEAD_PATH = "./proj_head.pth"
EMBEDDINGS_FILE = "embeddings.pt"
DATASET_FILE = "../flosnap_model/ayakkabilar.jsonl"
RESULT_COUNT = 12
IMAGES_FOLDER = os.path.join("..", "ayakkabi_scraper", "ayakkabi_resimleri")
SHOE_SIMILARITY_THRESHOLD = 0.33

# 2. Projeksiyon Başı
class ImageProjHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# 3. Model ve Özellikleri Yükleme
@st.cache_resource
def load_model_and_features():
    st.info("Model ve özellikler yükleniyor...")
    try:
        model = CLIPModel.from_pretrained(MODEL_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)

        proj_head = ImageProjHead(in_dim=512, out_dim=128)
        proj_head.load_state_dict(torch.load(PROJ_HEAD_PATH, map_location=torch.device('cpu')))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        proj_head.to(device).eval()

        if not os.path.exists(EMBEDDINGS_FILE):
            st.error(f"Hata: '{EMBEDDINGS_FILE}' bulunamadı. 'predict.py' ile oluşturun.")
            st.stop()

        all_embeddings, all_image_paths = torch.load(EMBEDDINGS_FILE, map_location=device)

        st.success("Model ve özellikler başarıyla yüklendi! ✅")
        return model, processor, proj_head, all_embeddings, all_image_paths, device
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()

# 4. Yardımcı Fonksiyonlar
def get_image_embedding(model, proj_head, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = proj_head(image_features)
    return image_features / image_features.norm(dim=-1, keepdim=True)

def get_text_embedding(model, proj_head, processor, text, device):
    if not text:
        return None
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)
        text_features = proj_head(text_features)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def load_metadata():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Hata: Veri bulunamadı: {DATASET_FILE}")
    metadata = {}
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            metadata[str(item.get('product_id'))] = item
    return metadata

# 5. Streamlit Arayüzü
st.title("👟 Ayakkabı Görsel Arama Motoru")
st.markdown("Aradığınız ayakkabıya en çok benzeyen ürünleri bulmak için bir resim yükleyin.")
st.markdown("---")

model, processor, proj_head, all_embeddings, all_image_paths, device = load_model_and_features()
metadata_dict = load_metadata()

uploaded_file = st.file_uploader("1. Bir resim yükleyin...", type=["jpg","jpeg","png"])
text_query = st.text_input("2. (İsteğe Bağlı) Arama kelimeleri girin (örn: 'siyah, erkek, spor').")

weight = st.slider(
    "3. Görsel Ağırlığı",
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="1.0: Sadece görseli kullan. 0.0: Sadece metni kullan. 0.5: İkisinin ortalamasını al."
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Resim", use_container_width=True)

    st.write("##### ✨ En Benzer Ürünler:")

    with st.spinner("Benzer ürünler aranıyor..."):
        # Görsel embedding
        query_embedding = get_image_embedding(model, proj_head, processor, image, device)

        # Ayakkabı kontrolü
        shoe_text_embedding = get_text_embedding(model, proj_head, processor, "ayakkabı", device)
        shoe_similarity = (query_embedding @ shoe_text_embedding.T).squeeze().item()
        st.info(f"Yüklenen görselin 'ayakkabı' benzerlik skoru: {shoe_similarity:.4f}")
        if shoe_similarity < SHOE_SIMILARITY_THRESHOLD:
            st.warning("Yüklenen görsel bir ayakkabı değil. Lütfen ayakkabı yükleyin.")
            st.stop()

        # Görsel benzerliği
        visual_similarities = (query_embedding @ all_embeddings.T).squeeze(0)

        # Metin embedding (örn. renk)
        text_embedding = get_text_embedding(model, proj_head, processor, text_query, device)
        if text_embedding is not None and text_query.strip():
            text_similarities = (text_embedding @ all_embeddings.T).squeeze(0)
            combined_scores = (weight * visual_similarities) + ((1 - weight) * text_similarities)
        else:
            combined_scores = visual_similarities

        # En yüksek skorlara göre sıralama
        top_indices = torch.topk(combined_scores, k=len(all_embeddings), dim=-1).indices

        filtered_results = []
        seen_product_ids = set()

        for idx in top_indices:
            image_path = all_image_paths[idx]
            product_id = os.path.basename(os.path.dirname(image_path))
            if product_id in seen_product_ids:
                continue
            item_meta = metadata_dict.get(product_id)
            if item_meta:
                # Metin filtreleme: weight çok düşükse katı filtre uygula
                if text_query:
                    query_lower = text_query.lower()
                    meta_text = f"{item_meta.get('brand','')} {item_meta.get('gender','')} {item_meta.get('color','')} {item_meta.get('product_name','')}".lower()
                    if not (query_lower in meta_text) and weight < 0.3:
                        continue
                if item_meta.get('image_urls'):
                    first_image_url = item_meta['image_urls'][0]
                    local_main_image_path = os.path.join(IMAGES_FOLDER, product_id, os.path.basename(first_image_url))
                else:
                    continue
                filtered_results.append({
                    'path': local_main_image_path,
                    'similarity': combined_scores[idx].item(),
                    'metadata': item_meta
                })
                seen_product_ids.add(product_id)
                if len(filtered_results) >= RESULT_COUNT:
                    break

    # Sonuçları göster
    cols_per_row = 4
    cols = st.columns(cols_per_row)
    for i, img in enumerate(filtered_results):
        with cols[i % cols_per_row]:
            with st.container(border=True):
                img_path = img['path']
                item_meta = img['metadata']
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning("Resim bulunamadı.")
                st.markdown(f"**Marka:** {item_meta.get('brand','Bilinmiyor')}")
                st.markdown(f"**Cinsiyet:** {item_meta.get('gender','Bilinmiyor')}")
                st.markdown(f"**Renk:** {item_meta.get('color','Bilinmiyor')}")
                st.markdown(f"**Benzerlik Skoru:** {img['similarity']:.4f}")
                if 'url' in item_meta and item_meta['url']:
                    st.link_button("Ürüne Git 🔗", url=item_meta['url'])
else:
    st.info("Lütfen arama yapmak için bir resim yükleyin. 👆")

