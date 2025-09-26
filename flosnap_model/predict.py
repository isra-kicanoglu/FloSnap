import torch
from torchvision import transforms
from PIL import Image
import os
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
#KAYDEDİLMİŞ EMBEDDİNG.PY ÜZERİNDEN SORGU YAPIYO MODEL YÜKLÜYO ANCAK EMBED ÇIAKRMIYO TEKRAR EMBED FEATURE_EXTRACTURE
# Sabitler
MODEL_SAVE_PATH = "./finetuned_clip_model"
PROJ_HEAD_PATH = "./proj_head.pth"
EMBEDDINGS_FILE = "embeddings.pt"

# Projeksiyon başı
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

# Model ve processor yükleme
def load_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_SAVE_PATH)
    processor = CLIPProcessor.from_pretrained(MODEL_SAVE_PATH)
    proj_head = ImageProjHead(in_dim=512, out_dim=128)
    proj_head.load_state_dict(torch.load(PROJ_HEAD_PATH, map_location=device))

    model.to(device)
    proj_head.to(device)
    model.eval()
    proj_head.eval()
    return model, proj_head, processor, device

# Tek görsel için embedding çıkarma
def get_image_embedding(model, proj_head, image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.get_image_features(pixel_values=image)
        features = proj_head(features)
    return features / features.norm(dim=-1, keepdim=True)

# Benzer görselleri bulma
def find_similar_images(query_embedding, all_embeddings, image_paths, top_k=5):
    similarities = (query_embedding @ all_embeddings.T).squeeze(0)
    top_indices = torch.topk(similarities, k=top_k).indices
    return [{'path': image_paths[i], 'similarity': similarities[i].item()} for i in top_indices]

if __name__ == '__main__':
    model, proj_head, processor, device = load_model_and_processor()

    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Hata: '{EMBEDDINGS_FILE}' bulunamadı. Önce feature_extractor.py ile embeddings oluşturun.")
        exit()

    all_embeddings, all_image_paths = torch.load(EMBEDDINGS_FILE, map_location=device)

    query_image_path = input("Aramak istediğiniz görselin yolunu girin: ")
    if not os.path.exists(query_image_path):
        print(f"Hata: {query_image_path} bulunamadı.")
        exit()

    query_embedding = get_image_embedding(model, proj_head, query_image_path, device)
    similar_images = find_similar_images(query_embedding, all_embeddings, all_image_paths, top_k=5)

    print("\n--- En Benzer Görseller ---")
    for i, img in enumerate(similar_images):
        print(f"{i+1}. Yol: {img['path']}  Benzerlik Skoru: {img['similarity']:.4f}")

