import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, AdamW
from tqdm import tqdm
import jsonlines

# ---------------------------
# Ayarlar
# ---------------------------
MODEL_PATH = "./finetuned_clip_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-5
TEMPERATURE = 0.07

PRODUCTS_FOLDER = "../ayakkabi_scraper/ayakkabi_resimleri"
JSONL_PATH = "../ayakkabi_scraper/ayakkabilar.jsonl"

# ---------------------------
# Dataset
# ---------------------------
class ShoesDataset(Dataset):
    def __init__(self, products_folder, json_data):
        self.items = []
        self.json_data = json_data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for folder in os.listdir(products_folder):
            folder_path = os.path.join(products_folder, folder)
            if not os.path.isdir(folder_path):
                continue
            img_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                         if f.lower().endswith(("jpg", "jpeg", "png"))]
            if not img_files:
                continue
            text = self.get_text(folder)
            if text is None:
                continue
            # Sadece ilk resmi alıyoruz
            self.items.append({"image": img_files[0], "text": text})

    def get_text(self, product_id):
        for item in self.json_data:
            if str(item.get("product_id")) == product_id:
                parts = [
                    item.get("gender", ""),
                    item.get("brand", ""),
                    item.get("name", ""),
                    item.get("color", "")
                ]
                return " ".join(filter(None, parts))
        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]["image"]
        text = self.items[idx]["text"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # PIL -> Tensor
        return img, text

# ---------------------------
# SupCon Loss
# ---------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        device = features.device
        features = nn.functional.normalize(features, dim=1)
        batch_size = features.shape[0]

        if labels is None:
            labels = torch.arange(batch_size, device=device)
        else:
            labels = labels.to(device)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        exp_sim = torch.exp(similarity_matrix) * (1 - torch.eye(batch_size, device=device))
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return loss.mean()

# ---------------------------
# Model
# ---------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(DEVICE)

# ---------------------------
# Dataset ve DataLoader
# ---------------------------
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    reader = jsonlines.Reader(f)
    json_data = list(reader)

dataset = ShoesDataset(PRODUCTS_FOLDER, json_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# Optimizer ve Loss
# ---------------------------
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = SupConLoss(temperature=TEMPERATURE)

# ---------------------------
# Training Loop
# ---------------------------
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for images, texts in loop:
        images = images.to(DEVICE)
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True).to(DEVICE)
        image_inputs = {"pixel_values": images}

        optimizer.zero_grad()
        img_features = model.get_image_features(**image_inputs)
        txt_features = model.get_text_features(**text_inputs)

        combined = nn.functional.normalize((img_features + txt_features) / 2, dim=1)

        labels = torch.arange(combined.size(0)).to(DEVICE)
        loss = loss_fn(combined, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix({"loss": loss.item()})

    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

# ---------------------------
# Kaydet
# ---------------------------
os.makedirs(MODEL_PATH, exist_ok=True)
model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)
print(f"✅ Model kaydedildi: {MODEL_PATH}")



