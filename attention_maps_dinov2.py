import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as pth_transforms
from PIL import Image
from transformers import Dinov2ForImageClassification

# Configurazione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percorsi dei file
output_dir = "/home/ldistefan/project/output"
os.makedirs(output_dir, exist_ok=True)

# Caricamento del modello addestrato
model_path = os.path.join(output_dir, 'best_model_dinov2_aug.pth')  #best_model_dinov2.pth
model = Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-small-imagenet1k-1-layer",
    num_labels=6,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Processamento dell'immagine
image_path = '/home/ldistefan/project/data/test/Adenoma/0a4a98cf-ef8a-474a-b814-4c4f3c3537a3.png'
image_size = (224, 224)
patch_size = 14

transform = pth_transforms.Compose([
    pth_transforms.Resize(image_size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Ottenimento delle attention maps
def get_attention_map(model, img_tensor, device):
    outputs = model(img_tensor, output_attentions=True)
    attentions = outputs.attentions[-1]  # Prende l'ultimo layer di attention
    nh = attentions.shape[1]  # numero di teste di attenzione

    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, img_tensor.shape[-2] // patch_size, img_tensor.shape[-1] // patch_size)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
    attentions = attentions.detach().cpu().numpy()

    return attentions

attentions = get_attention_map(model, img_tensor, device)

# Creazione della figura
num_heads = attentions.shape[0]
fig, axs = plt.subplots(1, num_heads + 1, figsize=(15, 5))

# Plot dell'immagine originale
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Plot delle attention heads con sovrapposizione dell'immagine originale
for j in range(num_heads):
    axs[j + 1].imshow(img, alpha=1)  # Mostra l'immagine originale con trasparenza
    axs[j + 1].imshow(attentions[j], cmap='viridis', alpha=0.4)  # Sovrappone la mappa di attenzione con trasparenza
    axs[j + 1].set_title(f'Att Head {j}')
    axs[j + 1].axis('off')

# Salvataggio della figura combinata
plt.tight_layout()
combined_image_path = os.path.join(output_dir, 'combined_attention_maps_dinov2_aug.png')  #combined_attention_maps_dinov2.png
plt.savefig(combined_image_path)
plt.show()

print(f'Attention maps combinate salvate in {combined_image_path}')