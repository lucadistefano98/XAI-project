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
model_path = os.path.join(output_dir, 'best_model_dinov2_aug.pth')
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

# Funzione per ottenere la mappa di attenzione
def get_attention_map(model, img_tensor, device):
    outputs = model(img_tensor, output_attentions=True)
    attentions = outputs.attentions[-1]  # Prende l'ultimo layer di attention
    nh = attentions.shape[1]  # numero di teste di attenzione

    # Reshape delle attenzioni per la visualizzazione
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, img_tensor.shape[-2] // patch_size, img_tensor.shape[-1] // patch_size)
    
    # Interpolazione delle attention maps per riscalarle alla dimensione originale dell'immagine
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[0]
    attentions = attentions.detach().cpu().numpy()

    return attentions

attentions = get_attention_map(model, img_tensor, device)

# Visualizzazione
fig, axes = plt.subplots(1, attentions.shape[0] + 1, figsize=(20, 5))

# Mostra immagine originale
axes[0].imshow(np.array(img))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Sovrapposizione delle attention maps all'immagine originale
for i, attn in enumerate(attentions):
    axes[i+1].imshow(np.array(img), alpha=0.5)  # Visualizza l'immagine originale
    axes[i+1].imshow(attn, cmap='jet', alpha=0.5)  # Sovrappone la mappa di attenzione
    axes[i+1].set_title(f'Att Head {i}')
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()