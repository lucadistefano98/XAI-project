import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import Dinov2ForImageClassification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
from torchvision.ops import box_iou

# Caricamento del file dataset.yaml
with open("/home/ldistefan/project/data2/dataset.yaml", "r") as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)

# Estrazione dei percorsi e dei nomi delle classi dal file YAML
base_path = dataset_config['path']
train_dir = os.path.join(base_path, dataset_config['train'])
val_dir = os.path.join(base_path, dataset_config['val'])
test_dir = os.path.join(base_path, dataset_config['test'])
class_names = list(dataset_config['names'].values())  

output_dir = '/home/ldistefan/project/output'

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, img_size=(224, 224)):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size

        # Ottieni solo le immagini che hanno un file di etichetta corrispondente
        self.images = [
            img for img in os.listdir(self.img_dir)
            if os.path.exists(os.path.join(self.label_dir, img.replace('.png', '.txt')))
        ]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.png', '.txt'))

        # Caricamento dell'immagine
        image = Image.open(img_path).convert('RGB')

        try:
            with open(label_path, 'r', encoding='latin1') as f:
                label_data = f.readline().strip().split()

                if not label_data or not label_data[0].isdigit():
                    return self._handle_invalid_data(image)

                # Class ID e bounding box
                class_id = int(label_data[0])
                x_center, y_center, width, height = map(float, label_data[1:])
                img_w, img_h = self.img_size
                x_min = (x_center - width / 2) * img_w
                y_min = (y_center - height / 2) * img_h
                x_max = (x_center + width / 2) * img_w
                y_max = (y_center + width / 2) * img_h
                bbox = torch.tensor([x_min, y_min, x_max, y_max])

        except Exception as e:
            print(f"Errore nel file {label_path}: {e}")
            return self._handle_invalid_data(image)

        if self.transform:
            image = self.transform(image)

        return image, bbox, class_id

    def _handle_invalid_data(self, image):
        bbox = torch.tensor([0, 0, 0, 0])  # Bounding box vuoto
        class_id = -1  # Classe non valida
        if self.transform:
            image = self.transform(image)
        return image, bbox, class_id

    def _find_num_classes(self):
        class_ids = set()
        for label_file in os.listdir(self.label_dir):
            label_path = os.path.join(self.label_dir, label_file)
            with open(label_path, 'r', encoding='latin1') as f:
                label_data = f.readline().strip().split()
                if label_data and label_data[0].isdigit():
                    class_id = int(label_data[0])
                    class_ids.add(class_id)
        return len(class_ids)

# Estensione del modello DINOV2 per estrarre le attention maps
class Dinov2WithAttention(Dinov2ForImageClassification):
    def forward(self, images):
        outputs = super().forward(images, output_attentions=True)
        return outputs.logits, outputs.attentions[-1]  # Ultimo strato di attention maps


class AttentionBoundingBoxLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(AttentionBoundingBoxLoss, self).__init__()
        self.lambda_weight = lambda_weight  # Fattore che bilancia la penalità sull'IoU
        self.classification_loss_fn = nn.CrossEntropyLoss()  # Definisci la cross-entropy loss

    def forward(self, logits, labels, attention_maps, bounding_boxes):
        classification_loss = self.classification_loss_fn(logits, labels)

        if attention_maps is not None:
            predicted_boxes = self.attention_to_box(attention_maps)
            bounding_boxes = bounding_boxes.to(logits.device)
            predicted_boxes = predicted_boxes.to(logits.device)
            iou = box_iou(predicted_boxes, bounding_boxes).mean()  # Media dell'IoU
            iou_loss = 1 - iou  # Penalità
            total_loss = classification_loss + self.lambda_weight * iou_loss
        else:
            total_loss = classification_loss

        return total_loss

    def attention_to_box(self, attention_maps):
        threshold = 0.5
        coords = (attention_maps > threshold).nonzero(as_tuple=False)
        if coords.numel() == 0:  
            return torch.zeros((1, 4))  # Bounding box vuoto

        x_min, y_min = coords[:, 1].min().item(), coords[:, 2].min().item()
        x_max, y_max = coords[:, 1].max().item(), coords[:, 2].max().item()
        return torch.tensor([[x_min, y_min, x_max, y_max]])

# Parametri e Configurazione
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni per il dataset
train_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.Resize((224, 224)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset
train_dataset = CustomDataset(
    img_dir=os.path.join(base_path, "images/train"),  
    label_dir=os.path.join(base_path, "labels/train"), 
    transform=train_transform  
)
val_dataset = CustomDataset(
    img_dir=os.path.join(base_path, "images/val"),  
    label_dir=os.path.join(base_path, "labels/val"),  
    transform=val_transform
)
test_dataset = CustomDataset(
    img_dir=os.path.join(base_path, "images/test"),  
    label_dir=os.path.join(base_path, "labels/test"),  
    transform=test_transform
)

# Numero di classi
num_classes = train_dataset._find_num_classes()
print(f"Numero di classi rilevate: {num_classes}")

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Modello DINOV2 con numero di classi rilevato
model = Dinov2WithAttention.from_pretrained(
    "facebook/dinov2-small-imagenet1k-1-layer",
    num_labels=num_classes,  
    ignore_mismatched_sizes=True
)
model.to(device)

# Ottimizzatore e scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = AttentionBoundingBoxLoss(lambda_weight=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

# Ciclo di addestramento
epochs = 100  
best_val_acc = 0.0  
metrics = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

for epoch in range(epochs):
    model.train()
    train_loss, val_loss = 0.0, 0.0
    train_correct, val_correct = 0, 0

    for images, bounding_boxes, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        images, labels, bounding_boxes = images.to(device), labels.to(device), bounding_boxes.to(device)

        valid_mask = labels != -1
        images, labels, bounding_boxes = images[valid_mask], labels[valid_mask], bounding_boxes[valid_mask]

        if labels.size(0) == 0:  # Se non ci sono immagini valide
            continue

        optimizer.zero_grad()

        logits, attention_maps = model(images)
        loss = criterion(logits, labels, attention_maps, bounding_boxes)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        train_correct += torch.sum(preds == labels.data)

    train_acc = train_correct.double() / len(train_dataset)

    # Validazione
    model.eval()
    with torch.no_grad():
        for images, bounding_boxes, labels in val_loader:
            images, labels, bounding_boxes = images.to(device), labels.to(device), bounding_boxes.to(device)

            valid_mask = labels != -1
            images, labels, bounding_boxes = images[valid_mask], labels[valid_mask], bounding_boxes[valid_mask]

            if labels.size(0) == 0:
                continue

            logits, attention_maps = model(images)
            val_loss += criterion(logits, labels, attention_maps, bounding_boxes).item() * images.size(0)
            _, preds = torch.max(logits, 1)
            val_correct += torch.sum(preds == labels.data)

    val_acc = val_correct.double() / len(val_dataset)

    # Stampa delle metriche
    print(f'Epoch {epoch+1}/{epochs}, '
          f'Train Loss: {train_loss/len(train_loader.dataset):.4f}, '
          f'Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss/len(val_loader.dataset):.4f}, '
          f'Val Acc: {val_acc:.4f}')

    # Aggiorna il best accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(output_dir, "best_mode_sperimentale.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Salvato il miglior modello con val_acc: {val_acc}")

    lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)

# Confusion Matrix
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, bounding_boxes, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        logits, attention_maps = model(images)
        _, preds = torch.max(logits, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

cm_test = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds), normalize='true')
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
disp_test.plot(cmap=plt.cm.Blues)

cm_test_output_path = os.path.join(output_dir, "confusion_matrix_sperimentale.png")
plt.savefig(cm_test_output_path)
plt.close()

print(f"Confusion matrix sui dati di test salvata in {cm_test_output_path}")