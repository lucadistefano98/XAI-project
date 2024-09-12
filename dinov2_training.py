import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import Dinov2ForImageClassification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter

# Percorsi dei dati e della cartella di output
data_dir = "/home/ldistefan/project/data"
output_dir = "/home/ldistefan/project/output"
batch_size = 32

# Data augmentation per il training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),  # Flip orizzontale casuale
    RandomVerticalFlip(p=0.5),    # Flip verticale casuale
    RandomRotation(degrees=15),   # Rotazione casuale entro ±15 gradi
    ColorJitter(brightness=0.1, contrast=0.1),  # Modifica casuale di luminosità e contrasto
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# trasformazioni per validation set senza aug 
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Caricamento dei dataset
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform) #se non voglio aug ci metto val_transform
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "validation"), transform=val_transform)

# Determinazione automatica del numero di classi
num_classes = len(train_dataset.classes)
print(f"Numero di classi rilevate: {num_classes}")

# Caricamento del modello con il numero  di classi
model = Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-small-imagenet1k-1-layer",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

# Config del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ottimizzatore, loss function e scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

# Variabili per il ciclo di addestramento
epochs = 100
best_acc = 0.0
patience_s = 25
early_stop_counter = 0
metrics = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

# Loop di addestramento
for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    val_loss, val_correct = 0.0, 0

    # Loop di addestramento con barra di avanzamento
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    # Loop di validazione
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    # Calcolo delle metriche
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct.double() / len(val_loader.dataset)

    # Salvataggio delle metriche
    metrics["epoch"].append(epoch + 1)
    metrics["train_loss"].append(train_loss)
    metrics["val_loss"].append(val_loss)
    metrics["train_acc"].append(train_acc.item())
    metrics["val_acc"].append(val_acc.item())

    # Stampa delle metriche per l'epoca corrente
    print(f'Epoch {epoch+1}/{epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Aggiornamento del learning rate
    scheduler.step(val_acc)

    # Controllo per early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_dinov2_aug.pth'))  #best_model_dinov2.pth
        early_stop_counter = 0  # Reset del contatore se troviamo una nuova best accuracy
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience_s:
            print(f"Early stopping attivato dopo {epoch+1} epoche")
            break

# Salva le metriche in un file CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(output_dir, 'training_metrics_dinov2_aug.csv'), index=False)  #training_metrics_dinov2.csv

# Generazione della matrice di confusione normalizzata
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(pixel_values=inputs).logits
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=train_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(output_dir, 'confusion_matrix_dinov2_aug.png'))  #confusion_matrix_dinov2.png
plt.close()

print("fine")