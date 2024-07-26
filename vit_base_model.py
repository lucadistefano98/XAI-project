import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  

# dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# trasformazioni per i dati di addestramento e di test
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Caricamento dei dati
train_dataset = datasets.ImageFolder('/home/ldistefan/project/data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('/home/ldistefan/project/data/validation', transform=test_transform)
test_dataset = datasets.ImageFolder('/home/ldistefan/project/data/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Verifica numero di immagini nel dataset
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images: {len(test_dataset)}")

# Configurazione del modello
num_classes = len(train_dataset.classes)
config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, output_attentions=True)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=config, ignore_mismatched_sizes=True)
model = model.to(device)
print('configurazione modello completata')

#  funzioni di perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Inizializzazione dello scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

#  parametri per l'early stopping
early_stopping_patience = 5
early_stopping_counter = 0

# addestramento
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training")  # Barra di avanzamento
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Aggiorna descrizione barra di avanzamento
        progress_bar.set_postfix(loss=running_loss / total, accuracy=correct / total)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# validazione
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(val_loader, desc="Validation")  # Barra di avanzamento
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Aggiorna descrizione barra di avanzamento
            progress_bar.set_postfix(loss=running_loss / total, accuracy=correct / total)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Ciclo di addestramento
num_epochs = 20
best_acc = 0.0
print('Inizio addestramento...')
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Step dello scheduler con validation accuracy
    scheduler.step(val_acc)
    print(f"Current learning rate: {scheduler.get_last_lr()}")
    
    # Early stopping check
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, '/home/ldistefan/project/best_vit_model.pt')
        torch.save(model.state_dict(), '/home/ldistefan/project/best_vit_model.pth')
        print('Modello migliore salvato con accuracy:', best_acc)
        early_stopping_counter = 0  # Reset counter quando trova un modello migliore
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

# Funzione per generare e salvare confusion matrix e accuracy
def save_metrics(model, test_loader, device, output_dir='/home/ldistefan/project/output'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds, normalize='true')  # Normalizzazione della confusion matrix
    acc = accuracy_score(all_labels, all_preds)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'confusion_matrix_vitbase.npy'), cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_vitbase.png'))
    plt.close()
    
    with open(os.path.join(output_dir, 'accuracy.txt'), 'w') as f:
        f.write(f'Accuracy: {acc:.4f}\n')
    
    print(f'Confusion matrix and accuracy saved in {output_dir}')

# Valutazione finale sul test set
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# Salvataggio confusion matrix e accuracy
save_metrics(model, test_loader, device)