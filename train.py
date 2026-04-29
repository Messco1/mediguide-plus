import torch
import torch.nn as nn
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import MedicalDataset, collate_fn

# ============================================================
# 1. Architecture du modèle
# ============================================================
class MediGuideModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_specialties, num_urgency=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_specialty = nn.Linear(hidden_dim * 2, num_specialties)
        self.fc_urgency = nn.Linear(hidden_dim * 2, num_urgency)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        # Global max pooling
        pooled = torch.max(lstm_out, dim=1).values
        pooled = self.dropout(pooled)
        specialty = self.fc_specialty(pooled)
        urgency = self.fc_urgency(pooled)
        return specialty, urgency

# ============================================================
# 2. Règles d'urgence automatiques
# ============================================================
URGENCY_MAP = {
    'Emergency Room Reports': 3,
    'Surgery': 2,
    'Cardiovascular / Pulmonary': 2,
    'Neurology': 2,
    'Neurosurgery': 2,
    'Hematology - Oncology': 2,
    'Pediatrics - Neonatal': 2,
}

def get_urgency(specialty):
    return URGENCY_MAP.get(specialty.strip(), 1)

# ============================================================
# 3. Chargement des données
# ============================================================
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('data/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

train_df = pd.read_csv('data/train.csv')
val_df   = pd.read_csv('data/val.csv')

# Ajouter colonne urgence
for df in [train_df, val_df]:
    df['urgency'] = df['medical_specialty'].apply(get_urgency)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)

train_set = MedicalDataset('data/train.csv', vocab)
val_set   = MedicalDataset('data/val.csv', vocab)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ============================================================
# 4. Initialisation du modèle
# ============================================================
NUM_SPECIALTIES = len(le.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device} | Spécialités : {NUM_SPECIALTIES}")

model = MediGuideModel(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=256,
    num_specialties=NUM_SPECIALTIES
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 5. Boucle d'entraînement
# ============================================================
EPOCHS = 20
best_val_acc = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss, correct, total = 0, 0, 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        urgency_labels = torch.zeros(labels.size(0), dtype=torch.long).to(device)

        optimizer.zero_grad()
        spec_out, urg_out = model(texts)
        loss = criterion(spec_out, labels) + 0.3 * criterion(urg_out, urgency_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (spec_out.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            urgency_labels = torch.zeros(labels.size(0), dtype=torch.long).to(device)
            spec_out, urg_out = model(texts)
            loss = criterion(spec_out, labels) + 0.3 * criterion(urg_out, urgency_labels)
            val_loss += loss.item()
            val_correct += (spec_out.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model/best_model.pt')
        print(f"  ✅ Meilleur modèle sauvegardé (val_acc={val_acc:.3f})")

# ============================================================
# 6. Courbes
# ============================================================
import os
os.makedirs('model', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss')
ax1.legend()

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy')
ax2.legend()

plt.savefig('model/training_curves.png')
print(f"\n✅ Entraînement terminé ! Meilleure val_acc : {best_val_acc:.3f}")
print("Courbes sauvegardées dans model/training_curves.png")