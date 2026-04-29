import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

# ============================================================
# 1. Construction du vocabulaire
# ============================================================
def build_vocab(texts, max_vocab=10000):
    word_count = {}
    for text in texts:
        for word in text.split():
            word_count[word] = word_count.get(word, 0) + 1
    
    # Trier par fréquence
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in sorted_words[:max_vocab - 2]:
        vocab[word] = len(vocab)
    return vocab

# ============================================================
# 2. Classe Dataset PyTorch
# ============================================================
class MedicalDataset(Dataset):
    def __init__(self, csv_path, vocab=None, max_len=300):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['clean_text'])
        label = int(self.df.iloc[idx]['label'])
        tokens = [self.vocab.get(w, 1) for w in text.split()[:self.max_len]]
        return tokens, label

# ============================================================
# 3. Collate function (padding)
# ============================================================
def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)
    padded = [t + [0] * (max_len - len(t)) for t in texts]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# ============================================================
# 4. Test rapide
# ============================================================
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # Charger train pour construire le vocab
    train_df = pd.read_csv('data/train.csv')
    vocab = build_vocab(train_df['clean_text'].astype(str))
    print(f"Taille du vocabulaire : {len(vocab)}")

    # Sauvegarder le vocab
    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # Créer les datasets
    train_set = MedicalDataset('data/train.csv', vocab)
    val_set   = MedicalDataset('data/val.csv', vocab)
    test_set  = MedicalDataset('data/test.csv', vocab)

    # Créer les DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Test : charger un batch
    texts, labels = next(iter(train_loader))
    print(f"Batch textes : {texts.shape}")
    print(f"Batch labels : {labels.shape}")
    print("✅ DataLoaders OK !")
    