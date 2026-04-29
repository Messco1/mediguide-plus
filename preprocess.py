import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Chargement
df = pd.read_csv('data/mtsamples.csv')
print(f"Shape initial : {df.shape}")

# 2. Suppression des lignes sans transcription
df = df.dropna(subset=['transcription', 'medical_specialty'])
df = df[df['transcription'].str.strip() != '']
print(f"Shape après nettoyage : {df.shape}")

# 3. Nettoyage du texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['transcription'].apply(clean_text)

# 4. Nettoyage spécialités
df['medical_specialty'] = df['medical_specialty'].str.strip()

# 5. Encodage des labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['medical_specialty'])
print(f"\nNombre de spécialités : {len(le.classes_)}")
print(f"Spécialités : {list(le.classes_)}")

# 6. Split train/val/test 70/15/15
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

print(f"\nTrain : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

# 7. Sauvegarde
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

with open('data/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✅ Données prêtes ! Fichiers sauvegardés dans data/")