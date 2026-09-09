import os
import time
import re
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_PATH     = BASE_DIR / "data" / "preprocessed_tweets.csv"
GLOVE_PATH    = BASE_DIR / "data" / "glove.6B.100d.txt"
OUTPUT_DIR    = BASE_DIR / "outputs" / "embeddings_models"
EMBEDDING_DIM = 100
MAX_LEN       = 50
HIDDEN_DIM    = 64
DROPOUT_PROB  = 0.3
BATCH_SIZE    = 64
LR            = 1e-3
EPOCHS        = 10
PATIENCE      = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Cargar y dividir datos ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df['tokens'] = df['tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
texts = df['tokens']
labels = df['label'].values
texts_train, texts_val, y_train, y_val = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 2. Calcular pesos de clase ---------------------------------------------------------
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)

# 3. Construir vocabulario y embeddings ---------------------------------------------------------------------------
word2idx = {'<pad>': 0, '<unk>': 1}
emb_list = [
    np.zeros(EMBEDDING_DIM),
    np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
]
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec  = np.array(parts[1:], dtype=np.float32)
        word2idx[word] = len(word2idx)
        emb_list.append(vec)
embedding_matrix = torch.tensor(np.stack(emb_list), dtype=torch.float32)

#  4. Dataset y DataLoader ------------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=MAX_LEN):
        self.texts    = list(texts)
        self.labels   = labels
        self.word2idx = word2idx
        self.max_len  = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        idxs   = [self.word2idx.get(tok, self.word2idx['<unk>']) for tok in tokens]
        if len(idxs) < self.max_len:
            idxs += [self.word2idx['<pad>']] * (self.max_len - len(idxs))
        else:
            idxs = idxs[:self.max_len]
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_ds = TweetDataset(texts_train, y_train, word2idx)
val_ds   = TweetDataset(texts_val,   y_val,   word2idx)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 5. Definición de modelos -----------------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        num_embeddings, emb_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.fc1  = nn.Linear(emb_dim * MAX_LEN, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.fc2  = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x):
        emb  = self.embedding(x)                
        flat = emb.view(x.size(0), -1)           
        hid  = self.relu(self.fc1(flat))         
        hid  = self.dropout(hid)
        return self.fc2(hid)                     

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        num_embeddings, emb_dim = embedding_matrix.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT_PROB
        )
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 2)

    def forward(self, x):
        emb = self.embedding(x)                          
        _, (h_n, _) = self.lstm(emb)                     
        hid = torch.cat((h_n[0], h_n[1]), dim=1)          
        hid = self.dropout(hid)
        return self.fc(hid)                              

# 6. Función genérica de entrenamiento + validación ------------------------------------------------------------------
def train_and_evaluate(model_cls, name):
    model     = model_cls(embedding_matrix).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_f1   = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds  = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy())

        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        print(f"{name} Epoch {epoch} | Val F1 {f1:.4f}")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"{name}: no improvement in {PATIENCE} epochs, stopping.")
                break

    # Restaurar mejor estado y calcular métricas finales
    model.load_state_dict(best_state)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    acc   = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc   = roc_auc_score(all_labels, all_probs)
    cm    = confusion_matrix(all_labels, all_preds)

    # Calcular latencia para 100 ejemplos de validación
    start = time.time()
    for i in range(100):
        sample = torch.tensor(val_ds[i][0]).unsqueeze(0).to(DEVICE)
        _ = model(sample)
    latency = (time.time() - start) / 100 * 1000

    # Guardar modelo entrenado
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name.lower()}_model.pt"))
    metrics = {
        'accuracy':        acc,
        'precision':       prec,
        'recall':          rec,
        'f1_score':        f1,
        'auc_roc':         auc,
        'latency_ms':      latency,
        'confusion_matrix': cm.tolist(),
        'best_val_f1':     best_f1
    }
    return metrics, model  

# 7. Entrenar MLP y BiLSTM ----------------------------------------------------------------------
results = {}
mlp_metrics, mlp_model   = train_and_evaluate(MLPClassifier, 'MLP')
bilstm_metrics, bilstm_model = train_and_evaluate(BiLSTMClassifier, 'BiLSTM')
results['MLP'] = mlp_metrics
results['BiLSTM'] = bilstm_metrics

# 8. Guardar vocabulario y embeddings para inferencia posterior -----------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, 'word2idx.pkl'), 'wb') as f:
    pickle.dump(word2idx, f)
torch.save(embedding_matrix, os.path.join(OUTPUT_DIR, 'embedding_matrix.pt'))

# 9. Volcar métricas en JSON ----------------------------------------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, 'metrics_glove_neural.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f"Embeddings-based models and metrics saved in {OUTPUT_DIR}")

# 10. Clasificación de un ejemplo en el mismo script  -------------------------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\d+', '', text)
    return text.split()

# Ejemplo de frase a clasificar
example_tweet = "I just love getting stuck in traffic—best way to spend my evening."

# 10.1) Preprocesar y tokenizar
tokens = tokenize(example_tweet)
idxs = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens][:MAX_LEN]
if len(idxs) < MAX_LEN:
    idxs += [word2idx['<pad>']] * (MAX_LEN - len(idxs))
X_ids = torch.tensor([idxs], dtype=torch.long).to(DEVICE)

# 10.2) Inferencia con MLP
mlp_model.eval()
with torch.no_grad():
    logits_mlp = mlp_model(X_ids)
    probs_mlp  = torch.softmax(logits_mlp, dim=1)[0]
    prob_sar_mlp = probs_mlp[1].item()
    pred_mlp     = torch.argmax(logits_mlp, dim=1).item()

# 10.3) Inferencia con BiLSTM
bilstm_model.eval()
with torch.no_grad():
    logits_bi = bilstm_model(X_ids)
    probs_bi  = torch.softmax(logits_bi, dim=1)[0]
    prob_sar_bi = probs_bi[1].item()
    pred_bi     = torch.argmax(logits_bi, dim=1).item()

# 10.4) Mostrar resultados
print("\n=== Ejemplo de inferencia en el mismo script ===")
print(f"Tweet: {example_tweet}")
print(f"MLP + GloVe    → Prob(sarcasm) = {prob_sar_mlp:.6f}, Prediction = {'SARCASTIC' if pred_mlp == 1 else 'NOT SARCASTIC'}")
print(f"BiLSTM + GloVe → Prob(sarcasm) = {prob_sar_bi:.6f}, Prediction = {'SARCASTIC' if pred_bi == 1 else 'NOT SARCASTIC'}")