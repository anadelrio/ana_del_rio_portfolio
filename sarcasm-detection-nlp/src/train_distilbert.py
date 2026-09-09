import os
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from multiprocessing import freeze_support

class CachedTweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    # Configuración 
    BASE_DIR       = Path(__file__).resolve().parent.parent
    DATA_PATH      = BASE_DIR / "data" / "preprocessed_tweets.csv"
    MODEL_NAME     = "distilbert-base-uncased"
    OUTPUT_DIR     = BASE_DIR / "outputs" / "distilbert_sarcasm"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TUNING=True entrena en minutos con un 10% de los datos, 1 época y secuencias cortas
    TUNING         = True
    if TUNING:
        SUBSAMPLE_RATIO = 0.1
        NUM_EPOCHS      = 1
        MAX_LENGTH      = 32
        BATCH_SIZE      = 32
        NUM_WORKERS     = 0
    else:
        SUBSAMPLE_RATIO = None
        NUM_EPOCHS      = 3
        MAX_LENGTH      = 134
        BATCH_SIZE      = 32
        NUM_WORKERS     = 4

    LEARNING_RATE = 5e-5
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE      = 1

    # Carga y Split
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['tokens'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
    labels = df['label'].values
    texts  = df['text'].values

    txt_train, txt_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    if TUNING:
        idxs = np.random.choice(len(txt_train),
                                size=int(SUBSAMPLE_RATIO * len(txt_train)),
                                replace=False)
        txt_train = txt_train[idxs]
        y_train   = y_train[idxs]

    # pesos de clase
    class_weights_np = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)

    # Tokenización 
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(list(txt_train), truncation=True, padding=True, max_length=MAX_LENGTH)
    val_enc   = tokenizer(list(txt_val),   truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = CachedTweetDataset(train_enc, y_train)
    val_dataset   = CachedTweetDataset(val_enc,   y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Modelo y Entrenamiento 
    model   = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = GradScaler() if DEVICE.type=='cuda' else None

    best_f1       = 0.0
    epochs_no_imp = 0

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            labels_batch = batch.pop('labels').to(DEVICE)
            inputs = {k: v.to(DEVICE) for k,v in batch.items()}
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    outputs = model(**inputs)
                    loss = CrossEntropyLoss(weight=class_weights)(outputs.logits, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**inputs)
                loss = CrossEntropyLoss(weight=class_weights)(outputs.logits, labels_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Validación
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                labels_batch = batch.pop('labels').to(DEVICE)
                inputs = {k: v.to(DEVICE) for k,v in batch.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[:,1].cpu().numpy()
                preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels_batch.cpu().numpy())
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Val Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            epochs_no_imp = 0
            model.save_pretrained(OUTPUT_DIR)
        else:
            epochs_no_imp += 1
            if epochs_no_imp > PATIENCE:
                print(f"Early stopping after {epochs_no_imp} epochs without improvement.")
                break

    # Evaluación final 
    model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
    model.eval()

    final_preds = []
    final_probs = []
    with torch.no_grad():
        for batch in val_loader:
            labels_batch = batch.pop('labels').to(DEVICE)
            inputs = {k: v.to(DEVICE) for k,v in batch.items()}
            outputs = model(**inputs)
            final_probs.extend(torch.softmax(outputs.logits, dim=1)[:,1].cpu().numpy())
            final_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

    acc   = accuracy_score(y_val, final_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, final_preds, average='binary')
    auc   = roc_auc_score(y_val, final_probs)
    cm    = confusion_matrix(y_val, final_preds)

    # Latencia CPU (100 muestras)
    cpu_model = DistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR).to('cpu')
    cpu_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    start = time.time()
    for i, batch in enumerate(cpu_loader):
        if i>=100: break
        batch.pop('labels')
        with torch.no_grad():
            _ = cpu_model(**{k:v for k,v in batch.items()})
    latency_ms = (time.time() - start)/100*1000

    # Guardar métricas
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc,
        'latency_ms': latency_ms,
        'confusion_matrix': cm.tolist(),
        'best_val_f1': best_f1
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Best Val F1: {best_f1:.4f}. Model and metrics saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    freeze_support()
    main()
