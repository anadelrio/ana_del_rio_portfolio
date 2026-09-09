import os
import json
import re
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
SAMPLE_PATH = BASE_DIR / "data" / "sample.csv"

# helpers
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

#  1. Load all metrics ----------------------------------------------------------------------------------
metrics = {}
# Classic
with open(OUTPUTS_DIR / 'classic' / 'metrics_classic.json') as f:
    m = json.load(f)
    for name, vals in m.items():
        metrics[f'classic/{name}'] = vals['f1_score']
# Embeddings
with open(OUTPUTS_DIR / 'embeddings_models' / 'metrics_glove_neural.json') as f:
    m = json.load(f)
    for name, vals in m.items():
        metrics[f'emb/{name}'] = vals['f1_score']
# DistilBERT
with open(OUTPUTS_DIR / 'distilbert_sarcasm' / 'metrics.json') as f:
    m = json.load(f)
    metrics['distilbert'] = m['f1_score']

#  2. Select best model ----------------------------------------------------------------------------------
best_key = max(metrics, key=lambda k: metrics[k])
print(f"Best model: {best_key} (F1={metrics[best_key]:.4f})")

#  3. Load sample data ------------------------------------------------------------------
try:
    df = pd.read_csv(SAMPLE_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(SAMPLE_PATH, encoding='latin-1')

if 'text' in df.columns:
    texts = df['text'].astype(str).tolist()
else:
    raise ValueError("sample.csv needs 'text' column")

#  4. Predict with best model --------------------------------------------------------------
if best_key.startswith('classic/'):
    # classical: load vectorizer + model
    vec = joblib.load(OUTPUTS_DIR / 'classic' / 'tfidf_vectorizer.joblib')
    name = best_key.split('/')[-1]
    file_map = {
        'LogisticRegression': 'logreg_model.joblib',
        'RandomForest':       'rf_model.joblib'
    }
    mdl_path = OUTPUTS_DIR / 'classic' / file_map[name]
    mdl = joblib.load(mdl_path)
    X = vec.transform(texts)
    probs = mdl.predict_proba(X)[:,1]
    preds = mdl.predict(X)

elif best_key.startswith('emb/'):
    # embeddings: load GloVe vocab + models; define architectures
    # 1) rebuild vocab & embeddings
    EMBED_DIM=100; MAX_LEN=50; HIDDEN_DIM=64; DROPOUT=0.3
    word2idx={'<pad>':0,'<unk>':1}
    emb_list=[np.zeros(EMBED_DIM), np.random.normal(scale=0.6, size=(EMBED_DIM,))]
    with open(DATA_DIR / 'glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split()
            word, vec=parts[0], np.array(parts[1:],dtype=np.float32)
            word2idx[word]=len(word2idx)
            emb_list.append(vec)
    embedding_matrix = torch.tensor(np.stack(emb_list),dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.embedding=nn.Embedding.from_pretrained(emb,freeze=True)
            self.fc1=nn.Linear(emb.size(1)*MAX_LEN,HIDDEN_DIM)
            self.relu=nn.ReLU(); self.drop=nn.Dropout(DROPOUT)
            self.fc2=nn.Linear(HIDDEN_DIM,2)
        def forward(self,x):
            e=self.embedding(x).view(x.size(0),-1)
            h=self.drop(self.relu(self.fc1(e)))
            return self.fc2(h)

    class BiLSTM(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.embedding=nn.Embedding.from_pretrained(emb,freeze=True)
            self.lstm=nn.LSTM(emb.size(1),HIDDEN_DIM,1,batch_first=True,
                              bidirectional=True,dropout=DROPOUT)
            self.drop=nn.Dropout(DROPOUT)
            self.fc=nn.Linear(HIDDEN_DIM*2,2)
        def forward(self,x):
            _,(h,_) = self.lstm(self.embedding(x))
            h = torch.cat((h[0],h[1]),dim=1)
            return self.fc(self.drop(h))

    # load the chosen model
    model_name = best_key.split('/')[-1]
    ModelCls = MLP if model_name=='MLP' else BiLSTM
    model = ModelCls(embedding_matrix).eval()
    model.load_state_dict(torch.load(
        OUTPUTS_DIR / "embeddings_models" / f"{model_name.lower()}_model.pt",
        map_location='cpu'
    ))
    # preprocess texts
    idxs=[]
    for t in texts:
        toks=clean_text(t).split()
        ids=[word2idx.get(w,1) for w in toks][:MAX_LEN]
        ids=ids+ [0]*(MAX_LEN-len(ids))
        idxs.append(ids)
    X = torch.tensor(idxs,dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits,dim=1)[:,1].numpy()
        preds = logits.argmax(dim=1).numpy()

else:
    # distilbert
    tok = DistilBertTokenizerFast.from_pretrained(OUTPUTS_DIR / 'distilbert_sarcasm')
    mdl = DistilBertForSequenceClassification.from_pretrained(OUTPUTS_DIR / 'distilbert_sarcasm').eval()
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        out = mdl(**enc)
        probs = torch.softmax(out.logits,dim=1)[:,1].numpy()
        preds = out.logits.argmax(dim=1).numpy()

# 5. Show predictions -------------------------------------------------------------------------------------
for text,p,prob in zip(texts, preds, probs):
    label = 'SARCASTIC' if p==1 else 'NOT'
    print(f"[{label} ({prob:.3f})] {text}")
