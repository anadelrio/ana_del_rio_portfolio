import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

BASE_DIR   = Path(__file__).resolve().parent.parent
data_path  = BASE_DIR / "data" / "preprocessed_tweets.csv"  # Preprocessed tokens+label
output_dir = BASE_DIR / "outputs" / "classic"
os.makedirs(output_dir, exist_ok=True)

# 1. Cargar y preparar datos
df = pd.read_csv(data_path)
# Reconstruye el texto a partir de la columna tokens
df['text'] = df['tokens'].apply(lambda toks: ' '.join(eval(toks) if isinstance(toks, str) else toks))
X = df['text'].values
y = df['label'].values

# Split estratificado
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 2. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

# 3. Modelos y Grid Search
results = {}

# Logistic Regression
lr = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)
lr_params = {'C': [0.01, 0.1, 1, 10]}
lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train_tfidf, y_train)
best_lr = lr_grid.best_estimator_
results['LogisticRegression'] = {
    'best_params': lr_grid.best_params_
}

# Random Forest
rf = RandomForestClassifier(
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train_tfidf, y_train)
best_rf = rf_grid.best_estimator_
results['RandomForest'] = {
    'best_params': rf_grid.best_params_
}

## 4. Evaluación y latencia
def evaluate(model, X_val_feats, y_val, name):
    # Predicción y métricas
    preds = model.predict(X_val_feats)
    probs = model.predict_proba(X_val_feats)[:,1] if hasattr(model, 'predict_proba') else None
    acc   = accuracy_score(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='binary')
    auc   = roc_auc_score(y_val, probs) if probs is not None else None
    cm    = confusion_matrix(y_val, preds)

    # Latencia CPU (100 muestras) usando directamente X_val_feats
    start = time.time()
    for i in range(100):
        sample = X_val_feats[i]
        _ = model.predict(sample)   # sample es un vector TF-IDF precomputado
    latency = (time.time() - start) / 100 * 1000  # ms

    # Guardar resultados
    results[name].update({
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc,
        'latency_ms': latency,
        'confusion_matrix': cm.tolist()
    })
    print(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}, Latency: {latency:.1f} ms")

evaluate(best_lr, X_val_tfidf, y_val, 'LogisticRegression')
evaluate(best_rf, X_val_tfidf, y_val, 'RandomForest')

# 5. Guardado de artefactos
joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
joblib.dump(best_lr,    os.path.join(output_dir, 'logreg_model.joblib'))
joblib.dump(best_rf,    os.path.join(output_dir, 'rf_model.joblib'))

# Guardar métricas en JSON
metrics_file = os.path.join(output_dir, 'metrics_classic.json')
with open(metrics_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Classic baselines and metrics saved in {output_dir}")
