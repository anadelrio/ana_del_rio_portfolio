# Detección de sarcasmo en redes sociales (NLP)

Proyecto académico — asignatura *Natural Language Processing*, Sapienza Università di Roma
(curso 2024–2025). **Nota: 10.**

## Motivación

El sarcasmo es una de las formas de expresión más difíciles de detectar automáticamente: el
significado literal de un texto suele ser el opuesto de la intención real de quien lo escribe.
Los sistemas de NLP tradicionales, orientados a una interpretación literal del lenguaje, tienden
a fallar en estos casos. El proyecto compara distintas familias de modelos para evaluar hasta
qué punto cada una es capaz de captar esa ambigüedad.

## Datos

Dataset público de Kaggle *"Tweets with sarcasm and irony"*, con textos cortos etiquetados como
sarcásticos o no sarcásticos. Los modelos con embeddings usan además los vectores preentrenados
**GloVe (100d)**. 

## Metodología

- **Preprocesado** (`preprocess.py`): limpieza (URLs, menciones, hashtags, puntuación, números),
  tokenización y binarización de la etiqueta (`irony`/`sarcasm`/`figurative` → 1, resto → 0).
- **Modelos clásicos** (`train_classic.py`): TF-IDF (uni/bigramas, 5000 features, stopwords en
  inglés) + Regresión Logística y Random Forest, con `GridSearchCV`.
- **Modelos con embeddings** (`train_glove_neural.py`): MLP y BiLSTM sobre embeddings GloVe
  congelados, con pesos de clase y early stopping.
- **Transformer** (`train_distilbert.py`): fine-tuning de DistilBERT con AdamW, scheduler
  lineal, mixed precision y early stopping.
- **Evaluación del MLP** (`evaluation.ipynb`): classification report, matriz de confusión,
  barrido de umbral de decisión (el óptimo resultó ser 0.3, no el 0.5 por defecto) y un grid
  search ligero que justifica los hiperparámetros finales (`hidden_dim=64`, `dropout=0.3`,
  `max_len=50`).
- **Inferencia** (`automat_predict_sarcasm.py`): compara las métricas guardadas de los 5 modelos y
  predice automáticamente con el que tenga mejor F1.

## Resultados

| Modelo | F1 | AUC-ROC | Latencia |
|---|---|---|---|
| Regresión Logística | 0.9992 | 0.99999 | 1.36 ms |
| Random Forest | 0.9998 | 0.999996 | 211.62 ms |
| MLP + GloVe | 0.9903 | 0.99674 | 2.50 ms |
| BiLSTM + GloVe | 0.9996 | 0.99990 | 4.38 ms |
| DistilBERT | 0.9939 | 0.99904 | 80.49 ms |

La Regresión Logística ofrece el mejor equilibrio entre precisión y velocidad, ideal para
aplicaciones en tiempo real. El BiLSTM+GloVe logra el F1 más alto manteniendo una latencia baja.
DistilBERT aporta mayor comprensión semántica del contexto, a costa de un coste computacional
mucho mayor por lo que resulta más adecuado en casos donde la latencia no es crítica.

## Notas sobre reproducibilidad

- **`automat_predict_sarcasm.py` espera un `sample.csv`** en la raíz del proyecto con una columna `text`
  — no viene incluido en el repo, hay que crearlo a mano con los tweets que se quieran probar.

## Tecnologías

Python · scikit-learn · PyTorch · HuggingFace Transformers · GloVe · pandas

## Estructura del repositorio

```
sarcasm-detection-nlp/
├── src/
│   ├── preprocess.py           # limpieza y binarización de etiquetas
│   ├── train_classic.py        # TF-IDF + Regresión Logística / Random Forest
│   ├── train_glove_neural.py   # MLP y BiLSTM sobre embeddings GloVe
│   ├── evaluation.ipynb        # evaluación y ajuste de umbral del MLP
│   ├── train_distilbert.py     # fine-tuning de DistilBERT
│   └── automat_predict_sarcasm.py        # selección automática del mejor modelo + inferencia
├── docs/
│   └── paper_sarcasm_detection.pdf   # paper completo del proyecto
├── data/       
│    └── sample.csv                            # mensajes para hacer la evaluación
├── requirements.txt
└── README.md
```

## Cómo ejecutar

```bash
pip install -r requirements.txt

# Descargar los datos (no incluidos en el repo):
#    - Kaggle: "Tweets with sarcasm and irony": https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony -> meterlo en data/tweets-with-sarcasm-and-irony.csv
#    - GloVe 100d: https://nlp.stanford.edu/projects/glove/ -> meterlo en data/glove.6B.100d.txt

cd src
python preprocess.py
python train_classic.py
python train_glove_neural.py
python train_distilbert.py       # requiere GPU para tiempos razonables
python infer_sarcasm.py          # usa el modelo con mejor F1 sobre sample.csv
```
