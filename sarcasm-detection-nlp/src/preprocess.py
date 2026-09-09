import re
import os
from pathlib import Path
import pandas as pd

# Rutas relativas a la raíz del proyecto 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
data_in  = DATA_DIR / "tweets-with-sarcasm-and-irony.csv"
data_out = DATA_DIR / "preprocessed_tweets.csv"

# Función de limpieza
def clean_text(text):
    text = text.lower()
    # Quitar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Quitar menciones @usuario
    text = re.sub(r'@\w+', '', text)
    # Quitar el símbolo # pero conservar la palabra
    text = re.sub(r'#', '', text)
    # Quitar puntuación (salvo apostróficos si se desea)
    text = re.sub(r"[^\w\s']", '', text)
    # Quitar números
    text = re.sub(r'\d+', '', text)
    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Cargar dataset original
print(f"Cargando datos desde {data_in}...")
df = pd.read_csv(data_in)

# Mapear la columna 'class' a etiqueta binaria 'label'
def map_label(cls):
    if cls in ['irony', 'sarcasm', 'figurative']:
        return 1
    else:
        return 0

print("Mapeando clases a labels binarios...")
df['label'] = df['class'].apply(map_label)

# Limpiar y tokenizar
print("Limpiando texto y tokenizando...")

def tokenize(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return tokens

# Aplicar limpieza y tokenización
df['tokens'] = df['tweets'].fillna('').astype(str).apply(tokenize)

# Guardar tokens y label
df_out = df[['tokens', 'label']]
print(f"Guardando preprocesado en {data_out}...")
DATA_DIR.mkdir(parents=True, exist_ok=True)
df_out.to_csv(data_out, index=False)
print("Preprocesamiento completo.")
