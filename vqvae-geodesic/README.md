# VQ-VAE con Cuantización Geodésica — MNIST

Proyecto académico — asignatura *Deep Learning and Applied AI*, Sapienza Università di Roma
(2025).

## Motivación

Un VQ-VAE estándar aprende su codebook discreto conjuntamente con el encoder, cuantizando por
distancia euclídea en el espacio latente. Esa distancia euclídea puede no respetar la geometría
real del manifold sobre el que viven los datos. Este proyecto explora una alternativa: entrenar
primero un VAE continuo, y cuantizar su espacio latente *a posteriori* usando **distancias
geodésicas** (calculadas sobre un grafo k-NN), bajo la hipótesis de que esto preserva mejor la
estructura intrínseca del manifold que una cuantización euclídea directa.

## Datos

MNIST (60.000 imágenes de entrenamiento / 10.000 de test, 28×28 en escala de grises). Se
descarga automáticamente la primera vez que se ejecuta cualquier script.

## Metodología

- **VAE convolucional** (`src/models/vae.py`): encoder/decoder convolucional con un latente
  vectorial (dim=20), entrenado con ELBO (BCE + KL).
- **VQ-VAE convolucional** (`src/models/vqvae.py`): el encoder produce una **rejilla espacial**
  de vectores (7×7 posiciones), cada una cuantizada de forma independiente contra un codebook
  compartido de K=64 códigos, actualizado por media móvil exponencial (EMA) — el diseño estándar
  de van den Oord et al. (2017).
- **Cuantización geodésica** (`src/geodesic/graph_tools.py`): sobre los latentes del VAE, se
  construye un grafo k-NN (k=10) y se calculan distancias geodésicas mediante *Landmark Isomap*
  (Dijkstra desde ~500 puntos de referencia, para que sea viable con 60.000 puntos sin tener que
  calcular una matriz de distancias N×N completa), seguido de K-means sobre el embedding
  resultante.
- **Control euclídeo**: el mismo K-means, mismo K, mismos latentes, pero con distancia euclídea
  directa — permite cuantificar si la distancia geodésica aporta algo real frente al enfoque
  estándar, en vez de darlo por hecho.
- **Prior autoregresivo** (`src/models/autoregressive.py`): una GRU que predice, en orden
  raster, cada código de la rejilla del VQ-VAE a partir de los anteriores dentro de la misma
  imagen — permite generar dígitos nuevos desde cero.

## Resultados

**Reconstrucción** (conjunto de test, métricas por separado — no sumadas en un único "loss
total", para que sean comparables entre modelos de naturaleza distinta):

| Modelo | BCE/img | SSIM | PSNR | Otro término |
|---|---|---|---|---|
| VAE | 75.6 | 0.872 | 19.8 dB | KL = 25.7 |
| VQ-VAE | 95.8 | 0.826 | 17.6 dB | Perplejidad codebook = 26.2/64 |
| Geo-VQ | 187.2 | 0.603 | 13.8 dB | — |
| Eucl-VQ (control) | 186.4 | 0.603 | 13.8 dB | — |

*Geo-VQ y VQ-VAE no son directamente comparables en calidad de reconstrucción: Geo-VQ cuantiza
un único vector global (64 reconstrucciones posibles en total), frente a las 64⁴⁹ combinaciones
de la rejilla espacial del VQ-VAE — es una limitación inherente al diseño "a posteriori", no del
método geodésico en sí.*

**Ablation geodésico vs. euclídeo** (mismos latentes, mismo K, evaluado contra la etiqueta real
del dígito):

| Método | Pureza | NMI |
|---|---|---|
| Geodésico | **0.953** | **0.651** |
| Euclídeo | 0.926 | 0.611 |

El clustering geodésico produce grupos más puros y más informativos sobre el dígito real que el
euclídeo, con el mismo presupuesto de clusters.

Ver `experiments/figures/comparison.png` (reconstrucciones lado a lado) y
`experiments/figures/ar_generated.png` (dígitos nuevos generados por el modelo autoregresivo).

## Limitaciones conocidas

- El VQ-VAE muestra un ligero patrón de rayado en las reconstrucciones, un artefacto típico de
  las convoluciones transpuestas (`ConvTranspose2d`) — se podría mitigar con upsampling +
  convolución en vez de transposed conv.
- El término de *commitment loss* del VQ-VAE crece de forma sostenida durante el entrenamiento.
  No afecta gravemente a la calidad final, pero indica que la escala del encoder se desacopla
  gradualmente del codebook; normalizar la salida del encoder o ajustar β debería estabilizarlo.

## Tecnologías

Python · PyTorch · scikit-learn · scipy (Dijkstra) · scikit-image (SSIM/PSNR) · matplotlib

## Estructura del repositorio

```
vqvae-geodesic/
├── src/
│   ├── data/dataset.py              # MNIST desde los archivos idx-ubyte (sin torchvision)
│   ├── models/
│   │   ├── vae.py                   # VAE convolucional (latente vectorial)
│   │   ├── vqvae.py                 # VQ-VAE convolucional, cuantización espacial + EMA
│   │   └── autoregressive.py        # Prior autoregresivo sobre la rejilla de códigos
│   ├── geodesic/graph_tools.py      # Cuantización geodésica (Landmark Isomap) + control euclídeo
│   ├── training/
│   │   ├── train_vae.py
│   │   ├── train_vqvae.py
│   │   ├── train_geodesic.py
│   │   ├── train_autoregressive.py
│   │   └── generate_autoregressive.py
│   ├── evaluation/
│   │   ├── metrics.py               # SSIM, PSNR
│   │   └── evaluate.py              # Comparación de métricas + figura comparativa
│   └── utils/paths.py
├── experiments/
│   ├── checkpoints/                 # Pesos entrenados
│   ├── latents/                     # Latentes, centroides, ablation geodésico vs. euclídeo
│   ├── logs/                        # Logs de entrenamiento + tabla de resultados
│   └── figures/                     # comparison.png, ar_generated.png
├── docs/                            # Informe y apéndice del proyecto
├── requirements.txt
└── README.md
```

## Cómo ejecutar

Los scripts tienen que ejecutarse en este orden, porque cada uno depende de los checkpoints o
archivos que genera el anterior. Todos aceptan `--resume` para continuar desde el último
checkpoint si se interrumpen. MNIST se descarga solo la primera vez.

| Paso | Script | Depende de | Qué hace | Qué genera | Qué verás al terminar |
|---|---|---|---|---|---|
| 1 | `train_vae.py` | — (descarga MNIST solo) | Entrena el VAE convolucional | `checkpoints/vae_mnist.pt`, `logs/vae_train_log.csv` | En consola, BCE y KL bajando cada época |
| 2 | `train_vqvae.py` | — (descarga MNIST solo) | Entrena el VQ-VAE con cuantización espacial | `checkpoints/vqvae_mnist.pt`, `logs/vqvae_train_log.csv` | La *perplejidad* del codebook subiendo (indica que se están usando más de los 64 códigos, no solo unos pocos) |
| 3 | `train_geodesic.py` | Paso 1 (`vae_mnist.pt`) | Extrae los latentes del VAE sobre las 60.000 imágenes; ejecuta clustering geodésico y, como control, euclídeo | `latents/vae_latents.npy`, `latents/geodesic_centroids.npy`, `latents/euclidean_centroids.npy`, `latents/clustering_ablation.csv` | Una tabla en consola con la pureza y el NMI de cada método frente al dígito real |
| 4 | `train_autoregressive.py` | Paso 2 (`vqvae_mnist.pt`) | Extrae la rejilla de códigos de cada imagen y entrena la GRU autoregresiva sobre ellas | `checkpoints/autoregressive_model.pt`, `logs/autoregressive_train_log.csv` | El NLL por token bajando cada época |
| 5 | `generate_autoregressive.py` | Pasos 2 y 4 | Genera rejillas de códigos nuevas desde cero con la GRU, y las decodifica en imágenes con el VQ-VAE | `figures/ar_generated.png` | Una rejilla de dígitos nuevos, generados sin partir de ninguna imagen real |
| 6 | `evaluate.py` | Pasos 1, 2 y 3 | Evalúa los 4 modelos (VAE, VQ-VAE, Geo-VQ, Eucl-VQ) sobre el conjunto de test: BCE, SSIM, PSNR y perplejidad | `logs/results_table.json`, `figures/comparison.png` | La tabla de métricas en consola, y una figura con 5 filas (original + 4 reconstrucciones) para comparar a simple vista |

```bash
pip install -r requirements.txt
cd src/training

python train_vae.py --epochs 20                    # Paso 1 · ~8 min en CPU
python train_vqvae.py --epochs 20                   # Paso 2 · ~8 min en CPU
python train_geodesic.py                            # Paso 3 · ~1 min
python train_autoregressive.py --epochs 15           # Paso 4 · ~9 min en CPU
python generate_autoregressive.py --n 16             # Paso 5 · unos segundos

cd ../evaluation
python evaluate.py                                   # Paso 6 · ~1 min
```

Al terminar el Paso 6 ya tienes todo lo necesario para revisar el proyecto: la tabla de métricas
(`experiments/logs/results_table.json`), la figura comparativa de reconstrucciones
(`experiments/figures/comparison.png`), la tabla de ablation geodésico vs. euclídeo
(`experiments/latents/clustering_ablation.csv`), y los dígitos generados desde cero
(`experiments/figures/ar_generated.png`).

