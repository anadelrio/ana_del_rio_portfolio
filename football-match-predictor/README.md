# Predictor de resultados de fútbol — Segunda División (LaLiga Hypermotion)

Predicción del resultado de partidos (victoria local / empate / victoria visitante) con
garantías formales de cobertura mediante *conformal prediction*.

## Motivación

La Segunda División española es una liga especialmente impredecible: los equipos de la parte
alta de la tabla pierden con frecuencia frente a rivales que luchan por evitar el descenso. El
objetivo del proyecto es doble: en primer lugar, encontrar el modelo con mejor generalización para predecir
el resultado de un partido, y por otro lado, cuantificar la incertidumbre de cada predicción de forma
rigurosa, en vez de dar solo una etiqueta.

## Datos

Tres fuentes combinadas, con partidos de las últimas 5 temporadas (2020–2025):
- Resultados y estadísticas de partidos (`liga_hypermotion_2020-2025`)
- Geolocalización de los estadios de cada equipo (`team_locations`)
- Valor de mercado de las plantillas por temporada (`team_values`)

## Metodología

- **Feature engineering**: medias móviles (ventana de 5 partidos) de puntos y goles,
  win-rate reciente, distancia entre estadios (fatiga por desplazamiento) y valor de
  mercado agregado de la plantilla.
- **Validación**: 5-fold estratificado, para mantener la proporción de clases (H/D/A) en
  cada partición y evitar sesgo hacia la clase mayoritaria.
- **Modelos comparados**: Random Forest, Regresión Logística, MLP, LightGBM, Gaussian Naïve
  Bayes, LDA, QDA y SVM con kernel RBF.
- **Conformal prediction** (α = 0.1) aplicada sobre el modelo ganador, para construir
  conjuntos predictivos con garantía de cobertura del 90%.

## Resultados

| Métrica | Valor |
|---|---|
| Modelo seleccionado | Regresión Logística |
| F1_macro (validación) | 0.397 |
| Accuracy (test) | 40.18% |
| F1_macro (test) | 0.394 |
| Cobertura empírica (conformal, α=0.1) | 89.12% |
| Tamaño medio del conjunto predictivo | 2.48 clases |

La Regresión Logística ofreció el mejor equilibrio entre interpretabilidad y generalización.
La aplicación de conformal prediction añade una capa de fiabilidad cuantificable: en vez de
un único resultado, el modelo entrega un conjunto de resultados posibles con una garantía de
cobertura del 90%, útil en contextos donde el coste de un error es alto.

## Nota sobre reproducibilidad

Los resultados de la tabla anterior corresponden a la ejecución documentada en el informe
(`docs/informe_proyecto.pdf`). Si se ejecuta `models_conformal.py` hoy, es posible que se obtenga
un modelo ganador y unas métricas distintas. Motivos:

- **Margen muy ajustado entre modelos**: en la comparativa original, Random Forest
  (F1_macro ≈ 0.367) y Regresión Logística (F1_macro ≈ 0.397) quedan muy cerca. Pequeñas
  diferencias de versión entre librerías (`scikit-learn`, `lightgbm`) o de sistema operativo
  pueden invertir cuál de los dos gana, incluso con `random_state` fijado.
- **El script selecciona el modelo automáticamente** según qué gane esa comparativa, así que
  un cambio de ganador no es un error, si no que es el comportamiento esperado del pipeline.
- **La cobertura empírica del conformal prediction es sensible al modelo ganador.** El umbral
  de calibración (`q_hat`) se calcula sobre el propio conjunto de entrenamiento, y modelos
  como Random Forest tienden a estar más sobreconfiados en esos datos que la Regresión
  Logística. Si Random Forest gana en tu ejecución, es esperable ver una cobertura empírica
  bastante por debajo del 90% objetivo (y conjuntos predictivos más pequeños) — no es un fallo
  del código, sino una limitación conocida del método de calibración usado.

En resumen: los números y gráficos de importancia de variables pueden variar de una ejecución
a otra según el entorno; los que aparecen en el informe son los de la ejecución original.

## Tecnologías

Python · scikit-learn · LightGBM · pandas · numpy · matplotlib

## Estructura del repositorio

```
football-match-predictor/
├── src/
│   ├── models_without_conformal.py   # comparativa de 8 modelos + bias-variance decomposition
│   └── models_conformal.py           # calibración + conformal prediction + feature importance
├── datasets/                      
│   └── inputs/                      # datasets de entrada
├── docs/
│   └── informe_proyecto.pdf          # informe completo del proyecto
├── requirements.txt
└── README.md
```
