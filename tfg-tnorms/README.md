## 2526-TFG---Ana-del-Rio-del-Barrio

### Análisis estadístico y comparativo de t-normas paramétricas en funciones de Agregación 
Repositorio del Trabajo de Fin de Grado presentado en el Grado en Ciencia e Ingeniería de Datos de la Universidad de Oviedo (Escuela Politécnica de Ingeniería de Gijón).
Autora: Ana del Río del Barrio
Tutores: Agustina Bouchet Gutiérrez y Emilio Torres Manzanera
Curso: 2025–2026

#### Descripción 
Este TFg replica de manera independiente el estudio de Luigi Troiano, Luis J. Rodríguez-Muñiz, Pasquale Marinaro e Irene Díaz (2014), Statistical analysis of parametric t-norms, publicado en Information Sciences (vol. 257, pp. 138–162). El objetivo es verificar que los resultados del artículo original son robustos y reproducibles con una implementación propia, desarrollada desde cero en Python.
Este trabajo se organiza en torno a cuatro bloques principales: 
1. Marco teórico:  introducción a la lógica difusa, normas y conormas triangulares, test estadísticos no paramétricos y distribuciones de entrada.
2. Estudio paramétrico de la familia de Yager: análisis del efecto del tamaño de muestra, la aridad y la distribución de los datos de entrada sobre el comportamiento estadístico de la t-norma.
3. Comparación paramétrica por pares: comparación de las cinco familias de t-normas consideradas en el estudio original (Frank, Yager, Hamacher, Schweizer-Sklar y Sugeno-Weber) mediante el test de Wilcoxon-Mann-Whitney, identificando las regiones del espacio paramétrico en las que dos familias producen salidas estadísticamente equivalentes.
4. Efecto de la aridad en la comparación paramétrica: se comprueba que el efecto de la aridad visto en el punto 2 afecta a las regiones de equivalencia obtenidas en el punto 3.

Los resultados obtenidos se presentan en tablas, diagramas de contornos de p-valores medios y mapas de densidad de tasas de no rechazo sobre una malla logarítmica de 100×100 puntos.

#### Estructura del repositorio
```
├── docs/
│   └── TFG_GCINGD_delRío_delBarrio_Ana_2026_07.pdf    # Memoria del TFG
├── src/
│   ├── marco_teorico/
│   │   ├── codigo_imagenes_tnormas.py          # Visualizaciones de t-normas no paramétricas
│   │   └── codigo_imagenes_tconormas.py        # Visualizaciones de t-conormas
│   │
│   ├── simulacion/
│   │   ├── simulacion_caso_motivacional.py     # Caso motivacional introductorio
│   │   ├── benchmark_experiment.py             # Experimento de referencia
│   │   ├── efecto_de_la_aridad.py              # Efecto del número de argumentos
│   │   ├── efecto_tamano_muestra.py            # Efecto del tamaño de muestra
│   │   └── efecto_distribucion_entrada_yager.py # Efecto de la distribución de entrada
│   │
│   ├── comparacion_parametrica/
│   │   ├── yager_frank.py
│   │   ├── yager_hamacher.py
│   │   ├── yager_schweizer-sklar.py
│   │   ├── yager_sugeno-weber.py
│   │   ├── frank_hamacher.py
│   │   ├── frank_schweizer_sklar.py
│   │   ├── frank_sugeno_weber.py
│   │   ├── hamacher_schweizer_sklar.py
│   │   ├── hamacher_sugeno_weber.py
│   │   └── schweizer-sklar_sugeno-weber.py
│   │
│   └── efecto_de_la_aridad/
│       ├── yager_frank.py # Efecto del número de argumentos en la comparativa yager vs frank
│       └── yager_hamacher.py # Efecto del número de argumentos en la comparativa yager vs hamacher
│
└── references/
    └── REFERENCIAS.md                          
```

#### Requisitos
Python 3.9 o superior. Las dependencias necesarias son: numpy, scipy, matplotlib y tqdm.
Pueden instalarse con: pip install numpy scipy matplotlib tqdm. 

#### Uso
Cada script es independiente y puede ejecutarse directamente. 
Los scripts de simulación generan los resultados del estudio paramétrico de Yager. Los scripts de comparación paramétrica generan las figuras de contorno y los mapas de densidad correspondientes. Los scripts del efecto d ela aridad generan figuras de contornos que ilustran cómo varían las regiones a medida que cambia el número de argumentos. Todos los outputs se guardan en el mismo directorio desde el que se ejecuta el script.

#### Parámetros de las simulaciones 
1. Tamaño de malla: 100 × 100
2. Iteraciones por punto: 1000
3. Pares de entrada por iteración: 100
4. Distribución de entrada: en general se usa Uniforme U(0,1), pero en el estudio paramétrico de Yager en el efecto de la distribución de entrada se usan distintas distribuciones Beta.
5. Test estadístico: Wilcoxon-Mann-Whitney
6. Nivel de significación: 0.05

#### Referencia principal
Troiano, L., Rodríguez-Muñiz, L.J., Marinaro, P., Díaz, I. (2014).  
Statistical analysis of parametric t-norms.  
*Information Sciences*, 257, 138–162.  
https://doi.org/10.1016/j.ins.2013.09.024

El listado completo de referencias bibliográficas se encuentra en [`referencias/REFERENCIAS.md`](referencias/REFERENCIAS.md).
