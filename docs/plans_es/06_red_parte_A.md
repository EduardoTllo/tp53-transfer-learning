# Plan · 06 Red, parte A (elegir y entrenar la red de la fuente)

## Para qué sirve
Elige la forma de la red y cómo entrenarla usando **solo la fuente**. Después entrena la red final con los 2,860 pacientes. Esa red es la **estrategia 1** (aplicar directo) y el punto de partida de la **estrategia 2** (ajuste fino).

## Requisito
Keras 3.15 con PyTorch 2.14 (instalados el 13-oct; TensorFlow se cae en macOS 26.4). `comun.py` fija `KERAS_BACKEND=torch`.

## Entrada
- `deep_learning/datos/processed/X_tcga_brca.parquet`, `X_metabric.parquet`, `y_tcga_brca.csv`, `y_metabric.csv`
- `deep_learning/scripts/comun.py`

CPTAC y SMC **no se cargan** en este notebook.

## Salida
| Archivo | Qué es |
|---|---|
| `deep_learning/resultados/tablas/06_busqueda.csv` | AUC de validación y época con el AUC más alto de cada combinación en cada partición. Se agrega al terminar cada combinación: si el notebook se corta, sigue desde la última guardada |
| `deep_learning/resultados/tablas/06_historiales.csv` | Pérdida y AUC por época de cada red (para la Figura 06-3) |
| `deep_learning/resultados/tablas/06_elegida.json` | Arquitectura, tasa, lote, dropout, optimizador y número de épocas elegidos |
| `deep_learning/datos/processed/experimento/red_fuente.keras` | Red final entrenada con los 2,860 (fuente completa) |

## Secciones del notebook

### 0. Título y texto inicial
> **06 · Red neuronal, parte A**
> Una **red densa (MLP)** es una cadena de capas. Cada capa combina los genes (o la capa anterior) y pasa el resultado a la siguiente. La última capa da la probabilidad de TP53 mutado.
> Antes de entrenar hay que decidir algunas cosas que la red no aprende sola (**hiperparámetros**): cuántas capas y neuronas, qué tan grandes son los pasos con que corrige sus errores (**tasa de aprendizaje**), cuántos pacientes mira antes de corregirse (**lote**), cuántas neuronas apaga al azar para no memorizar (**dropout**) y qué regla usa para corregirse (**optimizador**: Adam o SGD).
> Aquí se eligen probando combinaciones con **validación cruzada de 5 particiones dentro de la fuente**. Los valores salen de la literatura: Hanczar 2022, Sabbatini 2026, Yang 2023, STAMP y Triantafyllidis 2025.

### 1. Cargar y normalizar la fuente
- `comun.fuente()`. 2,860 pacientes × 18,040 genes.

### 2. Medir cuánto tarda una red
- Entrenar una sola red (la más grande, `[512, 256, 128]`) en una partición.
- Mostrar segundos por época y épocas hasta parar.
- Texto: el tiempo estimado, para saber cuándo termina. **El dropout (0 y 0.3) se prueba siempre, aunque alargue la búsqueda** (decisión del autor, 13-oct). Ya no hay regla de bajar a 36 combinaciones.

### 3. La red (`armar_red` y `pesos_clase` en `comun.py`, porque el 07 y el 08 usan la misma red)
```
entrada (18,040)
→ Dense(capas[0], ReLU) → Dropout
→ Dense(capas[1], ReLU) → Dropout       (si hay)
→ ...
→ Dense(1, sigmoide)
pérdida: entropía cruzada binaria, con pesos por clase
```
- **Pesos por clase:** equivocarse con un mutado cuesta más, en proporción a cuántos hay menos.
- **Parada temprana:** se deja de entrenar cuando el AUC de validación no mejora en 10 épocas (paciencia de STAMP), con un máximo de 200 épocas. Se guarda la época con el AUC de validación más alto.

### 4. Búsqueda
Rejilla:
| Hiperparámetro | Valores | Por qué |
|---|---|---|
| Capas | `[256]`, `[512, 128]`, `[512, 256, 128]` | 1–3 capas que se achican (Sabbatini); la última es la de Yang para TP53 |
| Tasa | 1e-4, 3e-4, 1e-3 | STAMP usó 1e-4; rango de Triantafyllidis |
| Lote | 16, 32 | Lotes chicos dieron una accuracy más alta (Hanczar); STAMP 10, Triantafyllidis 16 |
| Dropout | 0, 0.3 | Hanczar: no ayudó; los demás lo usan |
| Optimizador | Adam, SGD | SGD fue más estable (Hanczar). SGD con momento 0.9, el valor habitual: sin momento, con tasa 1e-4, casi no avanza en 200 épocas |

- 5 particiones estratificadas por cohorte (TCGA/METABRIC) y TP53. Semilla 2026.
- Para cada combinación: AUC de validación promedio de las 5 y épocas promedio.
- Se elige el AUC promedio más alto. Si dos difieren en menos de 0.005, se elige la red más chica.

**Figura 06-1 · Resultado de la búsqueda** (`06_busqueda.png`)
Un punto por combinación: eje x = AUC de validación promedio, eje y = combinaciones ordenadas. Barras = DE entre las 5 particiones. Color = optimizador. La elegida resaltada.
Cómo se lee: "Si muchas combinaciones están juntas arriba, la elección importa poco, como dijo Hanczar."

**Figura 06-2 · Qué hiperparámetro importa más** (`06_efecto_hiperparametros.png`)
Cinco paneles chicos (uno por hiperparámetro): AUC promedio según cada valor.
Cómo se lee: "El panel con más diferencia entre valores es el hiperparámetro que más pesa."

**Figura 06-3 · Curvas de entrenamiento de la elegida** (`06_curvas_entrenamiento.png`)
Pérdida y AUC por época, entrenamiento y validación, en las 5 particiones.
Cómo se lee: "Cuando la curva de validación deja de mejorar y la de entrenamiento sigue, la red empieza a memorizar. Ahí corta la parada temprana."

### 5. Entrenar la red final
- Con los 2,860 pacientes, la combinación elegida y el número de épocas promedio de la búsqueda (no hay validación para parar).
- Semilla 2026. Guardar `red_fuente.keras` y `06_elegida.json`.
- Texto: "Esta red es la estrategia 1. También es el punto de partida del ajuste fino (estrategia 2)."

## Tiempo
Lo dice el paso 2. Medido el 13-oct (con el 05 corriendo a la vez): 2–4 s por época según la red. Si cada red entrena ~50 épocas, 360 redes ≈ 10–20 horas. Correr de noche. Prueba mínima (3 combinaciones, 2 épocas): ~2 min, todo bien.
