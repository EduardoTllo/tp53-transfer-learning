# Plan general del experimento de transfer learning

Este documento explica cómo se organiza el código del experimento. Hay un plan por notebook (04 a 09). Cada plan dice qué entra, qué sale, qué texto lleva el notebook, qué gráficas se muestran y cómo se escribe el código.

Las decisiones vienen del vault: `30 Proyecto/Decisiones.md` (12-oct) y `30 Proyecto/DL - Resumen de la tarea.md`, sección "Diseño del experimento".

## La pregunta
¿Un modelo de TP53 entrenado con cohortes grandes (TCGA-BRCA + METABRIC) se puede llevar a una cohorte nueva (CPTAC, SMC 2018)? ¿Cuántos pacientes locales necesita para quedar bueno y estable?

## Orden de los notebooks
| # | Notebook | Qué hace | Necesita |
|---|---|---|---|
| 04 | `04_examenes_y_repeticiones.ipynb` | Elige los 5 exámenes y las 100 repeticiones | Notebook 02 |
| 05 | `05_elasticnet.ipynb` | Estrategias 4, 5 y 6 | 04 |
| 06 | `06_red_parte_A.ipynb` | Elige la arquitectura de la red con la fuente y la entrena | 02 |
| 07 | `07_ensayo_general.ipynb` | TCGA → METABRIC para elegir cómo hacer el ajuste fino | 06 |
| 08 | `08_red_experimento.ipynb` | Estrategias 1, 2 y 3 en CPTAC y SMC | 04, 06, 07 |
| 09 | `09_resultados.ipynb` | Gráficas, pruebas y número de pacientes recomendado | 05, 08 |

CPTAC y SMC **no se abren** en los notebooks 06 y 07.

Después, si los resultados lo piden: `10_sensibilidad.ipynb` con la normalización A (media y DE solo de los n locales) y QN-Z, definidas de antemano el 12-oct.

## Palabras
- **Fuente:** TCGA-BRCA + METABRIC (2,860 pacientes).
- **Cohorte objetivo:** CPTAC (105) o SMC 2018 (167).
- **Examen:** 40 pacientes de la cohorte objetivo que nunca se usan para entrenar. 5 exámenes por cohorte.
- **Pacientes locales:** n pacientes de la cohorte objetivo, fuera del examen, con los que se ajusta el modelo. n = 10, 15, 20, 40 (y 80 solo en SMC).
- **Repetición:** una elección al azar de n pacientes locales. 100 por examen y por n.
- Se dice "estrategia" (no "brazo") y "repetición" (no "sorteo").

## Las seis estrategias
| # | Estrategia | Con qué aprende | Notebook |
|---|---|---|---|
| 1 | Red: aplicar directo | Solo la fuente | 08 |
| 2 | Red preentrenada + ajuste fino | Red 1, ajustada con n locales | 08 |
| 3 | Red desde cero | Solo n locales | 08 |
| 4 | ElasticNet: aplicar directo | Solo la fuente | 05 |
| 5 | ElasticNet local | Solo n locales | 05 |
| 6 | ElasticNet fuente + locales | Fuente + n locales (locales = 50% del peso) | 05 |

## Rutas
| Qué | Ruta (desde la raíz del repo) | ¿En git? |
|---|---|---|
| Expresión (log2, sin normalizar) | `deep_learning/datos/processed/X_{tcga_brca, metabric, cptac, smc_2018}.parquet` | No |
| Etiquetas | `deep_learning/datos/processed/y_{...}.csv` (columnas `patient_id`, `y`, `pam50`) | No |
| Resultados grandes (repeticiones, AUC, modelos) | `deep_learning/datos/processed/experimento/` | No |
| Funciones comunes | `deep_learning/scripts/comun.py` | Sí |
| Figuras | `deep_learning/resultados/figuras/` | Sí |
| Tablas chicas para el paper | `deep_learning/resultados/tablas/` | Sí |

Filas de `X_*.parquet` = pacientes (mismo orden que el índice de `y_*.csv`). Columnas = 18,040 genes (Entrez ID como texto). METABRIC tiene 14 valores vacíos.

## Funciones comunes (`deep_learning/scripts/comun.py`)
Se escriben una sola vez y todos los notebooks las importan. Así la normalización es idéntica en todas partes.

| Función | Qué hace | Estado |
|---|---|---|
| `cargar(cohorte)` | Lee `X` e `y` y los pone en el mismo orden. | Hecha (13-oct) |
| `zscore(X, ref)` | Llena vacíos con la mediana de `ref` y normaliza cada gen con la media y la DE de `ref`. Si la DE es 0, la deja en 1. Fuente: `ref` = toda la cohorte. Objetivo (opción B): `ref` = pacientes fuera del examen. | Hecha (13-oct) |
| `fuente()` | TCGA y METABRIC, cada una normalizada con todos sus pacientes, y juntas. Devuelve X, y y la cohorte de cada fila. | Hecha (13-oct) |
| `armar_red(capas, dropout, tasa, optimizador)` | La red densa de los notebooks 06–08 (Keras). Así las estrategias 1, 2 y 3 usan la misma red. | Hecha (13-oct) |
| `pesos_clase(y)` | Peso de cada clase en la pérdida de la red. | Hecha (13-oct) |
| `pitman_morgan(a, b)` | Compara la DE de dos listas pareadas de AUC. Devuelve el valor p (dos colas) y un valor z con signo: + si `a` tiene menor DE. | Se agrega en el 09 |
| `stouffer(zs)` | Suma los z de los 5 exámenes y divide por √5. Devuelve Z y su valor p de dos colas. | Se agrega en el 09 |

`pitman_morgan`: `r = corr(a + b, a − b)`, `t = r·√(m − 2) / √(1 − r²)`, con m = número de repeticiones. p de la t de Student con m − 2 grados de libertad. z = signo · Φ⁻¹(1 − p/2).

## Motor de las redes
Keras 3.15 con **PyTorch 2.14** por debajo (`KERAS_BACKEND=torch`, lo fija `comun.py`). TensorFlow 2.20 y 2.21 se caen al importarse en macOS 26.4 (13-oct). El código de Keras no cambia. Versiones fijadas en `requirements.txt`.

## Semillas
- Exámenes: semilla `2026 + e`, con e = 0 a 4. Los exámenes 1 a 3 son los mismos de la prueba rápida (notebook 03). No se cambian.
- Repeticiones: semilla `100000·e + 1000·n + r`, con r = 0 a 99.
- Redes: semilla `r` (la repetición). La misma para las estrategias 2 y 3.

## Reglas que no se rompen
1. Nada se elige mirando el examen.
2. CPTAC y SMC se analizan por separado.
3. Todas las estrategias usan los mismos exámenes y las mismas repeticiones.
4. Cada notebook empieza con un texto que dice para qué sirve, qué entra y qué sale, y termina con un "Cómo se lee" bajo cada gráfica.
5. Textos en español, frases cortas. Cada término se explica la primera vez que aparece.
