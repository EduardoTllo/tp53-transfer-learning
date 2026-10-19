# Plan · 07 Ensayo general (TCGA → METABRIC)

## Para qué sirve
El ajuste fino (estrategia 2) y la red desde cero (estrategia 3) necesitan una tasa de aprendizaje y un número de épocas. No se pueden elegir con CPTAC o SMC, porque sería elegir con el examen. Con 10 pacientes tampoco se puede apartar validación.

La solución es **simular el experimento completo dentro de la fuente**: METABRIC hace de "cohorte nueva". La combinación con el AUC más alto aquí se congela y se usa tal cual en el notebook 08.

Sirve también para probar el código antes de tocar CPTAC y SMC.

## Entrada
- `X_tcga_brca.parquet`, `y_tcga_brca.csv`, `X_metabric.parquet`, `y_metabric.csv`
- `deep_learning/resultados/tablas/06_elegida.json` (arquitectura, lote, dropout y optimizador)
- `deep_learning/scripts/comun.py`

CPTAC y SMC **no se cargan**.

## Salida
| Archivo | Qué es |
|---|---|
| `deep_learning/datos/processed/experimento/ensayo_auc.csv` | `examen`, `estrategia`, `tasa`, `epocas`, `n`, `repeticion`, `auc` (examen), `auc_locales` (con sus propios pacientes locales; brecha = sobreajuste). Incluye estrategias 1 y 4 (`n` = 0) |
| `deep_learning/resultados/tablas/07_ajuste_elegido.json` | Tasa y épocas elegidas para la estrategia 2 y para la 3 |
| `deep_learning/resultados/tablas/07_resumen.csv` | AUC promedio y DE por estrategia, combinación y n |
| `deep_learning/datos/processed/experimento/ensayo_curvas.parquet` | `examen`, `estrategia`, `tasa`, `n`, `repeticion`, `epoca`, `perdida`, `perdida_examen`, `auc_examen` de **todas** las redes |

Se guarda por examen en `experimento/partes_07/`; si se corta, sigue desde el último examen.

## Secciones del notebook

### 0. Título y texto inicial
> **07 · Ensayo general**
> Aquí se hace un ensayo del experimento sin tocar CPTAC ni SMC. Se entrena la red solo con **TCGA** y se usa **METABRIC** como si fuera una cohorte nueva. TCGA es RNA-seq y METABRIC es microarreglo, así que el salto se parece al de la fuente a CPTAC y SMC.
> Se prueban algunas tasas de aprendizaje y números de épocas para el ajuste fino y para la red desde cero. Las dos estrategias prueban **la misma cantidad de combinaciones**, para que ninguna tenga ventaja (Hanczar 2022).

### 1. Red preentrenada solo con TCGA
- Normalizar TCGA con su propia media y DE (961 pacientes).
- `comun.armar_red` con la arquitectura, lote, dropout y optimizador de `06_elegida.json`.
- Épocas: parada temprana con 20% de TCGA apartado como validación (estratificado, semilla 2026).
- Guardar en memoria (no hace falta archivo).

### 2. Exámenes y repeticiones en METABRIC
- 5 exámenes de 40, estratificados por TP53, semillas 2026–2030.
- n = 10, 15, 20, 40. **20 repeticiones** por n y examen (no 100: aquí solo se elige, no se reporta). Semillas como en el notebook 04.
- Normalización B: media y DE de los METABRIC fuera de cada examen.

### 3. Referencia: aplicar directo
- AUC de la red de TCGA en cada examen de METABRIC, sin ajustar.
- También ElasticNet entrenado solo con TCGA (mezcla y fuerza del notebook 05).
- Texto: "STAMP llevó su modelo de TCGA a METABRIC y obtuvo 0.902 con su red de grafos y 0.841 con ElasticNet. Si nuestros números quedan muy por debajo, hay que revisar algo antes de seguir."

### 4. Combinaciones que se prueban
| Estrategia | Tasa | Épocas | Combinaciones |
|---|---|---|---|
| 2. Ajuste fino (todas las capas libres) | 1e-5, 1e-4, 1e-3 | 10, 30, 100 | 9 |
| 3. Desde cero (misma red, pesos nuevos) | 1e-4, 1e-3, 1e-2 | 10, 30, 100 | 9 |

- Lote = todos los n locales juntos (con 10 pacientes no tiene sentido partir).
- El ajuste fino usa tasas más chicas porque parte de una red que ya sabe; pasos grandes borrarían lo aprendido.
- Semilla de la red = número de repetición, igual en 2 y 3.
- Una misma combinación para todos los n (decisión del 12-oct).
- **Un solo entrenamiento por tasa (13-oct):** se entrena hasta 100 épocas y se mide en las épocas 10, 30 y 100. Es idéntico a entrenar 10, 30 y 100 por separado (misma semilla, lote completo, tasa fija) y cuesta un tercio. En esas épocas se mide el AUC exacto en el examen y con los propios pacientes locales (**brecha**, decisión del autor del 13-oct).
- La red de TCGA se entrena con todo TCGA durante las épocas que dio la parada temprana con 20% apartado, igual que la red final del 06.

### 5. Elegir
- Para cada estrategia: la combinación con mayor AUC promedio (todos los n y los 5 exámenes juntos).
- Si dos difieren en menos de 0.005, la de menor DE promedio.
- Guardar `07_ajuste_elegido.json`.

**Figura 07-1 · AUC según tasa y épocas** (`07_rejilla.png`)
Dos mapas de calor 3 × 3 (estrategia 2 y 3), con el AUC promedio. La elegida marcada.
Cómo se lee: "Una zona pareja significa que la elección no es delicada. Un solo casillero alto, rodeado de bajos, sería sospechoso."

**Figuras 07-2 y 07-3, juntas · AUC, estabilidad y brecha según n** (`07_auc_de_brecha_vs_n.png`)
Tres paneles. Eje x = n. (a) AUC en el examen de METABRIC, líneas 2 y 3 (± DE) y líneas horizontales de aplicar directo (red y ElasticNet). (b) DE del AUC entre repeticiones. (c) Brecha = AUC con los locales − AUC en el examen.
Cómo se lee: "Es un anticipo de la Figura 09-1. Si la red preentrenada no da un AUC mayor que aplicar directo ni con 40 pacientes, el ajuste fino podría no ayudar tampoco en CPTAC y SMC."

Cómo se lee (b): "Anticipo de la pregunta principal: ¿la red preentrenada varía menos que la red desde cero?" (c): "Una brecha grande = la red memorizó a sus pocos pacientes. Se espera que baje con n y que sea menor en la preentrenada."

### 6. Curvas de pérdida y AUC por época
Sirven para ver si las épocas elegidas en el paso 5 caen en una zona razonable o en un borde de la rejilla. Se puede mirar el examen porque aquí es METABRIC (parte de la fuente): no toca CPTAC ni SMC.

- Con la tasa elegida para cada estrategia, se entrena hasta **100 épocas** (el máximo de la rejilla) y se anota en cada época la pérdida de entrenamiento, y la pérdida y el AUC en el examen (`validation_data` = el examen).
- Salen de los mismos entrenamientos del paso 4 (no hace falta entrenar aparte): todas las repeticiones y exámenes, n = 10 y 40.

**Figura 07-4 · Pérdida y AUC por época** (`07_curvas_epocas.png`)
2 × 2 paneles: filas = pérdida (entrenamiento con línea continua, examen con línea punteada) y AUC en el examen; columnas = n = 10 y n = 40. Una línea por estrategia (2 y 3): promedio de las 10 repeticiones y banda = ± DE. Línea vertical en las épocas elegidas en el paso 5 para cada estrategia.
Cómo se lee: "Mientras la pérdida del examen baja, la red aprende algo que sirve fuera de sus pacientes locales. Cuando la de entrenamiento sigue bajando y la del examen sube, la red empieza a memorizar a sus pocos pacientes. Se espera que la red desde cero memorice antes que la preentrenada, y que las épocas elegidas queden antes de ese punto."

## Lo que este notebook no hace
- No reporta resultados para el paper: METABRIC es parte de la fuente.
- No cambia nada de los notebooks 04–06.

## Tiempo
2 estrategias × 3 tasas × 4 n × 5 exámenes × 20 repeticiones = 2,400 entrenamientos de 100 épocas (las épocas 10 y 30 salen del mismo entrenamiento). Del orden de horas. Prueba mínima (13-oct, red de prueba, 1 examen, 2 repeticiones, épocas 2/3/5): todo corre en ~2 min.
