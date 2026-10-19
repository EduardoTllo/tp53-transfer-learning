# Plan · 05 ElasticNet (estrategias 4, 5 y 6)

## Para qué sirve
Corre las tres estrategias sin deep learning con las reglas definitivas. Da la vara: la red tiene que dar un AUC mayor que estas estrategias. Es la versión final de la prueba rápida (notebook 03).

## Entrada
- `deep_learning/datos/processed/X_*.parquet` y `y_*.csv` (las 4 cohortes)
- `deep_learning/datos/processed/experimento/examenes.csv` y `repeticiones.csv` (notebook 04)
- `deep_learning/scripts/comun.py`

## Salida
| Archivo | Columnas |
|---|---|
| `deep_learning/datos/processed/experimento/auc_elasticnet.csv` | `cohorte`, `examen`, `estrategia` (4, 5, 6), `peso_locales` (0.5, 0.25, 0.10; vacío en 4 y 5), `n` (0 en la 4), `repeticion`, `auc` |
| `deep_learning/datos/processed/experimento/prob_elasticnet.parquet` | `cohorte`, `examen`, `estrategia`, `peso_locales`, `n`, `repeticion`, `muestra`, `prob` (probabilidad de mutado de cada paciente del examen) |
| `deep_learning/datos/processed/experimento/elasticnet_fuente.joblib` | Modelo de la estrategia 4 |
| `deep_learning/resultados/tablas/05_hiperparametros_fuente.csv` | AUC de validación de cada combinación |
| `deep_learning/resultados/tablas/05_resumen.csv` | AUC promedio y DE por cohorte, estrategia y n |

Se guarda al terminar cada examen. Si el notebook se corta, sigue desde el último examen guardado.

## Secciones del notebook

### 0. Título y texto inicial
> **05 · ElasticNet**
> ElasticNet es una regresión logística con dos frenos. Uno apaga los genes que no sirven (L1). El otro reparte el peso entre genes parecidos (L2). La **mezcla** dice cuánto pesa cada freno. La **fuerza** (C) dice cuánto frena en total: C chico = frena mucho.
> Se corren tres estrategias:
> - **4. Aplicar directo:** se entrena con la fuente y se usa tal cual.
> - **5. Local:** se entrena solo con los n pacientes locales.
> - **6. Fuente + locales:** se entrena con la fuente y los n locales juntos. Los locales suman el 50% del peso.

### 1. Cargar y normalizar la fuente
- `fuente()` de `comun.py`: TCGA y METABRIC, cada una con su propia media y DE. Los 14 vacíos de METABRIC se llenan con su mediana.
- Mostrar: 2,860 pacientes, 18,040 genes, % de mutados.

### 2. Estrategia 4: elegir mezcla y fuerza con la fuente
- Validación cruzada de 5 particiones **solo en la fuente**, estratificada por cohorte y TP53.
- Rejilla: mezcla `l1_ratio` ∈ {0.1, 0.5, 0.9} × fuerza `C` ∈ {0.01, 0.05, 0.1}. Se elige el AUC de validación más alto.
- Entrenar con los 2,860 y guardar `elasticnet_fuente.joblib`.
- Modelo: `LogisticRegression(penalty="elasticnet", solver="saga", max_iter=1000, tol=1e-3, random_state=0)`. Con 300 iteraciones (las del notebook 03) no convergían la fuente ni el modelo local (13-oct).
- Texto: "CPTAC y SMC no se miran para elegir. Así el resultado no queda inflado."

**Figura 05-1 · Elección de mezcla y fuerza** (`05_rejilla_fuente.png`)
Mapa de calor 3 × 3 con el AUC de validación, marcando la combinación elegida.
Cómo se lee: "Si los valores son parecidos, la elección no importa mucho."

### 3. Normalizar cada cohorte objetivo (por examen)
- Para cada examen: `zscore(X, fuera_del_examen)`. Media y DE de los pacientes que no están en ese examen, sin etiquetas.
- Texto: "La regla de medir es la misma para las 6 estrategias y todos los n. Solo cambia de un examen a otro."

### 4. Estrategia 4 en cada examen
- AUC del modelo de la fuente en cada examen. Un valor por examen.

### 5. Estrategia 5: ElasticNet local
Para cada examen, n y repetición:
- Mezcla fija = la elegida en el paso 2.
- Fuerza `C` por validación cruzada de 3 partes **dentro de los n locales**, estratificada. Rejilla `C` ∈ {0.01, 0.1, 1, 10}. Si hay empate, se elige el C más chico (frena más).
- Entrenar con los n y medir el AUC en el examen.
- Texto: "Con 10 pacientes, la validación cruzada es ruidosa, pero no toca el examen. Es justo."

### 6. Estrategia 6: fuente + locales
Para cada examen, n y repetición:
- Juntar los 2,860 de la fuente y los n locales.
- Pesos: fuente = 1 por paciente. Locales = `N_fuente / n` cada uno (así suman el 50%). Después se reescalan todos para que sumen `N_fuente + n` (así C significa lo mismo que en la estrategia 4).
- Mezcla y C = los de la estrategia 4. Arranca desde los pesos del modelo de la fuente (`warm_start`); llega al mismo resultado, pero más rápido.
- **Sensibilidad:** pesos 25% (`N_fuente / (3n)`) y 10% (`N_fuente / (9n)`), solo con n = 10 y 40.

### 7. Resumen
Tabla `05_resumen.csv`: por cohorte, estrategia y n → AUC promedio y DE entre repeticiones (DE calculada en cada examen y después promediada en los 5).

**Figura 05-2 · AUC según el número de pacientes locales** (`05_auc_vs_n.png`)
Dos paneles (CPTAC, SMC). Eje x = n. Eje y = AUC en el examen. Línea 5 y línea 6 con barras = ± DE. Línea punteada horizontal = estrategia 4 (promedio de los 5 exámenes).
Cómo se lee: "Donde una curva cruza la línea punteada, esa estrategia empieza a dar un AUC mayor que aplicar directo."

**Figura 05-3 · Estabilidad según el número de pacientes locales** (`05_de_vs_n.png`)
Dos paneles. Eje y = DE del AUC entre repeticiones. Líneas 5 y 6.
Cómo se lee: "Más abajo = más estable. Se espera que la DE baje al aumentar n."

**Figura 05-4 · Sensibilidad al peso de los locales** (`05_sensibilidad_peso.png`)
AUC promedio (puntos) y DE (barras) de la estrategia 6 con 10%, 25% y 50%, para n = 10 y 40.
Cómo se lee: "Si los tres pesos dan casi lo mismo, la conclusión no depende del peso elegido."

## Tiempo (estimado con el notebook 03)
- Paso 2: ~1–2 horas (45 modelos con 2,860 pacientes).
- Estrategia 5: minutos.
- Estrategia 6: el notebook 03 tardó ~9 s por modelo con 6 núcleos. 4,500 modelos al 50% ≈ **8–11 horas**. Sensibilidad (4,000 modelos) ≈ **8–10 horas**. Conviene correrla de noche y guardar por examen.
