# Plan · 08 Red en CPTAC y SMC (estrategias 1, 2 y 3)

## Para qué sirve
Corre las tres estrategias con red en las cohortes objetivo, con todo ya fijado en los notebooks 04, 06 y 07. Aquí la red ve CPTAC y SMC por primera vez. Nada se elige en este notebook.

## Entrada
- `deep_learning/datos/processed/X_cptac.parquet`, `X_smc_2018.parquet`, `y_cptac.csv`, `y_smc_2018.csv`
- `deep_learning/datos/processed/experimento/examenes.csv`, `repeticiones.csv` (notebook 04)
- `deep_learning/datos/processed/experimento/red_fuente.keras` (notebook 06)
- `deep_learning/resultados/tablas/06_elegida.json`, `07_ajuste_elegido.json`
- `deep_learning/scripts/comun.py`

## Salida
| Archivo | Columnas |
|---|---|
| `deep_learning/datos/processed/experimento/auc_red.csv` | `cohorte`, `examen`, `estrategia` (1, 2, 3), `n` (0 en la 1), `repeticion`, `auc` (examen), `auc_locales` (con sus propios pacientes locales; vacío en la 1). Mismo formato que `auc_elasticnet.csv` más `auc_locales` |
| `deep_learning/datos/processed/experimento/prob_red.parquet` | `cohorte`, `examen`, `estrategia`, `n`, `repeticion`, `muestra`, `prob` (probabilidad de mutado de cada paciente del examen) |
| `deep_learning/datos/processed/experimento/perdida_red.parquet` | `cohorte`, `examen`, `estrategia` (2, 3), `n`, `repeticion`, `epoca`, `perdida` (pérdida de **entrenamiento** por época, la que ya devuelve `fit`) |

Se guarda al terminar cada examen. Si se corta, sigue desde el último guardado.

**Dos configuraciones del ajuste fino (14-oct, decisión del autor):** principal = tasa 1e-5 y 10 épocas (elegida en el 07); sensibilidad = tasa 1e-4 y 10 épocas (empató en AUC en el 07, pero sí cambia la red). Columna nueva `variante` (`principal` / `sensibilidad`) en `auc_red.csv`, `prob_red.parquet` y `perdida_red.parquet`. Las figuras de control usan solo la principal. Se reportan las dos; no se elige una según su resultado.

## Secciones del notebook

### 0. Título y texto inicial
> **08 · Red neuronal en CPTAC y SMC**
> Se corren tres estrategias con la misma red:
> - **1. Aplicar directo:** la red de la fuente tal cual.
> - **2. Ajuste fino:** la red de la fuente, ajustada con n pacientes locales (todas las capas).
> - **3. Desde cero:** la misma forma de red, con pesos nuevos, entrenada solo con los n locales.
> La tasa y las épocas de 2 y 3 se eligieron en el ensayo general (notebook 07) y no se cambian.

### 1. Cargar lo fijado
- Mostrar en una tabla lo que se usa: arquitectura, lote, optimizador (06) y tasa y épocas de 2 y 3 (07).

### 2. Bucle principal
```
para cada cohorte (CPTAC, SMC):
    para cada examen (1..5):
        zscore(X, fuera_del_examen)
        estrategia 1: AUC de red_fuente en el examen
        para cada n y cada repetición (0..99):
            locales = repeticiones.csv filtrado
            estrategia 2: copiar red_fuente → entrenar con locales → AUC en el examen
            estrategia 3: armar_red() con semilla = repetición → entrenar con locales → AUC
        guardar auc_red.csv y prob_red.parquet
```
- Pesos por clase también en el ajuste fino.
- Texto: "La estrategia 2 parte de lo que la red aprendió con 2,860 pacientes. La 3 parte de cero. Es la única diferencia entre las dos."

### 3. Control rápido (no es el análisis final)
**Figura 08-1 · Primer vistazo con n = 10** (`08_vistazo_n10.png`)
Dos paneles (CPTAC, SMC). Para cada examen, cajas con los 100 AUC de las estrategias 2 y 3, y un punto con la estrategia 1.
Cómo se lee: "Cada caja muestra cuánto varía el AUC según qué 10 pacientes tocaron. Una caja más corta es una estrategia más estable."

**Figura 08-3 · Brecha de sobreajuste según n** (`08_brecha_vs_n.png`)
Dos paneles (CPTAC, SMC). Eje x = n. Eje y = AUC con los pacientes locales − AUC en el examen, promedio ± DE. Líneas 2 y 3.
Cómo se lee: "Una brecha grande = la red memorizó a sus pocos pacientes. Si la preentrenada tiene menos brecha que la desde cero, el preentrenamiento protege contra la memorización." (Decisión del autor, 13-oct.)

**Figura 08-2 · Pérdida de entrenamiento por época** (`08_perdida_entrenamiento.png`)
2 × 2 paneles: filas = CPTAC y SMC; columnas = n = 10 y n = 40. Una línea por estrategia (2 y 3): promedio de las 500 redes (5 exámenes × 100 repeticiones) y banda = ± DE.
Cómo se lee: "Es un control: muestra que el ajuste fino y la red desde cero sí entrenaron (la pérdida baja). Solo usa los pacientes locales. **No se grafica nada del examen por época**: mirar el AUC del examen época por época invita a elegir las épocas con el examen, que es lo que se evita. Las épocas ya se fijaron en el ensayo general (07)."

- Comprobar: 100 AUC por examen, n y estrategia; ningún vacío.
- El análisis completo está en el notebook 09.

## Tiempo
(4 n en CPTAC + 5 en SMC) × 5 exámenes × 100 repeticiones × 2 estrategias = 9,000 entrenamientos chicos. Varias horas. Correr de noche.

Prueba mínima (14-oct, red real del 06, tasa y épocas de prueba (5 y 10 épocas), 2 repeticiones): 190 redes en ~1 min en la Mac, todo corre. Con 100 repeticiones y las épocas reales del 07, en la Mac serían del orden de 1–3 horas; en Khipu, más (es más lento por núcleo).

Detalles del código (14-oct): lote = todos los n locales; semilla = repetición; el ajuste fino copia los pesos de `red_fuente.keras` en una red armada con `armar_red` (optimizador nuevo, tasa del 07). Guarda por examen en `experimento/partes_08/`.
