# Plan · 09 Resultados

## Para qué sirve
Junta las 6 estrategias y responde las preguntas del estudio con las reglas fijadas el 12-oct. No entrena nada.

## Entrada
- `deep_learning/datos/processed/experimento/auc_elasticnet.csv` (notebook 05)
- `deep_learning/datos/processed/experimento/auc_red.csv` (notebook 08)
- `deep_learning/scripts/comun.py` (`pitman_morgan`, `stouffer`)

## Salida
Figuras en `deep_learning/resultados/figuras/` y tablas en `deep_learning/resultados/tablas/` (nombres abajo).

## Secciones del notebook

### 0. Título y texto inicial
> **09 · Resultados**
> Este notebook responde tres preguntas:
> 1. ¿Qué AUC da cada estrategia? → AUC promedio.
> 2. ¿Qué estrategia es más estable? → DE del AUC entre las 100 repeticiones.
> 3. ¿Con cuántos pacientes locales conviene el ajuste fino?
> Todo se calcula por separado en CPTAC y en SMC. Que las dos coincidan es la evidencia más directa de que el resultado se repite.

### 1. Juntar resultados
- Unir los dos archivos de AUC (peso 50% para la estrategia 6; las otras filas de peso van a la sección 7).
- Comprobar: 100 repeticiones por examen, n y estrategia (2, 3, 5, 6); 1 valor por examen en 1 y 4.

### 2. Pregunta 1 · ¿Qué AUC da cada estrategia?
Tabla `09_resumen.csv`: por cohorte, estrategia y n → AUC promedio (5 exámenes) y DE entre repeticiones (promedio de la DE de cada examen).

**Figura 09-1 · AUC según el número de pacientes locales** (`09_auc_vs_n.png`) · *figura central del paper*
Dos paneles (CPTAC, SMC). Eje x = n (10, 15, 20, 40, 80). Eje y = AUC en el examen. Cuatro líneas con barras ± DE: 2, 3, 5, 6. Dos líneas punteadas horizontales: 1 y 4. Mismo color por familia (red / ElasticNet), distinto trazo.
Cómo se lee: "Cada punto es el AUC promedio con n pacientes locales. Las barras muestran cuánto cambia según qué pacientes tocaron. Las líneas punteadas son no ajustar nada."

### 3. Pregunta 2 · ¿Qué es más estable?
**Figura 09-2 · Estabilidad según n** (`09_de_vs_n.png`)
Dos paneles. Eje y = DE del AUC entre repeticiones. Líneas 2, 3, 5, 6.
Cómo se lee: "Más abajo = más estable. Las estrategias 1 y 4 no aparecen porque no usan pacientes locales: su DE es 0."

**Figura 09-3 · Los 100 AUC con n = 10** (`09_cajas_n10.png`)
Cajas por estrategia (2, 3, 5, 6), un panel por cohorte, los 5 exámenes juntos con un color por examen.
Cómo se lee: "Una caja corta = el resultado casi no depende de qué 10 pacientes tocaron."

**Comparaciones principales** (n = 10, sin corrección):
1. Estrategia 2 contra 3: ¿el preentrenamiento hace más estable a la red?
2. Estrategia 2 contra 6: ¿la red es más estable que el ElasticNet que también usa la fuente?

Para cada una, en cada cohorte:
- En cada examen: `pitman_morgan(auc_2, auc_x)` con las 100 repeticiones pareadas → z con signo (+ = la estrategia 2 es más estable).
- Juntar los 5 z con `stouffer` → Z y p de dos colas. Significativo si |Z| > 1.96.
- Tabla `09_principales.csv`: cohorte, comparación, DE de cada estrategia, razón de DE, z de cada examen, Z combinado, p.

**Figura 09-4 · Comparaciones principales** (`09_forest_principales.png`)
Gráfico de bosque. Una fila por examen (5) y una fila "Combinado" en negrita, para cada comparación y cohorte. Eje x = z. Líneas verticales en −1.96, 0 y +1.96.
Cómo se lee: "Puntos a la derecha de +1.96 = la estrategia 2 es más estable. Si los 5 exámenes caen del mismo lado, el resultado es consistente."

**Comparaciones de apoyo:** los otros pares de 2, 3, 5 y 6, y los otros n. Mismo cálculo. Valores p corregidos con Benjamini-Hochberg. Tabla `09_apoyo.csv`.

### 4. Pregunta 3 · ¿Con cuántos pacientes conviene el ajuste fino?
Regla fijada el 12-oct: el n más chico en que el AUC promedio (5 exámenes) del modelo ajustado es mayor que el de aplicar directo. Red: 2 contra 1. ElasticNet: 6 contra 4.
- La resta se hace examen por examen (mismo examen), y después se promedia.
- Tabla `09_n_recomendado.csv`: cohorte, familia, diferencia por n, n recomendado (o "ninguno").

**Figura 09-5 · Diferencia de AUC con aplicar directo** (`09_diferencia_vs_n.png`)
Eje x = n. Eje y = AUC ajustado − AUC aplicar directo (mismo examen), promedio y rango de los 5 exámenes. Línea en 0. Red y ElasticNet.
Cómo se lee: "Arriba de 0 = ajustar con los pacientes locales mejora. El primer n arriba de 0 es el número recomendado. Si nunca pasa de 0, el modelo de la fuente sirve tal cual."

### 5. Resultado secundario: estabilidad por paciente
Con `prob_red.parquet` y `prob_elasticnet.parquet`: para cada paciente del examen, la DE de su probabilidad entre repeticiones.

**Figura 09-6 · Cuánto cambia la predicción de cada paciente** (`09_prob_por_paciente.png`)
Cajas de esa DE por estrategia, n = 10.
Cómo se lee: "Aunque el AUC sea parecido, un paciente puede pasar de 'mutado' a 'no mutado' según el modelo. Esta figura muestra cuánto pasa eso."

### 6. Sensibilidad al peso de la estrategia 6
Repetir la Figura 09-2 y la comparación 2 contra 6 con pesos 10% y 25% (solo n = 10 y 40). Tabla `09_sensibilidad_peso.csv`.

### 7. Texto final
Un párrafo por cohorte con los números de las tablas, en el estilo del manuscrito: frases cortas, resultados neutrales ("un resultado es un resultado").

## Cómo quedó el notebook (14-oct)
Escrito y probado con datos de prueba para la red (los de ElasticNet copiados); corre en ~30 s. Falta correrlo con `auc_red.csv` y `prob_red.parquet` reales del 08.

- `pitman_morgan` y `stouffer` agregadas a `comun.py` (comprobadas: Stouffer reproduce el ejemplo de la nota Potencia estadística, Z = 3.76, p = 0.0002).
- Las probabilidades no guardan la etiqueta: se toma de `y_*.csv` para la calibración.
- Análisis principal: estrategia 2 con la configuración principal (tasa 1e-5, 10 épocas) y estrategia 6 al 50%.
- **Tres figuras extra (decisión del autor, 14-oct):** diferencia pareada, calibración y PCA de cohortes.

| Figura | Archivo | Qué muestra |
|---|---|---|
| 09-1 | `09_auc_vs_n.png` | AUC según n, 6 estrategias |
| 09-2 | `09_de_vs_n.png` | DE según n |
| 09-3 | `09_cajas_n10.png` | Los 500 AUC con n = 10, color por examen |
| 09-4 | `09_forest_principales.png` | z de Pitman-Morgan por examen y Stouffer, 2 contra 3 y 2 contra 6 |
| 09-5 | `09_diferencia_pareada.png` | **Nueva.** Histograma de AUC 2 − AUC 3 y 2 − 6 por repetición, % > 0 |
| 09-6 | `09_diferencia_vs_n.png` | AUC ajustado − aplicar directo según n (número de pacientes recomendado) |
| 09-7 | `09_prob_por_paciente.png` | DE de la probabilidad de cada paciente, n = 10 |
| 09-8 | `09_calibracion.png` | **Nueva.** Probabilidad promedio contra proporción real de mutados, 10 grupos |
| 09-9 | `09_pca_cohortes.png` | **Nueva.** PCA de las 4 cohortes antes y después del z-score (datos reales: antes, METABRIC queda aparte y PC1 explica el 90%; después se mezclan) |

| 09-10 | `09_matriz_confusion.png` | **Nueva (16-oct).** Matriz de confusión con umbral 0.5 fijo, promedio de pacientes por examen de 40; sensibilidad y especificidad |

Tablas: `09_confusion.csv`, `09_resumen.csv`, `09_principales.csv`, `09_apoyo.csv` (BH), `09_n_recomendado.csv`, `09_sensibilidad.csv` (ajuste fino con tasa 1e-4; estrategia 6 al 25% y 10%). El paso 7 imprime un párrafo por cohorte con los números.
