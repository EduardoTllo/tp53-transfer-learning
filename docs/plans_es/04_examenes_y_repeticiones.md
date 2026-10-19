# Plan · 04 Exámenes y repeticiones

## Para qué sirve
Elige, una sola vez, qué pacientes van a cada examen y qué pacientes locales se usan en cada repetición. Todas las estrategias leen este archivo. Así la comparación es pareada: en la repetición 37, todas entrenan con los mismos pacientes y se evalúan en el mismo examen.

## Entrada
- `deep_learning/datos/processed/y_cptac.csv`
- `deep_learning/datos/processed/y_smc_2018.csv`

Solo hacen falta las etiquetas, para estratificar por TP53. No se usa la expresión.

## Salida
| Archivo | Columnas | Filas aprox. |
|---|---|---|
| `deep_learning/datos/processed/experimento/examenes.csv` | `cohorte`, `examen` (1–5), `muestra` | 2 × 5 × 40 = 400 |
| `deep_learning/datos/processed/experimento/repeticiones.csv` | `cohorte`, `examen`, `n`, `repeticion` (0–99), `muestra` | ~125,000 |
| `deep_learning/resultados/tablas/04_resumen_examenes.csv` | Tabla del paso 3 | 10 |

Una fila por paciente elegido. Formato largo: es fácil de filtrar (`examen == 2 and n == 10 and repeticion == 37`).

## Secciones del notebook

### 0. Título y texto inicial
> **04 · Exámenes y repeticiones**
> Este notebook elige qué pacientes se usan para evaluar y cuáles para entrenar. Se hace una sola vez y se guarda en un archivo.
> - Un **examen** son 40 pacientes de la cohorte objetivo que nunca se usan para entrenar. Hay 5 exámenes por cohorte.
> - Una **repetición** es una elección al azar de n pacientes locales entre los que no están en el examen. Hay 100 por examen y por n.
> Los dos se estratifican por TP53: cada grupo tiene la misma proporción de mutados que la cohorte. Así un grupo de 10 nunca queda sin mutados.

### 1. Cargar etiquetas
- `y = pd.read_csv(..., index_col=0)` para cada cohorte.
- Mostrar: pacientes y % de mutados (CPTAC 105, 39.0%; SMC 167, 48.5%).

### 2. Elegir exámenes y repeticiones
Código (un solo bucle, sin funciones extra):
```
para cada cohorte:
    para e en 0..4:
        resto, examen = train_test_split(ids, test_size=40, stratify=y, random_state=2026 + e)
        guardar examen
        para n en N[cohorte]:            # CPTAC: 10, 15, 20, 40 · SMC: 10, 15, 20, 40, 80
            para r en 0..99:
                locales, _ = train_test_split(resto, train_size=n, stratify=y[resto], random_state=100000*e + 1000*n + r)
                guardar locales
```
Texto: "Las semillas fijas hacen que cualquiera obtenga exactamente los mismos grupos."

### 3. Revisar que todo esté bien
Tabla `04_resumen_examenes.csv`: por cohorte y examen → pacientes en el examen, mutados en el examen, pacientes fuera del examen.

Comprobaciones con `assert` (si algo falla, el notebook se detiene):
- Cada examen tiene 40 pacientes.
- Ningún paciente local de una repetición está en su examen.
- Cada repetición tiene exactamente n pacientes y al menos 1 mutado y 1 no mutado.

### 4. Gráficas
**Figura 04-1 · Cuánto se repiten los pacientes entre exámenes** (`deep_learning/resultados/figuras/04_solapamiento_examenes.png`)
Matriz 5 × 5 por cohorte (dos paneles): cuántos pacientes comparten cada par de exámenes.
Cómo se lee: "Los exámenes comparten pacientes porque la cohorte es chica. Por eso no se tratan como cohortes distintas: la conclusión vale para cada cohorte."

**Figura 04-2 · Cuántas veces se usa cada paciente como local** (`04_uso_pacientes.png`)
Histograma por cohorte, solo n = 10.
Cómo se lee: "Si un paciente apareciera en casi todas las repeticiones, pesaría demasiado. Lo esperado es que todos aparezcan un número parecido de veces."

### 5. Guardar
- Escribir los tres archivos.
- Texto final: "Desde aquí, ningún notebook vuelve a elegir pacientes al azar. Todos leen `examenes.csv` y `repeticiones.csv`."

## Tiempo
Menos de 1 minuto.
