# Mundial 2026 — Predictor con Machine Learning

¿Y si pudiéramos predecir el Mundial 2026 utilizando un modelo de Machine Learning?

Este proyecto entrena un modelo XGBoost sobre datos históricos de partidos internacionales para estimar los **goles esperados (XG)** de cada equipo, y después usa esas predicciones para **simular el torneo completo** de 48 equipos de forma probabilística. Los resultados se visualizan en una web interactiva.

---

## Índice

- [Cómo funciona](#cómo-funciona)
- [Ejecución](#ejecución)
- [Los tres modelos: en qué se basa cada uno](#los-tres-modelos-en-qué-se-basa-cada-uno)
- [Cómo leer los resultados en la web](#cómo-leer-los-resultados-en-la-web)
- [Monte Carlo: qué es y para qué sirve aquí](#monte-carlo-qué-es-y-para-qué-sirve-aquí)
- [Apartado técnico: cómo se entrena el modelo](#apartado-técnico-cómo-se-entrena-el-modelo)
- [Validación del modelo y mejoras recientes](#validación-del-modelo-y-mejoras-recientes)
- [Limitaciones y cómo se podría mejorar](#limitaciones-y-cómo-se-podría-mejorar)
- [Estructura de datos](#estructura-de-datos)
- [Formato de grupos (Mundial 2026)](#formato-de-grupos-mundial-2026)

---

## Cómo funciona

El sistema tiene dos fases independientes:

### Fase 1 — Entrenamiento del modelo (`src/xg_preds.py`)

1. Carga el histórico de partidos internacionales desde 2016 hasta la fecha del Mundial (`data/models_csv/df_<variante>.csv`).
2. Calcula *features* por equipo: medias móviles de goles a favor y en contra (5 y 15 partidos), ELO medio del rival, PCA de estilo de juego (2 componentes), confederación e importancia del torneo (1–5, siendo el Mundial = 5), además de las mismas métricas del rival.
3. Aplica un **peso temporal exponencial** que da más importancia a los partidos recientes (semivida de **1277 días**, ≈ 3,5 años; fecha de referencia: 1 de junio de 2024).
4. Entrena un **regresor Tweedie con XGBoost**. Los hiperparámetros se afinaron con búsqueda aleatoria (`RandomizedSearchCV`) sobre validación temporal (`TimeSeriesSplit`) y quedaron fijados en el código.
5. Genera una **"foto fija"** con las métricas del último partido de cada selección antes del Mundial.
6. Exporta las predicciones de XG para cada jornada de la fase de grupos a `data/ai_models/xg_preds_J{1,2,3}_<variante>.csv`.

### Fase 2 — Simulación del torneo (`src/simulacion.py` + `src/clases_simulacion.py`)

1. Lee los CSV de XG generados en la Fase 1.
2. Simula cada partido **minuto a minuto** usando el `XG/90` como tasa de Poisson, con multiplicadores dinámicos según el marcador, el minuto y la fortaleza relativa de cada equipo.
3. Cada partido se itera **30 veces** para obtener una distribución de resultados y extraer el **más probable**.
4. Los partidos de eliminatorias incluyen **prórroga** y **penaltis** (75 % de acierto por lanzamiento).
5. La clasificación de los **8 mejores terceros** sigue las pautas de `data/mejores_terceros.csv`.
6. Exporta los resultados completos a `results/predictions_<variante>.csv`.

> El proyecto incluye además una **web en Astro** (`web/`) que convierte estos CSV en una visualización interactiva. Ver [Cómo leer los resultados en la web](#cómo-leer-los-resultados-en-la-web).

---

## Ejecución

### Modelo y simulación (Python)

```bash
# Dependencias (instalar manualmente)
pip install pandas numpy scikit-learn xgboost

# Reentrenar el modelo y generar predicciones de fase de grupos
python src/xg_preds.py

# Simular el torneo completo (genera los CSV de resultados)
python src/simulacion.py
```

Los dos scripts deben ejecutarse desde la raíz del repositorio.

### Web de visualización (Astro)

```bash
# Desde la raíz: convierte los CSV en módulos TypeScript
python3 scripts/export_data.py

# Arrancar la web en local
cd web && npm install && npm run dev
```

---

## Los tres modelos: en qué se basa cada uno

El motor de Machine Learning es **siempre el mismo**: lo que cambia entre los tres modelos es el **escenario de fuerzas** que se le plantea. Cada modelo aplica unos ajustes de ELO (el indicador de fortaleza de cada selección) sobre las valoraciones base, reflejando una **hipótesis distinta** sobre qué bloque continental llegará más fuerte al torneo. Es como preguntar a tres analistas con sesgos diferentes.

| Modelo | En qué se basa | Ajustes de ELO |
|---|---|---|
| **misterclaude** | El "analista neutral". Toma la salida pura del modelo, sin tocar nada. Es la predicción base, la más objetiva. | Sin ajustes |
| **gemaldini** | El "europeísta". Apuesta por que las grandes potencias europeas rendirán por encima de su nivel histórico. | +200 a Francia, España, Portugal, Inglaterra y Noruega · +100 al resto de Europa · +50 a Alemania y Países Bajos |
| **dav_gpo** | El "sudamericanista". Apuesta por un Mundial fuerte de CONMEBOL y la zona americana. | +200 a Argentina, Colombia, Ecuador y Paraguay · +150 a Uruguay y México · +100 a Brasil · +50 a Panamá |

**¿Por qué tres y no uno?** Porque ningún ajuste manual es "la verdad". Comparar los tres permite distinguir lo **robusto** de lo **frágil**: si los tres modelos coinciden en que un equipo pasa de ronda, esa predicción es sólida; si discrepan, ese cruce es una moneda al aire. La web hace visible justamente esa coincidencia o discrepancia.

> Los ajustes de cada modelo están documentados en `src/simulacion.py` y se aplican durante la preparación de los datos de cada variante.

---

## Cómo leer los resultados en la web

La web está pensada para que **cualquier persona** entienda las predicciones sin saber de modelos. Tiene un **selector de modelo** (MC = misterclaude, GEM = gemaldini, DAV = dav_gpo) para cambiar entre los tres escenarios, y cuatro pestañas:

### 1. Fase de Grupos

Muestra los **12 grupos (A–L)** con su clasificación final predicha y el resultado de cada partido. Cada tarjeta de partido (para partidos aún no jugados) contiene cuatro capas de información:

#### A. Marcador del modelo (número grande)
El resultado que el simulador Python extrae de **30 iteraciones** del partido. Cada iteración simula el partido **minuto a minuto**: `XG/90` es la tasa de Poisson por minuto, modulada por multiplicadores dinámicos según marcador y tiempo restante. De las 30 tiradas, se escoge el marcador más frecuente (la moda). Es el resultado más probable **según el simulador adaptativo completo**.

#### B. Barras de XG
Muestran el XG de cada equipo en ese partido: cuántos goles de calidad espera generar el modelo.

#### C. Barra 1X2 + probabilidades por resultado
Debajo de las barras XG, una barra tricolor muestra la **probabilidad de victoria del equipo local / empate / victoria del visitante**:
- Calculada **analíticamente** con la distribución de Poisson: `P(g1, g2) = Poisson(xg1, g1) × Poisson(xg2, g2)` sumada sobre todos los marcadores con `g1 > g2`, `g1 = g2` o `g1 < g2`.
- Es equivalente a ejecutar infinitas simulaciones Poisson puras — no es una estimación, es el valor **exacto** de la distribución.

#### D. Probabilidad del marcador predicho
Debajo del marcador (en pequeño) aparece el porcentaje de probabilidad de **ese marcador exacto**, calculado analíticamente con Poisson a partir del XG:

```
España  2 – 1  Uruguay
           8.4% exacto
```

Esto responde a: *"el modelo dice 2-1, ¿pero cuántas veces de cada 100 pasaría eso?"*. Valores típicos: 5–12%. El fútbol es impredecible incluso cuando el modelo tiene claro quién gana.

> **Nota técnica:** el marcador grande viene del simulador Python (30 iteraciones minuto a minuto con multiplicadores adaptativos). La probabilidad del `%  exacto` la calcula la web en tiempo de construcción usando la fórmula de Poisson directamente sobre el XG, sin necesidad de correr simulaciones adicionales — es el resultado matemático exacto.

#### E. Columna Clas% en la tabla de clasificación
La probabilidad de que cada equipo **clasifique a octavos de final (R32)**, calculada con las **500 simulaciones Monte Carlo** (ver sección Monte Carlo más abajo). En verde ≥ 70 %, en ámbar ≥ 40 %, en gris por debajo.

> Para partidos ya jugados, el marcador real sustituye al predicho y desaparecen las barras de probabilidad.

### 2. Cuadro del Torneo

El **bracket** completo desde dieciseisavos hasta la final. Aquí aparece la pieza clave para interpretar la fiabilidad:

- **Indicador de consenso** en cada cruce, que resume si los **tres modelos coinciden** en quién pasa:
  - **✓ 3/3** (verde): los tres modelos coinciden → predicción **sólida**.
  - **~ 2/3** (ámbar): dos de tres coinciden → predicción **dudosa**.
  - **✗ 0/3** (rojo): los tres discrepan → **totalmente abierto**.

Así, de un vistazo, sabes en qué partes del cuadro fiarte y en cuáles no.

### 3. Resultado Final

El resumen del torneo:

- **Campeón** según cada uno de los tres modelos (puede haber tres campeones distintos).
- **Probabilidades Monte Carlo:** para cada variante, un gráfico de barras muestra las **probabilidades de ser campeón** de los 12 mejores equipos según las 500 simulaciones Monte Carlo (ver sección Monte Carlo más abajo).
- **Shock del torneo:** el mayor **batacazo**. Es el partido en el que el ganador tenía, según su ELO, la **menor probabilidad de ganar**: la sorpresa más grande de toda la simulación.
- **Camino al título:** el recorrido completo del campeón, ronda a ronda.
- Estadísticas como el **partido con más goles** del torneo.

### 4. Simulador Interactivo

Tiene dos modos:

**Simulación única** — Lanza un torneo completo en el navegador y muestra un resultado distinto cada vez. La variedad entre ejecuciones refleja la incertidumbre real del modelo: el fútbol es probabilístico, no determinista.

**Distribución Monte Carlo (500 sims)** — Ejecuta 500 torneos completos en el navegador y muestra un ranking de campeones con la frecuencia con que ganó cada equipo (`N/500`). Se ejecuta en segundo plano (cediendo el hilo al navegador cada 50 sims) para no bloquear la interfaz.

> Nota técnica: el simulador del navegador reproduce el motor de Poisson del Python, pero para el XG de las eliminatorias usa una **aproximación basada en ELO** (`eloToXG()`), porque el modelo XGBoost no puede ejecutarse en el navegador. Las probabilidades de Monte Carlo en la pestaña Resultado se generan con el modelo XGBoost real (en Python, fuera del navegador).

---

## Monte Carlo: qué es y para qué sirve aquí

### El problema del marcador modal

El simulador principal (`src/simulacion.py`) corre 30 iteraciones por partido y se queda con el **marcador más frecuente** (la moda). Es eficiente para obtener una predicción concreta, pero oculta la incertidumbre: no dice nada sobre la probabilidad de que España llegue a semifinales, solo dice "España llegó" o "no llegó" en ese único camino.

### Qué hace el Monte Carlo

`src/monte_carlo.py` simula el torneo completo **N veces** (por defecto 1 000, se puede ajustar) con **tiradas Poisson puras** (una sola por partido, sin multiplicadores adaptativos, para ser eficiente). En cada simulación:

1. **Fase de grupos** — 6 partidos por grupo con Poisson(xg1) y Poisson(xg2), clasificación real.
2. **Octavos (R32)** — emparejamiento según mejores terceros (`data/mejores_terceros.csv`), igual que el torneo real.
3. **Sweet16 → Elite8 → Semis → Final** — brackets fijos; en eliminatorias, los empates van a prórroga y penaltis.
4. Se registra en qué rondas llegó cada equipo.

Al terminar las N simulaciones, se calcula la **frecuencia relativa** de cada ronda para cada equipo:

| Probabilidad | Significado |
|---|---|
| `pR32` | % de torneos en que clasificó a octavos |
| `pS16` | % de torneos en que pasó a dieciseisavos |
| `pE8` | % de torneos en que llegó a cuartos |
| `pSemis` | % de torneos en que llegó a semifinales |
| `pFinal` | % de torneos en que llegó a la final |
| `pChampion` | % de torneos en que ganó el Mundial |

### Dónde se ven estas probabilidades

- **Grupos → columna Clas%:** `pR32` de cada equipo, junto a su clasificación predicha.
- **Resultado → "Probabilidades de campeón":** ranking con `pChampion` por variante.
- **Simular → botón "Distribución (500 sims)":** versión en navegador (aproximación ELO).

### ¿Por qué no usar Monte Carlo para el marcador de cada partido?

El marcador más probable de un partido Poisson(xg1) vs Poisson(xg2) tiene solución **analítica exacta**: el argmax es `(floor(xg1), floor(xg2))` y su probabilidad se calcula en O(N²) directamente. Correr 500 simulaciones convergería al mismo resultado pero con varianza estadística. No hay ganancia. Por eso:

- **Para marcadores** → Poisson analítico (exacto, instantáneo).
- **Para probabilidades de avance** → Monte Carlo (necesario porque depende de *cómo se encadenan* los partidos a lo largo del torneo: los grupos determinan los emparejamientos de octavos, que determinan los de cuartos, etc.).

### Cómo ejecutar el Monte Carlo

```bash
# Desde la raíz del repositorio (requiere haber ejecutado xg_preds.py antes)
python3 src/monte_carlo.py         # 1 000 simulaciones por variante (por defecto)
python3 src/monte_carlo.py 5000    # 5 000 simulaciones (más precisas, más lentas)

# Luego regenerar los datos de la web:
python3 scripts/export_data.py
```

Los resultados se guardan en `results/probabilities_<variante>.json` y se convierten a TypeScript en `web/src/data/probabilities.ts`.

---

## Apartado técnico: cómo se entrena el modelo

Esta sección detalla el funcionamiento interno para quien quiera entender —o mejorar— el modelo.

### Los datos

- **Fuente:** ~3 millones de filas de partidos internacionales desde 2016, un dataset por variante (`data/models_csv/df_<variante>.csv`). Cada fila es un partido visto **desde la perspectiva de un equipo** (equipo, rival, goles, ELO, torneo, localía, fecha…).
- Las tres variantes parten del mismo histórico, pero con los ajustes de ELO de su escenario ya incorporados.

### Ingeniería de *features* (20 variables)

El modelo no usa los datos crudos, sino características derivadas que capturan la **forma** y el **contexto** de cada equipo:

| Grupo | Variables |
|---|---|
| **Fortaleza** | `elo` propio y del rival (`opponent_elo`), `elo_prom_5` (ELO medio de los rivales recientes) |
| **Forma reciente** | medias móviles de goles a favor y en contra a 5 y 15 partidos (`gf_prom_5`, `gc_prom_5`, `gf_prom_15`, `gc_prom_15`) |
| **Estilo de juego** | `PCA_1` y `PCA_2`: 2 componentes principales que resumen el estilo a partir de varias métricas de juego |
| **Contexto** | `is_home` (localía), `confed` (confederación: UEFA=1, CONMEBOL=2, CONCACAF=3, CAF=4, AFC=5, OFC=6), `tournament_num` (importancia del torneo, 1–5) |
| **El rival** | las mismas métricas de forma, ELO y estilo, pero del adversario (`rival_*`) |

### Ponderación temporal

Para que un amistoso de 2017 no pese igual que un partido reciente, cada fila se pondera con un **decaimiento exponencial**: `peso = exp(-λ · días_de_antigüedad)`, con `λ = ln(2) / 1277`. Esto da una **semivida de 1277 días**: un partido de hace ~3,5 años cuenta la mitad que uno de la fecha de referencia (1 de junio de 2024). Este peso se pasa como `sample_weight` al entrenar.

### El modelo

- **Objetivo (`target`):** el modelo predice **goles** (`goals`). Esa predicción es la que se interpreta y se usa como **XG** (goles esperados) en la simulación.
- **Algoritmo:** `XGBoost` con objetivo **Tweedie** (`reg:tweedie`, `tweedie_variance_power = 1.3`). Se elige Tweedie porque los goles son una variable **no negativa, discreta y con muchos ceros/valores bajos**, que la distribución de Tweedie modela mejor que una regresión normal.
- **Hiperparámetros** (fijados tras la búsqueda con `RandomizedSearchCV` + `TimeSeriesSplit`):

  ```python
  XGBRegressor(
      objective='reg:tweedie',
      tweedie_variance_power=1.3,
      n_estimators=500,
      learning_rate=0.01,
      max_depth=4,
      subsample=0.7,
      colsample_bytree=0.8,
      random_state=42,
  )
  ```

### De entrenamiento a predicción: la "foto fija"

El modelo no puede ver el futuro, pero sí puede congelar el **estado de cada equipo justo antes del torneo**. Eso es la **foto fija**: el último registro de cada selección en los datos históricos (antes del 11 de junio de 2026), que incluye:

- Las medias móviles de goles a favor/en contra de los últimos 5 y 15 partidos
- La media del ELO de los rivales recientes (proxy de la dificultad del calendario)
- Los componentes de estilo de juego (`PCA_1`, `PCA_2`)
- La confederación y el ELO propio en ese momento

Con esa foto fija como input, el modelo predice el XG esperado de cada partido del calendario real (dividido en J1, J2, J3 por fechas). Es decir: **el modelo entrena sobre el pasado, pero infiere sobre el presente usando el último snapshot disponible** de cada equipo.

La simulación convierte ese XG en goles minuto a minuto: `XG/90` es la probabilidad de marcar en cada minuto (proceso de Poisson), modulada por multiplicadores que reaccionan al marcador y al tiempo restante. 30 iteraciones por partido → se escoge el marcador más frecuente. En eliminatorias se añaden prórroga y penaltis.

### Actualización con resultados reales

Una vez el Mundial ha comenzado, la foto fija se puede actualizar con los resultados reales de cada jornada jugada:

1. `scripts/fetch_real_results.py` obtiene los marcadores de [football-data.org](https://www.football-data.org/) y los guarda en `data/world_cup_results.csv`.
2. `scripts/update_foto_fija.py` lee esos resultados y recalcula para cada equipo que ha jugado:
   - Las medias móviles de goles (la ventana deslizante ahora incluye los partidos reales)
   - El ELO (actualizado con la fórmula estándar, K=40 para el Mundial)
   - Escribe `data/ai_models/foto_fija_updated.csv`
3. Al re-ejecutar `python src/xg_preds.py`, el script detecta `foto_fija_updated.csv` y **sobreescribe automáticamente** las métricas de los equipos que ya han jugado antes de predecir las jornadas siguientes. El modelo XGBoost en sí no se re-entrena (tiene suficiente histórico); solo cambian los inputs de inferencia.
4. `python src/simulacion.py` recalcula el bracket con las predicciones actualizadas.

Todo este proceso se ejecuta **automáticamente cada día a las 8:00 UTC** mediante un workflow de GitHub Actions (`.github/workflows/daily_update.yml`), que además regenera la web y hace el commit. Solo hace falta configurar el secreto `FOOTBALL_DATA_API_KEY` en el repositorio (clave gratuita en [football-data.org](https://www.football-data.org/client/register)).

---

## Validación del modelo y mejoras recientes

Hasta ahora el modelo se entrenaba sobre **todo** el histórico y se daba por bueno sin ninguna métrica que lo respaldara: no había forma de saber si acertaba más que tirar una moneda. Esta tanda de cambios añade el **marco de validación que faltaba**, reactiva la búsqueda de hiperparámetros guiada por esa misma métrica, y corrige varios bugs. La idea de fondo: **dejar de optimizar a ciegas**.

### 1. Validación temporal y calibración (`src/validacion.py`)

Es la pieza más importante. Mide de forma **honesta** si el modelo es bueno, imitando lo que pasaría en la realidad: entrena solo con el pasado y se evalúa sobre partidos que **no ha visto**.

**Cómo funciona (sin trampas):**
- **Corte temporal:** entrena con los partidos anteriores al **16 de septiembre de 2025** (18.394 filas) y se evalúa sobre los posteriores (1.156 filas → 578 partidos reales).
- **Sin fuga de datos:** el PCA y el escalado se ajustan **solo con los datos de entrenamiento** y luego se aplican al test. Si se ajustaran con todo el histórico, el modelo estaría "viendo" el futuro y las métricas saldrían infladas.
- **Reconstrucción 1X2:** empareja las dos filas espejo de cada partido y calcula la probabilidad de victoria local / empate / visitante con la distribución de Poisson, comparándola con el resultado real.

**Resultados (corte 2025-09-16):** el modelo bate a todos los baselines.

| Métrica | Modelo | Baseline | Qué mide |
|---|---|---|---|
| MAE goles | **0,963** | 1,154 (media) | error medio al predecir goles; más bajo = mejor |
| RMSE goles | **1,290** | 1,520 | igual que MAE pero penaliza más los errores grandes |
| Poisson deviance | **1,220** | 1,637 | bondad de ajuste específica para conteos |
| Brier 1X2 | **0,497** | 0,526 (ELO) / 0,667 (azar) | calidad de las probabilidades 1X2 |
| Log-loss 1X2 | **0,845** | 0,897 (ELO) / 1,099 (azar) | penaliza la sobreconfianza |
| ECE (calibración) | **0,059** | — | desviación entre la probabilidad dicha y la real |

> **En cristiano:** el modelo predice goles un **17 % mejor** que "tirar siempre la media", y sus probabilidades de 1X2 superan a un baseline basado solo en ELO. Un ECE de 0,059 significa que cuando dice "60 % de victoria local", acierta cerca del 60 % de las veces — es decir, está bien **calibrado**.

> **Qué mide (y qué NO mide) esta tabla:** compara el modelo contra *baselines* (la media de goles, un baseline de solo-ELO y el azar) para responder *"¿es bueno el modelo?"*. **No** es un "antes vs después" de estos cambios: estas cifras ya eran ciertas del modelo previo. La aportación de esta sección no es un modelo más preciso, sino **poder medirlo por primera vez** (y detectar regresiones en el futuro).

```bash
python3 src/validacion.py                          # las 3 variantes, corte por defecto
python3 src/validacion.py misterclaude 2025-09-16  # una variante, corte concreto
```

Cada ejecución genera `results/validation_<variante>.json` con todas las métricas, la curva de calibración y el desglose de error por confederación.

> Las tres variantes dan métricas idénticas porque comparten el mismo histórico: los ajustes de ELO de cada escenario solo afectan a las 144 filas del Mundial 2026 (partidos futuros), no al pasado con el que se entrena y valida.

### 2. Búsqueda de hiperparámetros reactivada (`tune_hyperparameters`)

La búsqueda `RandomizedSearchCV` que estaba comentada vuelve a estar disponible, ahora **guiada por la misma métrica de validación** (MAE) y con adopción **controlada**.

- Lanza 20 combinaciones × 5 *folds* temporales (`TimeSeriesSplit`) y guarda el resultado en `results/tuning_<variante>.json` (mejores parámetros + importancia de cada *feature*).
- **No adopta nada automáticamente.** Por defecto `train_model()` sigue usando los parámetros fijos; para probar los nuevos hay que llamar a `train_model(name, use_tuned=True)`.

**¿Mejoraron los parámetros buscados?** No. Al evaluarlos en el mismo hold-out:

| | MAE | Log-loss 1X2 | ECE |
|---|---|---|---|
| Fijos (actuales) | **0,963** | **0,845** | 0,059 |
| Tuneados | 0,965 | 0,846 | **0,041** |

El mejor CV-MAE de la búsqueda (0,9355) **no generalizó** al test honesto (0,965 > 0,963). Como la regla de adopción era "solo cambiar si mejora MAE **y** log-loss", **se mantienen los parámetros fijos**. Esto no es un fracaso: ahora **sabemos** que los parámetros que ya teníamos eran buenos, en lugar de suponerlo.

### 3. Corrección de bugs

- **`underdog_20` (bug de pandas 3.0):** la línea original `df["underdog_20"].fillna(0, inplace=True)` **fallaba en silencio** bajo pandas 3.0 (por *copy-on-write*), dejando `NaN` donde debía haber ceros. Esos `NaN` **excluían 785 partidos** del cálculo del PCA. Corregida a `df["underdog_20"] = df["underdog_20"].fillna(0)`, el PCA ahora se ajusta sobre **19.257 filas en vez de 18.472**, recuperando el comportamiento que el autor original pretendía. Es una corrección de **corrección**, no de rendimiento: **cambia** el modelo (usa más datos y elimina un bug silencioso), pero medido en el hold-out las métricas quedan **prácticamente igual** (≈ +0,06 % de MAE frente al modelo anterior, dentro del ruido). En otras palabras: el modelo no predice mejor, simplemente es código correcto y reproducible.
- **Código muerto en `get_multiplier()`:** 36 líneas que se sobrescribían incondicionalmente (`mult1 = mult2 = 1.3`) y nunca llegaban a ejecutarse. Eliminadas; el Python queda alineado con el *port* de TypeScript de la web.
- **"Foto fija" más estable:** en lugar de tomar **solo el último** partido de cada equipo como instantánea pre-Mundial, ahora promedia los **últimos 3**. Reduce el ruido de un único partido atípico. Su efecto neto recae sobre `elo_prom_5`, ya que las demás columnas se sobrescriben por equipo con los resultados reales (`foto_fija_updated.csv`).

### En qué mejora todo esto

Importante: estas mejoras son de **fiabilidad, medición y corrección de código**, no de capacidad predictiva. El modelo predice esencialmente igual que antes; lo que cambia es que ahora **sabemos** lo bueno que es y el código es correcto.

| Antes | Ahora |
|---|---|
| Se entrenaba sin métricas: imposible saber si el modelo era bueno | Validación temporal honesta, con números concretos frente a baselines |
| Hiperparámetros fijos "porque sí" | Hiperparámetros **validados** contra una búsqueda guiada por datos (resultaron buenos) |
| Un bug de pandas dejaba 785 partidos fuera del PCA | El PCA aprovecha todo el histórico disponible (métricas iguales, código correcto) |
| Instantánea pre-Mundial basada en un solo partido | Promedio de los 3 últimos, más estable frente a partidos atípicos |

---

## Limitaciones y cómo se podría mejorar

El modelo es sólido como punto de partida, pero tiene márgenes claros de mejora:

**Limitaciones actuales**
- **Sin datos de jugadores:** no contempla lesiones, sanciones, convocatorias ni el estado de forma de futbolistas concretos. Una baja clave no se refleja.
- **Ajustes de ELO manuales:** los tres escenarios son hipótesis subjetivas, no aprendidas de los datos.
- **Sesgo del "marcador más probable":** tanto la moda de 30 iteraciones como el Poisson analítico tienden a favorecer resultados bajos y comunes (1-0, 1-1), infrarrepresentando goleadas (un 4-0 puede ser posible pero su probabilidad puntual sigue siendo baja).
- **Goles independientes:** el Poisson asume que los goles de cada equipo son independientes, sin correlación entre ataques (lo que infravalora empates y resultados ajustados). Una corrección conocida es el ajuste de **Dixon-Coles**.
- **Sin calibración externa:** no se contrasta con cuotas de casas de apuestas ni con probabilidades de mercado.
- **Monte Carlo con Poisson puro:** las probabilidades de avance usan Poisson puro (sin multiplicadores adaptativos) por eficiencia. El simulador principal (30 iter.) usa lógica minuto-a-minuto más refinada, pero es demasiado lento para miles de torneos.

> **Ya resuelto:** la falta de validación y la búsqueda de hiperparámetros desactivada (que figuraban aquí como pendientes) se abordaron en [Validación del modelo y mejoras recientes](#validación-del-modelo-y-mejoras-recientes). Hoy existe un marco de validación temporal con métricas frente a baselines, y la búsqueda de hiperparámetros está reactivada con adopción controlada.

**Posibles mejoras**
- **Adoptar los hiperparámetros tuneados solo cuando mejoren la validación:** la búsqueda ya está disponible (`tune_hyperparameters`); falta automatizar su evaluación en el hold-out dentro del reentrenamiento para adoptar nuevos valores únicamente si baten MAE **y** log-loss.
- **Incorporar datos de plantilla:** disponibilidad de titulares, minutos recientes, valor de mercado o un ELO por jugador.
- **Modelo de goles correlacionado:** usar un Poisson bivariante o el ajuste de **Dixon-Coles** para capturar la dependencia entre los goles de ambos equipos.
- **Calibrar el Monte Carlo contra el simulador adaptativo:** correr el Monte Carlo con los multiplicadores minuto-a-minuto (más preciso) aunque sea más lento, y comparar las probabilidades resultantes con las del Poisson puro para cuantificar el sesgo.
- **Calibrar contra el mercado** (cuotas) para validar y corregir sesgos sistemáticos.
- **Más iteraciones por partido** para estimaciones más estables.
- **Features de tiro/xG real** (datos de disparos) en vez de derivar el XG solo de goles históricos.
- **PCA dinámico**: actualmente el estilo de juego (`PCA_1`, `PCA_2`) se mantiene fijo de la foto fija base y no se actualiza con los partidos del Mundial — re-entrenar el PCA incrementalmente mejoraría la precisión.

---

## Estructura de datos

```
data/
  models_csv/          # Históricos de entrenamiento (un CSV por variante, ~3M filas)
  ai_models/           # XG predichos para cada jornada de grupos (salida de xg_preds.py)
    xg_preds_J{1,2,3}_<variante>.csv           # XG por jornada de grupos
    xg_preds_J1_<variante>_complete.csv        # "Foto fija" con métricas completas por equipo
  mejores_terceros.csv # Cuadro de emparejamientos para los mejores terceros
  world_cup_results.csv # Resultados reales (auto-actualizado por CI/CD)

api/models/            # Modelos XGBoost guardados (usados por Monte Carlo para XG de KO)
  xgb_<variante>.json

results/               # Salidas de simulacion.py, monte_carlo.py y validacion.py
  predictions_<variante>.csv       # Bracket completo (104 filas)
  probabilities_<variante>.json    # Probabilidades Monte Carlo por equipo y ronda
  validation_<variante>.json       # Métricas de validación temporal + calibración
  tuning_<variante>.json           # Mejores hiperparámetros de la búsqueda (no adoptados)

src/
  simulacion.py        # Simulación modal (30 iter/partido) → predictions CSV
  clases_simulacion.py # Clases Team, Match, Group, Knockouts, Tournament
  monte_carlo.py       # Monte Carlo (N torneos completos) → probabilities JSON
  xg_preds.py          # Entrenamiento XGBoost + predicciones de XG por jornada
  validacion.py        # Validación temporal sin fuga + calibración 1X2 vs baselines

web/                   # Web de visualización en Astro
scripts/
  export_data.py       # Convierte CSV/JSON en módulos TypeScript para la web
  fetch_real_results.py # Obtiene resultados reales de football-data.org
  update_foto_fija.py  # Actualiza métricas de equipos con resultados reales
```

---

## Formato de grupos (Mundial 2026)

El torneo sigue el formato real de 48 equipos: **12 grupos de 4 selecciones (A–L)**, con los dos primeros de cada grupo y los **8 mejores terceros** pasando a una eliminatoria de 32.

Los grupos están definidos en `src/clases_simulacion.py` con el ELO base de cada selección.
