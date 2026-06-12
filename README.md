# Mundial 2026 — Predictor con Machine Learning

¿Y si pudiéramos predecir el Mundial 2026 utilizando un modelo de Machine Learning?

Este proyecto entrena un modelo XGBoost sobre datos históricos de partidos internacionales para estimar los **goles esperados (XG)** de cada equipo, y después usa esas predicciones para **simular el torneo completo** de 48 equipos de forma probabilística. Los resultados se visualizan en una web interactiva.

---

## Índice

- [Cómo funciona](#cómo-funciona)
- [Ejecución](#ejecución)
- [Los tres modelos: en qué se basa cada uno](#los-tres-modelos-en-qué-se-basa-cada-uno)
- [Cómo leer los resultados en la web](#cómo-leer-los-resultados-en-la-web)
- [Apartado técnico: cómo se entrena el modelo](#apartado-técnico-cómo-se-entrena-el-modelo)
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

Muestra los **12 grupos (A–L)** con su clasificación final y el resultado predicho de cada partido.

- **El marcador que ves** (p. ej. `España 2 – 1 Uruguay`) es el resultado **más probable** de entre las 30 simulaciones de ese partido, **no una certeza**. El fútbol es aleatorio: el modelo dice "esto es lo que más veces pasa", no "esto pasará seguro".
- **XG (goles esperados):** junto a cada equipo verás su XG. Es una medida de **cuántas ocasiones de gol de calidad** se espera que genere. Un XG de 2,1 significa "con las ocasiones que suele crear este equipo contra este rival, lo normal sería que marcase ~2 goles". Cuanto mayor el XG, más dominio ofensivo espera el modelo.

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
- **Shock del torneo:** el mayor **batacazo**. Es el partido en el que el ganador tenía, según su ELO, la **menor probabilidad de ganar**: la sorpresa más grande de toda la simulación.
- **Camino al título:** el recorrido completo del campeón, ronda a ronda.
- Estadísticas como el **partido con más goles** del torneo.

### 4. Simulador Interactivo

Permite **lanzar una simulación nueva en tu navegador** y ver un torneo distinto cada vez. Como el proceso es probabilístico, **cada ejecución da un resultado diferente** — esa es precisamente la gracia: ver el abanico de futuros posibles, no un único destino.

> Nota técnica: el simulador del navegador reproduce el motor de Poisson del Python, pero para el XG de las eliminatorias usa una **aproximación basada en ELO** (`eloToXG()`), porque el modelo XGBoost no puede ejecutarse en el navegador.

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

### De entrenamiento a predicción

1. **Foto fija:** tras entrenar, se toma el **último estado conocido** (medias móviles, ELO, PCA…) de cada selección antes del Mundial.
2. Con esa foto fija se predice el XG de cada partido del calendario real, dividido en tres jornadas de grupos (J1, J2, J3) por fecha.
3. La simulación convierte ese XG en goles minuto a minuto: `XG/90` es la probabilidad de marcar en cada minuto (proceso de Poisson), modulada por multiplicadores que reaccionan al marcador y al tiempo restante. 30 iteraciones por partido → se escoge el marcador más frecuente. En eliminatorias se añaden prórroga y penaltis.

---

## Limitaciones y cómo se podría mejorar

El modelo es sólido como punto de partida, pero tiene márgenes claros de mejora:

**Limitaciones actuales**
- **Sin datos de jugadores:** no contempla lesiones, sanciones, convocatorias ni el estado de forma de futbolistas concretos. Una baja clave no se refleja.
- **Ajustes de ELO manuales:** los tres escenarios son hipótesis subjetivas, no aprendidas de los datos.
- **Sesgo del "marcador más probable":** quedarse con la moda de 30 iteraciones tiende a favorecer resultados bajos y comunes (1-0, 1-1), infrarrepresentando goleadas.
- **Goles independientes:** el Poisson asume que los goles de cada equipo son independientes, sin correlación entre ataques (lo que infravalora empates y resultados ajustados).
- **Sin calibración externa:** no se contrasta con cuotas de casas de apuestas ni con probabilidades de mercado.

**Posibles mejoras**
- **Reactivar la búsqueda de hiperparámetros** en cada reentrenamiento, en lugar de usar valores fijos, y reportar métricas de validación temporal.
- **Incorporar datos de plantilla:** disponibilidad de titulares, minutos recientes, valor de mercado o un ELO por jugador.
- **Modelo de goles correlacionado:** usar un Poisson bivariante o el ajuste de **Dixon-Coles** para capturar la dependencia entre los goles de ambos equipos.
- **Trabajar con la distribución completa**, no solo con el marcador modal: mostrar probabilidades de victoria/empate/derrota y márgenes de incertidumbre.
- **Calibrar contra el mercado** (cuotas) para validar y corregir sesgos sistemáticos.
- **Más iteraciones por partido** para estimaciones más estables, y **automatizar la actualización del ELO** en lugar de mantenerlo hardcodeado.
- **Features de tiro/xG real** (datos de disparos) en vez de derivar el XG solo de goles históricos.

---

## Estructura de datos

```
data/
  models_csv/          # Históricos de entrenamiento (un CSV por variante, ~3M filas)
  ai_models/           # XG predichos para cada jornada de grupos (salida de xg_preds.py)
  mejores_terceros.csv # Cuadro de emparejamientos para los mejores terceros

results/               # Predicciones finales del torneo (salida de simulacion.py)
  predictions_misterclaude.csv
  predictions_gemaldini.csv
  predictions_dav_gpo.csv
  predictions_definitive.csv

web/                   # Web de visualización en Astro
scripts/export_data.py # Convierte los CSV en módulos TypeScript para la web
```

---

## Formato de grupos (Mundial 2026)

El torneo sigue el formato real de 48 equipos: **12 grupos de 4 selecciones (A–L)**, con los dos primeros de cada grupo y los **8 mejores terceros** pasando a una eliminatoria de 32.

Los grupos están definidos en `src/clases_simulacion.py` con el ELO base de cada selección.
