# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIFA World Cup 2026 prediction system. An XGBoost model trained on historical match data predicts expected goals (XG) per team, which then drive a probabilistic match simulator for the full 48-team tournament.

> **Antes de empezar una tarea, leer `docs/registro_errores/README.md`** (errores ya cometidos y reglas para no repetirlos) y añadir ahí cada error nuevo.

## Running the Project

```bash
# Run the full tournament simulation (generates results/ CSVs)
python src/simulacion.py

# Retrain the XG model and regenerate group-stage predictions (data/ai_models/ CSVs)
python src/xg_preds.py
```

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest
python -m pytest -q tests             # characterization tests (~1s)
```

No build step, no linter configuration. All paths are resolved from the repo root via `src/config.py` (`path(...)`), so scripts can be run from any directory. `python src/xg_preds.py [variant ...] [--tuned]` retrains and saves `api/models/` + `data/ai_models/`; `Tournament` loads the saved model (`load_or_train_model`) instead of retraining on every instantiation.

The daily GitHub Action (`daily_update.yml`) is **paused** (only `workflow_dispatch`); re-enable its `schedule` block to resume updates.

## Architecture

### Pipeline: two distinct phases

**Phase 1 — Model training (`src/xg_preds.py`)**
- Loads historical match data from `data/models_csv/df_*.csv` (one CSV per model variant, ~19.7k rows each (two per match), from 2016 to present)
- Engineers features: 5- and 15-game rolling averages for goals/goals-against/ELO, PCA (2 components from 4 play-style variables), confederation encoding, tournament importance weights (1–5; World Cup = 5), and exponential time-decay (λ = ln(2)/1277 days)
- Trains an XGBoost Tweedie regressor (variance power 1.3) via `RandomizedSearchCV` with `TimeSeriesSplit`
- Produces per-round XG prediction CSVs for group stage: `data/ai_models/xg_preds_J{1,2,3}_<variant>.csv`

**Phase 2 — Tournament simulation (`src/simulacion.py` + `src/clases_simulacion.py`)**
- Reads the XG CSVs generated in Phase 1
- Instantiates a `Tournament` for each model variant and simulates the full 2026 World Cup
- Each match runs `num_iterations=30` probabilistic simulations using XG/90 as Poisson rates, with in-match multipliers that adapt to game state, minute, and team strength
- Knockout matches include extra time and a penalty shootout (75% success rate per kick)
- Best-8 third-place qualification reads `data/mejores_terceros.csv`
- Results written to `results/predictions_<variant>.csv`

### Core classes (`src/clases_simulacion.py`)

| Class | Role |
|---|---|
| `Team` | Holds team state: ELO, group, points, goal difference, rolling averages, PCA components |
| `Match` | Simulates one match; minute-by-minute scoring with adaptive attack multipliers |
| `Group` | Manages a 4-team group, simulates 6 matches, applies tie-breaking rules |
| `Knockouts` | Drives all knockout rounds; calls `predict_xg_matches()` to score each pairing using the trained model |
| `Tournament` | Top-level orchestrator: builds 12 groups, runs group stage, selects best thirds, runs knockouts, exports CSV |

### Three model variants

Each variant applies ELO adjustments on top of the base ratings before simulation:

- **misterclaude** — no adjustments (pure model output)
- **gemaldini** — +200 ELO to France, Spain, Portugal, England, Norway
- **dav_gpo** — +150–200 ELO to Argentina, Colombia, Ecuador, Paraguay, Uruguay, Brazil

Variant-specific ELO overrides live in `src/simulacion.py`; base ELO values for all 48 teams are hardcoded in `src/clases_simulacion.py` (lines ~723–818).

### Data layout

```
data/models_csv/       # Training datasets (one per variant)
data/ai_models/        # XG predictions output by xg_preds.py (inputs to simulation)
data/mejores_terceros.csv  # Bracket pairings for best 3rd-place teams
results/               # Final tournament predictions (output of simulacion.py)
```

### Key constants to know

- Feature vector (20 features), shared constants and paths: `src/config.py` (`FEATURES`, `VARIANTS`, `WC_START`, `path()`) — single source of truth for training, simulation, Monte Carlo and export
- Training/World Cup split dates: `WC_START` / `WC_J2_START` / `WC_J3_START` in `src/config.py`
- Group stage uses the real 2026 draw (12 groups A–L, 48 teams), hardcoded in `clases_simulacion.py`
- Results CSV has 104 data rows: rows 0–71 = groups (6 per group × 12), rows 72–87 = R32, 88–95 = S16, 96–99 = E8, 100–101 = Semis, 102 = 3rd place, 103 = Final

## Web Frontend (`web/`)

Static Astro app that visualizes the simulation outputs. Deploy target: Vercel.

### Running the web app

```bash
# From repo root — regenerate all XG + result CSVs and convert to TS
python3 scripts/export_data.py   # requires the Python sim to have run first

# Start dev server
cd web && npm install && npm run dev

# Production build
cd web && npm run build
```

### Web data pipeline

1. Python simulation outputs CSV files (`results/`, `data/ai_models/`)
2. `scripts/export_data.py` converts them to typed TS modules in `web/src/data/`
3. Astro imports those modules at build time — all data is static, no runtime API calls

The generated files (`web/src/data/predictions.ts`, `xg.ts`, `teamStats.ts`, `mejoresTerceros.ts`) are committed to the repo. Re-run `scripts/export_data.py` after any Python simulation run.

### Web architecture

Single-page app (`web/src/pages/index.astro`) with CSS-based tabs (Grupos, Cuadro, Cuadro real, Resultado, Simular, Porra and **La Liga**; the first four below):
- **Grupos** — 12 group cards with standings + XG match cards, variant toggle
- **Cuadro** — scrollable bracket with divergence indicators per match
- **Resultado** — champion podium × 3 variants, Shock del torneo, stats
- **Simular** — runs a fresh simulation in-browser using the TS simulation engine
- **La Liga** — `web/src/components/LigaTab.astro` reading `web/src/data/liga.ts` (generated by `scripts/export_liga.py` from `results/liga_probabilities.json`): freshness notice (`dataAsOf`), probability table by zone, 20×20 position heatmap and a "Cómo leerlo" box with the measured caveats. It hides the Mundial variant toggle and footer. It does **not** update itself: after new data run `fetch_liga_openfootball.py` → `ingest.py` → `model.py` → `simulate.py` → `export_liga.py` and commit (a test checks that the committed `liga.ts` matches the regeneration).

Key libraries in `web/src/lib/`:
- `computeStandings.ts` — group standings + Python-compatible tiebreaker
- `bracketBuilder.ts` — parses flat result rows into bracket structure
- `simulation.ts` — full Poisson simulation engine (port of Python logic)

**Critical note on simulation port**: `get_multiplier()` in Python has dead code (lines 67–105 are overwritten unconditionally). The TS port implements only the executed second block.

Knockout XG uses ELO-based approximation (`eloToXG()`) since XGBoost can't run in the browser.

### Vercel deployment

`vercel.json` at repo root; build command: `cd web && npm run build`.

## La Liga (en construcción)

Fase 1 — ingesta (`src/liga/ingest.py`, `scripts/download_liga.py`, ver `data/liga/README.md`): CSV de football-data → ELO propio → formato largo compatible con `xg_preds.engineer_features`. Los ascendidos se siembran con su ELO en Segunda (`SP2_*.csv`, `ELO_SP2_GAP`=100, supuesto sin calibrar); sin datos de SP2 reciben la media de los descendidos. La primera temporada arranca en frío (todos a 1500): excluirla o pesarla menos al entrenar. 

Fase 2 — modelo (`src/liga/model.py`, `python src/liga/model.py` tras la ingesta): reutiliza el pipeline de `xg_preds` con decaimiento anclado a la última fecha (`HALF_LIFE_DAYS`=730, elección sin calibrar), sin la primera temporada en el ajuste (ELO en frío), `is_home` real y `confed`/`tournament_num` constantes. Un único modelo `api/models/xgb_liga.json` (no está en `VARIANTS`). Predice XG de los partidos pendientes con el estado de cada equipo congelado tras su último partido (`data/liga/xg_pending_<aabb>.csv`) y valida con hold-out temporal frente a baselines (`results/validation_liga.json`). `train_liga` rechaza datos donde un equipo juega dos veces el mismo día (el cruce con el rival es por fecha). Los partidos de SP2 no están en el formato largo, así que las ventanas de un recién ascendido salen de su última etapa en Primera o se rellenan con la media. Probado solo con una liga sintética (`tests/test_liga_model.py`); falta validar con datos reales.

Fase 3 — simulación (`src/liga/simulate.py`, `python src/liga/simulate.py [n_sims]` tras ingesta y modelo): Monte Carlo vectorizado del resto de la temporada sobre `xg_pending_<aabb>.csv` con Dixon-Coles y el `rho` de `api/models/rho_liga.json` (aborta si falta). Desempates de La Liga (puntos → puntos y DG del enfrentamiento directo → DG y GF generales; con 3+ empatados una sola mini-liga, supuesto). Zonas por defecto (supuesto, parametrizables): 4 Champions, puestos 5-6 Europa League, 3 descensos. Salida: `results/liga_probabilities.json` con P(campeón/Champions/Europa/descenso), puntos y posición esperados y distribución de posiciones, más metadatos (`xg_frozen: true`). Valida que `pending`, `xg_pending` y la última temporada ingerida coincidan. Limitación: el XG es fijo (no recoge cambios de forma durante la temporada). El backtest (abajo) no encontró sobreconfianza medible. Probado solo con la liga sintética de `tests/liga_synth.py`.

Mejoras opcionales, **medidas y no adoptadas** (desactivadas por defecto; resultados en la sección de calidad): (1) `model.SHRINK_DEFAULT` — encoge las ventanas de equipos con poca historia hacia un prior lineal en el ELO (`fit_priors`/`apply_shrink`, modos `fill`/`const`); (2) `simulate.STRENGTH_SIGMA_DEFAULT` — incertidumbre en la fuerza de los equipos: cada simulación sortea `s ~ N(0, σ)` por equipo y escala los XG con `exp(s_h − s_a)`; con σ = 0 la salida es idéntica byte a byte a la anterior (test con hash de referencia). `src/liga/evaluate.py` puntúa predicciones con RPS de posiciones y log-loss de campeón/Champions/descenso (la cobertura se reporta, no se usa para decidir).

Pendiente: validar con datos reales y, opcionalmente, actualizar el estado (p. ej. solo el ELO) jornada a jornada dentro de la simulación.

### Fuentes de datos de La Liga (openfootball, abiertas)

football-data.co.uk sigue bloqueado en el entorno cloud, pero `raw.githubusercontent.com` es alcanzable. `python scripts/fetch_liga_openfootball.py` descarga [openfootball/football.json](https://github.com/openfootball/football.json) (CC0), lo limpia (`src/liga/sources.py`) y escribe `data/liga/raw/SP1_/SP2_<aabb>.csv` + `data/liga/cleaning_report.json` (caché del JSON original en `data/liga/source/`, ignorada por git). Limpieza: solo partidos con `ft` válido (0-12 goles, `ht ≤ ft`), sin resultados con fecha futura, errores ante duplicados / auto-partidos / mismo equipo el mismo día, alias de nombres (la fuente renombró 10 clubes en 2020/21) y guarda en la ingesta (`check_transitions`: >3 entradas en Primera = posible renombre). Validación cruzada con [footballcsv/cache.footballdata](https://github.com/footballcsv/cache.footballdata): 100 % de acuerdo de marcadores en 2016/17-2023/24 (3.040 partidos).

Limitaciones de la fuente: la temporada en curso llega solo hasta `data_as_of` (hoy 2026-09-20, ~6 semanas de retraso; `openfootball/espana` en `.txt` tiene los mismos resultados); faltan resultados en 2024/25 (10) y 2025/26 (15); Segunda ausente en 2021/22-2023/24 (los ascendidos de 2022/23-2024/25 se siembran con la media de los descendidos) y Segunda 2025/26 solo al 25 %. Otras ligas existen en la misma fuente pero exigirían una feature de liga y un ELO entre ligas. Para football-data directo: permitir `www.football-data.co.uk` en los ajustes del entorno (desde escritorio) y usar `scripts/download_liga.py`.

### Histórico fusionado y decisiones de fuente (nov 2026)

`python scripts/build_liga_history.py` fusiona el espejo [datasets/football-datasets](https://github.com/datasets/football-datasets) (PDDL, 1993/94-2025/26, **fijado a un commit con SHA-256 por fichero**) con openfootball y escribe `data/liga/raw_hist` + `history_report.json`; `ingest.py` lee ahora `raw_hist` con la cadena de ELO desde 2016/17. Hallazgos: 15 resultados de 2025/26 del espejo son 0-0 de relleno (en cuarentena); 10 de 2024/25 son reales y se incorporan (2024/25 queda completa); `Villareal` (errata) y `Santander` necesitan alias manual; no tiene Segunda ni la temporada en curso. Los códigos de temporada son texto: usar siempre `ingest.season_key` para ordenar/comparar (con `"9394"` el orden alfabético falla). Calentar el ELO desde 1993 se midió y **no se adopta** (ver `data/liga/README.md`). Efecto de incorporar las 10 filas en las cifras de la temporada actual: Barça 92,8 % → 94,6 % de campeón, Racing 87,5 % → 93,1 % de descenso (las cifras citadas más abajo son de la versión anterior de los datos): son números frágiles.

### Calidad y plausibilidad de las probabilidades (primera ejecución con datos reales)

- Hold-out 2025/26: devianza de Poisson 0,90 (modelo) frente a 0,974 (media local/visitante) y 1,012 (constante). Calibración por tramos de XG sobre 4 hold-outs: sin sobreestimación (infraestima algo los XG altos). Tasa de empates simulada 23,3 % frente a 25,2 % real; `rho` real −0,017.
- Backtest de temporada (`scripts/liga_backtest.py`, ver más abajo el unificado): se corta cada temporada tras la jornada 7 y se pasa ingesta → modelo → simulación solo con datos hasta ese punto. En 2020/21-2023/24 el intervalo central del 80 % de `position_dist` contiene la posición final real en el 87,5 % de los equipos; el campeón real recibió P = 0,34 / 0,51 / 0,60 / 0,27 y los favoritos (P media 0,47) ganaron 2 de 4. Ninguna probabilidad de campeón de ese backtest superó 0,60: valores como el 92,8 % del Barça (21/21 puntos, +6 sobre el Madrid) quedan fuera del rango validado y deben leerse como cota alta.
- Backtest unificado (`python scripts/liga_backtest.py` → `results/liga_backtest.csv`; temporadas 2019/20-2023/24, cortes en las jornadas 7/14/21, 2.000 simulaciones; selección con 2019/20-2022/23 y confirmación con 2023/24):
  - **σ de fuerza (mejora 2): no se adopta.** Con σ = 0 salen los mejores valores de selección (RPS 0,0696; log-loss de campeón 0,0600, de Champions 0,1084 y de descenso 0,2373); con σ = 0,10 son 0,0706 / 0,0619 / 0,1192 / 0,2390. La cobertura del 80 % ya es 0,90 con σ = 0 (sube a 0,97 con σ = 0,20) y los favoritos ganaron 10 de 12 veces con probabilidad media predicha de 0,62: no se encontró sobreconfianza (los 12 cortes son 4 temporadas × 3 jornadas, no muestras independientes; el 10/12 de favoritos equivale en la práctica a ~4 temporadas). Ojo: la mayor P(campeón) del favorito en la jornada 7 fue 0,59 (0,72 en la 14 y 0,86 en la 21), así que "sin sobreconfianza en promedio" no valida los extremos como el 92,8 % del Barça. En la confirmación (2023/24) el RPS mejora un poco con σ = 0,10-0,15 pero los log-loss empeoran. Efecto sobre la temporada actual si se activara: Barça 92,8 % → 83,4 % (σ = 0,10) y 66,0 % (σ = 0,20); Racing 87,5 % → 77,4 % / 59,0 %.
  - **Encogimiento de ventanas `fill` (mejora 1): no se adopta.** En el hold-out de devianza mejora ligeramente (1,0838 frente a 1,0848 en 2022/23-2024/25 y 0,896 frente a 0,900 en 2025/26) y en las filas afectadas (solo 15 filas por temporada, un equipo), pero en el backtest es peor en las cuatro puntuaciones de selección (RPS 0,0718 frente a 0,0696; log-loss de campeón 0,0623 frente a 0,0600; de Champions 0,1125 frente a 0,1084; de descenso 0,2538 frente a 0,2373) y mixto en la confirmación. Las variantes `const` mejoran algo la devianza total pero empeoran las filas afectadas. Efecto sobre la temporada actual si se activara `fill`: Racing 87,5 % → 25,1 % de descenso, Barça 92,8 % → 95,6 %.
- Equipos con poca historia: Racing de Santander solo tiene 7 partidos en los datos (todos de 2026/27), así que sus ventanas de 5/15 partidos son ruido de muestra pequeña (encajó 3,6 goles/partido en los últimos 5 y suma 7 puntos; el modelo lo ve como peor defensa y lo da 87 % de descenso). Málaga y Deportivo tienen historia solo de 2016/17-2017/18: su ventana de 15 partidos mezcla 7 partidos actuales con 8 de hace ocho años. El encogimiento hacia el ELO se probó y no compensó (ver arriba); el caso de Racing queda como incertidumbre conocida (87,5 % con el modelo actual, 25 % con `fill`).
- Frescura: las probabilidades valen a `data_as_of` (2026-09-20), no a la fecha de ejecución. Las 7 primeras jornadas de 2026/27 promedian 3,28 goles/partido (2,81 en 2025/26) y el estado congelado arrastra parte de esa subida (XG total medio de los pendientes 2,93).

### Pendiente (petición del dueño): usar las APIs directamente

Lo preferido es tirar de las APIs/fuentes originales en cuanto el dueño habilite los hosts del entorno "predicción mundial" (solo se puede hacer desde el navegador de escritorio, no desde el móvil): `www.football-data.co.uk` (resultados SP1/SP2, con `scripts/download_liga.py`) y `api.clubelo.com` (ELO de clubes; hoy se usa ELO propio). Mientras tanto se usa openfootball vía `raw.githubusercontent.com`.
