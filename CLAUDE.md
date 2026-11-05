# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIFA World Cup 2026 prediction system. An XGBoost model trained on historical match data predicts expected goals (XG) per team, which then drive a probabilistic match simulator for the full 48-team tournament.

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

Single-page app (`web/src/pages/index.astro`) with 4 CSS-based tabs:
- **Grupos** — 12 group cards with standings + XG match cards, variant toggle
- **Cuadro** — scrollable bracket with divergence indicators per match
- **Resultado** — champion podium × 3 variants, Shock del torneo, stats
- **Simular** — runs a fresh simulation in-browser using the TS simulation engine

Key libraries in `web/src/lib/`:
- `computeStandings.ts` — group standings + Python-compatible tiebreaker
- `bracketBuilder.ts` — parses flat result rows into bracket structure
- `simulation.ts` — full Poisson simulation engine (port of Python logic)

**Critical note on simulation port**: `get_multiplier()` in Python has dead code (lines 67–105 are overwritten unconditionally). The TS port implements only the executed second block.

Knockout XG uses ELO-based approximation (`eloToXG()`) since XGBoost can't run in the browser.

### Vercel deployment

`vercel.json` at repo root; build command: `cd web && npm run build`.
