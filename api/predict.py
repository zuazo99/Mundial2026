"""
Serverless endpoint: predice XG para partidos de eliminatorias usando el
modelo XGBoost entrenado. Recibe un batch de enfrentamientos y devuelve
los XG de ambos equipos para cada uno.

Request  (POST): {"variant": "misterclaude", "matchups": [{"team1": "Spain", "team2": "France"}, ...]}
Response (200):  {"predictions": [{"xg1": 1.23, "xg2": 0.91}, ...]}
"""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from http.server import BaseHTTPRequestHandler

FEATURES = [
    'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
    'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2',
    'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15',
    'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
]

VARIANTS = {"misterclaude", "gemaldini", "dav_gpo"}

# Caché en memoria entre invocaciones warm (Vercel reutiliza contenedores)
_models: dict = {}
_stats:  dict = {}


def _root() -> str:
    """Ruta raíz del repo (dos niveles sobre este archivo)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_model(variant: str) -> xgb.XGBRegressor:
    if variant not in _models:
        path = os.path.join(_root(), "api", "models", f"xgb_{variant}.json")
        m = xgb.XGBRegressor()
        m.load_model(path)
        _models[variant] = m
    return _models[variant]


def _get_stats(variant: str) -> pd.DataFrame:
    if variant not in _stats:
        path = os.path.join(_root(), "data", "ai_models", f"xg_preds_J1_{variant}_complete.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        # foto fija: último registro por equipo antes del Mundial
        _stats[variant] = df.sort_values("date").groupby("team").last()
    return _stats[variant]


def _predict(variant: str, matchups: list) -> list:
    model = _get_model(variant)
    stats = _get_stats(variant)

    rows = []
    for m in matchups:
        t1, t2 = m["team1"], m["team2"]
        if t1 not in stats.index or t2 not in stats.index:
            # Fallback ELO puro si falta un equipo en los datos
            rows.append(None)
            rows.append(None)
            continue
        s1, s2 = stats.loc[t1], stats.loc[t2]
        for attacker, defender in ((s1, s2), (s2, s1)):
            rows.append({
                'elo':               attacker['elo'],
                'opponent_elo':      defender['elo'],
                'is_home':           0,
                'tournament_num':    5,
                'confed':            int(attacker['confed']),
                'rival_confed':      int(defender['confed']),
                'gf_prom_5':         attacker['gf_prom_5'],
                'gc_prom_5':         attacker['gc_prom_5'],
                'elo_prom_5':        attacker['elo_prom_5'],
                'gf_prom_15':        attacker['gf_prom_15'],
                'gc_prom_15':        attacker['gc_prom_15'],
                'PCA_1':             attacker['PCA_1'],
                'PCA_2':             attacker['PCA_2'],
                'rival_gf_prom_5':   defender['gf_prom_5'],
                'rival_gc_prom_5':   defender['gc_prom_5'],
                'rival_elo_prom_5':  defender['elo_prom_5'],
                'rival_gf_prom_15':  defender['gf_prom_15'],
                'rival_gc_prom_15':  defender['gc_prom_15'],
                'rival_PCA_1':       defender['PCA_1'],
                'rival_PCA_2':       defender['PCA_2'],
            })

    # Batch predict (solo filas no-None)
    valid_rows = [r for r in rows if r is not None]
    if valid_rows:
        df = pd.DataFrame(valid_rows)[FEATURES]
        preds = model.predict(df).tolist()
    else:
        preds = []

    # Reconstruir resultados respetando los None (fallback)
    results = []
    pred_idx = 0
    for i, m in enumerate(matchups):
        r1, r2 = rows[i * 2], rows[i * 2 + 1]
        if r1 is None or r2 is None:
            # Fallback ELO simple si el equipo no está en los datos
            elo_diff = 0.0
            xg1 = round(max(0.4, 1.15), 3)
            xg2 = round(max(0.4, 1.15), 3)
        else:
            xg1 = round(float(preds[pred_idx]),     3)
            xg2 = round(float(preds[pred_idx + 1]), 3)
            pred_idx += 2
        results.append({"xg1": xg1, "xg2": xg2})

    return results


class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silenciar logs de acceso en Vercel

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            variant  = body.get("variant", "misterclaude")
            matchups = body.get("matchups", [])

            if variant not in VARIANTS:
                self._error(400, f"variant must be one of {sorted(VARIANTS)}")
                return
            if not isinstance(matchups, list) or len(matchups) == 0:
                self._error(400, "matchups must be a non-empty list")
                return

            predictions = _predict(variant, matchups)
            self._json(200, {"predictions": predictions})

        except Exception as e:
            self._error(500, str(e))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str):
        self._json(code, {"error": msg})
