"""Simulación Monte Carlo del resto de la temporada de La Liga.

Parte de los resultados ya jugados y del XG esperado de cada partido pendiente
(`xg_pending_<aabb>.csv`, salida de model.py), muestrea los marcadores a 90 minutos
con la corrección de Dixon-Coles y resuelve la clasificación con los desempates de
La Liga. Devuelve, por equipo, probabilidades de campeón, Champions, Europa League
y descenso, y la distribución completa de posiciones.

Supuestos (documentados también en la salida):
  * XG fijo ("xg_frozen"): el estado de los equipos no se actualiza durante la
    temporada, así que no recoge cambios de forma y las probabilidades salen
    algo sobreconfiadas.
  * Zonas por defecto: 4 Champions, puestos 5-6 Europa League, 3 descensos. Las plazas
    europeas cambian cada año (plaza extra, Copa del Rey) y no se modelan.
  * Desempate (reglamento de La Liga): puntos → puntos del enfrentamiento directo →
    diferencia de goles del enfrentamiento directo → diferencia de goles general →
    goles a favor generales. Con 3 o más empatados se hace UNA mini-liga entre todos
    ellos; no se vuelve a aplicar al subgrupo que siga empatado (supuesto).

Uso:  python src/liga/simulate.py [n_sims] [sigma]   # tras ingest.py y model.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import path  # noqa: E402

MAX_GOALS = 10
DEFAULT_ZONES = {"champions": 4, "europa": (5, 6), "relegation": 3}
DEFAULT_N_SIMS = 10_000
DEFAULT_SEED = 2026
STRENGTH_SIGMA_DEFAULT = 0.0     # incertidumbre en la fuerza de los equipos (desv. típica del log-factor)
BATCH = 200                      # simulaciones por lote cuando hay incertidumbre (memoria acotada)
_KS = np.arange(MAX_GOALS + 1)
_LOGFACT = np.cumsum(np.concatenate(([0.0], np.log(np.maximum(_KS[1:], 1)))))


# ─── muestreo ────────────────────────────────────────────────────────────────

def _poisson_pmf(lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(lam, 1e-9)[:, None]
    return np.exp(_KS[None, :] * np.log(lam) - lam - _LOGFACT[None, :])


def sample_scores(xg_h, xg_a, rho: float, n_sims: int, rng: np.random.Generator):
    """Marcadores a 90' de M partidos × n_sims simulaciones (Dixon-Coles). Devuelve (hg, ag) (n_sims, M)."""
    xg_h, xg_a = np.asarray(xg_h, dtype=float), np.asarray(xg_a, dtype=float)
    m = len(xg_h)
    if m == 0:
        return np.zeros((n_sims, 0), dtype=np.int16), np.zeros((n_sims, 0), dtype=np.int16)
    lam, mu = np.maximum(xg_h, 1e-9), np.maximum(xg_a, 1e-9)
    joint = _poisson_pmf(lam)[:, :, None] * _poisson_pmf(mu)[:, None, :]      # (M, 11, 11)
    joint[:, 0, 0] *= np.maximum(0.0, 1.0 - lam * mu * rho)
    joint[:, 1, 0] *= np.maximum(0.0, 1.0 + mu * rho)
    joint[:, 0, 1] *= np.maximum(0.0, 1.0 + lam * rho)
    joint[:, 1, 1] *= np.maximum(0.0, 1.0 - rho)
    flat = np.maximum(joint, 0.0).reshape(m, -1)
    cdf = np.cumsum(flat, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random((n_sims, m))
    idx = np.empty((n_sims, m), dtype=np.int64)
    for j in range(m):
        idx[:, j] = np.minimum(np.searchsorted(cdf[j], u[:, j]), flat.shape[1] - 1)
    return (idx // (MAX_GOALS + 1)).astype(np.int16), (idx % (MAX_GOALS + 1)).astype(np.int16)


def _sample_matrix(lam: np.ndarray, mu: np.ndarray, rho: float, rng: np.random.Generator):
    """Como sample_scores pero con un λ/μ distinto por simulación: lam, mu de forma (B, M)."""
    b, m = lam.shape
    l, u_ = np.maximum(lam, 1e-9).ravel(), np.maximum(mu, 1e-9).ravel()
    joint = _poisson_pmf(l)[:, :, None] * _poisson_pmf(u_)[:, None, :]
    joint[:, 0, 0] *= np.maximum(0.0, 1.0 - l * u_ * rho)
    joint[:, 1, 0] *= np.maximum(0.0, 1.0 + u_ * rho)
    joint[:, 0, 1] *= np.maximum(0.0, 1.0 + l * rho)
    joint[:, 1, 1] *= np.maximum(0.0, 1.0 - rho)
    flat = np.maximum(joint, 0.0).reshape(b * m, -1)
    cdf = np.cumsum(flat, axis=1)
    cdf /= cdf[:, -1:]
    idx = np.minimum((cdf < rng.random(b * m)[:, None]).sum(axis=1), flat.shape[1] - 1)
    return (idx // (MAX_GOALS + 1)).reshape(b, m).astype(np.int16), (idx % (MAX_GOALS + 1)).reshape(b, m).astype(np.int16)


# ─── clasificación y desempates ──────────────────────────────────────────────

def _table(hg: np.ndarray, ag: np.ndarray):
    """(puntos, goles a favor, goles en contra) de una matriz de resultados n×n (diagonal ignorada)."""
    n = hg.shape[0]
    off = ~np.eye(n, dtype=bool)
    hpts = np.where(off, np.where(hg > ag, 3, np.where(hg == ag, 1, 0)), 0)
    apts = np.where(off, np.where(ag > hg, 3, np.where(hg == ag, 1, 0)), 0)
    hgm, agm = np.where(off, hg, 0), np.where(off, ag, 0)
    pts = hpts.sum(axis=1) + apts.sum(axis=0)
    gf = hgm.sum(axis=1) + agm.sum(axis=0)
    ga = agm.sum(axis=1) + hgm.sum(axis=0)
    return pts, gf, ga


def final_order(hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    """Índices de equipo ordenados de 1.º a último con los desempates de La Liga.

    hg[i, j] = goles del local i ante el visitante j; ag[i, j] = goles del visitante j.
    """
    n = hg.shape[0]
    pts, gf, ga = _table(hg, ag)
    gd = gf - ga
    base = sorted(range(n), key=lambda i: (-pts[i], i))
    order, k = [], 0
    while k < n:
        j = k
        while j + 1 < n and pts[base[j + 1]] == pts[base[k]]:
            j += 1
        group = base[k:j + 1]
        if len(group) > 1:
            s = np.array(group)
            h_pts, h_gf, h_ga = _table(hg[np.ix_(s, s)], ag[np.ix_(s, s)])
            h_gd = h_gf - h_ga
            keyed = sorted(range(len(group)), key=lambda r: (-h_pts[r], -h_gd[r], -gd[s[r]], -gf[s[r]], s[r]))
            group = [group[r] for r in keyed]
        order.extend(group)
        k = j + 1
    return np.array(order)


# ─── simulación ──────────────────────────────────────────────────────────────

def _zones(zones):
    z = dict(DEFAULT_ZONES)
    z.update(zones or {})
    z["europa"] = tuple(z["europa"])
    return z


def _validate(teams, played, pending):
    n = len(teams)
    known = set(teams)
    for name, df in (("jugados", played), ("pendientes", pending)):
        bad = (set(df["home"]) | set(df["away"])) - known
        if bad:
            raise ValueError(f"Hay equipo(s) desconocido(s) en los partidos {name}: {sorted(bad)[:3]}")
    p_pairs = set(zip(played["home"], played["away"]))
    q_pairs = list(zip(pending["home"], pending["away"]))
    if p_pairs & set(q_pairs):
        raise ValueError("Hay partidos pendientes que ya constan como jugado(s): datos desfasados, reentrena el modelo.")
    if len(p_pairs) != len(played) or len(set(q_pairs)) != len(q_pairs) or len(played) + len(pending) != n * (n - 1):
        raise ValueError(f"El calendario no cuadra: {len(played)} jugados + {len(pending)} pendientes ≠ {n * (n - 1)}.")


def simulate_season(teams, played: pd.DataFrame, pending_xg: pd.DataFrame, rho: float,
                    n_sims: int, seed: int, zones: dict = None, strength_sigma: float = STRENGTH_SIGMA_DEFAULT) -> dict:
    """Monte Carlo del resto de la temporada. `played`: home, away, hg, ag; `pending_xg`: home, away, xg_home, xg_away.

    strength_sigma > 0: cada simulación sortea una fuerza por equipo s ~ N(0, σ), fija toda la temporada, y
    escala los XG como xg_home·exp(s_h − s_a) y xg_away·exp(s_a − s_h). Con σ = 0 el camino es el de siempre.
    """
    if strength_sigma < 0:
        raise ValueError("strength_sigma debe ser ≥ 0 (sigma negativa).")
    z = _zones(zones)
    teams = list(teams)
    n = len(teams)
    pos_of = {t: i for i, t in enumerate(teams)}
    _validate(teams, played, pending_xg)

    HG = np.zeros((n_sims, n, n), dtype=np.int16)
    AG = np.zeros((n_sims, n, n), dtype=np.int16)
    ph, pa = played["home"].map(pos_of).to_numpy(), played["away"].map(pos_of).to_numpy()
    HG[:, ph, pa] = played["hg"].to_numpy()
    AG[:, ph, pa] = played["ag"].to_numpy()

    rng = np.random.default_rng(seed)
    qh, qa = pending_xg["home"].map(pos_of).to_numpy(), pending_xg["away"].map(pos_of).to_numpy()
    xg_h, xg_a = pending_xg["xg_home"].to_numpy(dtype=float), pending_xg["xg_away"].to_numpy(dtype=float)
    if strength_sigma == 0:
        hg, ag = sample_scores(xg_h, xg_a, rho, n_sims, rng)
        if len(qh):
            HG[:, qh, qa] = hg
            AG[:, qh, qa] = ag
    elif len(qh):
        for start in range(0, n_sims, BATCH):
            b = min(BATCH, n_sims - start)
            s = rng.normal(0.0, strength_sigma, size=(b, n))
            f = np.exp(s[:, qh] - s[:, qa])
            hg, ag = _sample_matrix(xg_h[None, :] * f, xg_a[None, :] / f, rho, rng)
            HG[start:start + b, qh, qa] = hg
            AG[start:start + b, qh, qa] = ag

    positions = np.empty((n_sims, n), dtype=np.int64)       # positions[s, equipo] = puesto (0-based)
    points = np.empty((n_sims, n), dtype=np.int64)
    for s in range(n_sims):
        positions[s, final_order(HG[s], AG[s])] = np.arange(n)
        points[s] = _table(HG[s], AG[s])[0]

    pts_now = _points_played(played, pos_of)

    e_lo, e_hi = z["europa"]
    teams_out = {}
    for t, i in pos_of.items():
        p = positions[:, i]
        dist = np.bincount(p, minlength=n) / n_sims
        teams_out[t] = {
            "points_now": int(pts_now[i]),
            "exp_points": round(float(points[:, i].mean()), 2),
            "exp_position": round(float(p.mean() + 1), 2),
            "p_champion": float((p == 0).mean()),
            "p_champions": float((p < z["champions"]).mean()),
            "p_europa": float(((p >= e_lo - 1) & (p <= e_hi - 1)).mean()),
            "p_relegation": float((p >= n - z["relegation"]).mean()),
            "position_dist": [float(x) for x in dist],
        }
    return {
        "meta": {"n_sims": n_sims, "seed": seed, "rho": float(rho), "xg_frozen": True, "strength_sigma": float(strength_sigma),
                 "zones": {"champions": z["champions"], "europa": list(z["europa"]), "relegation": z["relegation"]},
                 "pending_matches": int(len(pending_xg)), "played_matches": int(len(played)),
                 "tiebreak": "puntos > puntos directo > DG directo > DG general > GF general (mini-liga única)"},
        "teams": teams_out,
    }


def _points_played(played: pd.DataFrame, pos_of: dict) -> np.ndarray:
    """Puntos actuales de cada equipo con solo los partidos jugados."""
    pts = np.zeros(len(pos_of), dtype=int)
    for h, a, gh, ga in played[["home", "away", "hg", "ag"]].itertuples(index=False):
        pts[pos_of[h]] += 3 if gh > ga else (1 if gh == ga else 0)
        pts[pos_of[a]] += 3 if ga > gh else (1 if gh == ga else 0)
    return pts


# ─── entradas y CLI ──────────────────────────────────────────────────────────

_HINT = "Ejecuta antes: python src/liga/ingest.py y python src/liga/model.py"


def load_inputs(data_dir: str, model_dir: str):
    long_path = os.path.join(data_dir, "df_liga.csv")
    pend_files = sorted(glob.glob(os.path.join(data_dir, "pending_*.csv")))
    if not (os.path.exists(long_path) and pend_files):
        raise SystemExit(f"Faltan artefactos en {data_dir}. {_HINT}")
    pending_path = pend_files[-1]
    code = os.path.splitext(os.path.basename(pending_path))[0].split("_", 1)[1]
    xg_path = os.path.join(data_dir, f"xg_pending_{code}.csv")
    if not os.path.exists(xg_path):
        raise SystemExit(f"No hay xg_pending_{code}.csv (modelo desfasado o sin ejecutar). {_HINT}")
    rho_path = os.path.join(model_dir, "rho_liga.json")
    if not os.path.exists(rho_path):
        raise SystemExit(f"Falta rho_liga.json en {model_dir}: no asumo rho=0 sin avisar. {_HINT}")

    long = pd.read_csv(long_path, dtype={"season": str})
    last = max(long["season"].unique())
    if code != last:
        raise SystemExit(f"pending_{code} no corresponde a la última temporada ingerida ({last}). {_HINT}")
    pending = pd.read_csv(pending_path)
    xg = pd.read_csv(xg_path)
    if set(zip(pending["home"], pending["away"])) != set(zip(xg["home"], xg["away"])) or len(pending) != len(xg):
        raise SystemExit(f"Los pares de pending_{code} y xg_pending_{code} no coinciden. {_HINT}")
    cur = long[(long["season"] == last) & (long["is_home"] == 1)]
    played = pd.DataFrame({"home": cur["team"], "away": cur["opponent"],
                           "hg": cur["goals"].astype(int), "ag": cur["goals_conceded"].astype(int)}).reset_index(drop=True)
    rho = float(json.load(open(rho_path, encoding="utf-8"))["rho"])
    teams = sorted(set(played["home"]) | set(played["away"]) | set(xg["home"]) | set(xg["away"]))
    data_as_of = str(pd.to_datetime(cur["date"]).max().date())
    return last, teams, played, xg, rho, data_as_of


def run(data_dir: str, model_dir: str, results_dir: str, n_sims: int = DEFAULT_N_SIMS,
        seed: int = DEFAULT_SEED, strength_sigma: float = STRENGTH_SIGMA_DEFAULT) -> dict:
    last, teams, played, xg, rho, data_as_of = load_inputs(data_dir, model_dir)
    out = simulate_season(teams, played, xg, rho, n_sims, seed, strength_sigma=strength_sigma)
    out["meta"]["season"] = last
    out["meta"]["data_as_of"] = data_as_of      # fecha del último resultado: las probabilidades valen a esa fecha
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "liga_probabilities.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    top = sorted(out["teams"].items(), key=lambda kv: -kv[1]["p_champion"])[:5]
    print(f"   (datos a fecha {data_as_of})")
    print(f"✅ {n_sims} simulaciones de {last} ({out['meta']['pending_matches']} partidos pendientes) → liga_probabilities.json")
    for t, v in top:
        print(f"   {t:<20} campeón {v['p_champion']:.1%}  Champions {v['p_champions']:.1%}  descenso {v['p_relegation']:.1%}")
    return out


if __name__ == "__main__":
    run(path("data", "liga"), path("api", "models"), path("results"),
        n_sims=int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_SIMS,
        strength_sigma=float(sys.argv[2]) if len(sys.argv) > 2 else STRENGTH_SIGMA_DEFAULT)
