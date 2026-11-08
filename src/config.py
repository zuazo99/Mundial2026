"""Configuración compartida: rutas, variantes, features y fechas de corte.

Única fuente de verdad para lo que antes estaba duplicado en xg_preds,
clases_simulacion, monte_carlo, validacion y scripts/.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts: str) -> str:
    """Ruta absoluta dentro del repo, independiente del directorio de ejecución."""
    return os.path.join(ROOT, *parts)


VARIANTS = ["misterclaude", "gemaldini", "dav_gpo"]

# Vector de features del modelo (el orden importa: coincide con el entrenamiento).
FEATURES = [
    'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
    'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2',
    'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15',
    'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2',
]
TARGET = "goals"

# Fechas de corte del Mundial 2026 (entrenamiento < WC_START <= predicción).
WC_START = "2026-06-11"
WC_J2_START = "2026-06-18"
WC_J3_START = "2026-06-24"

# Selecciones anfitrionas del Mundial 2026 (únicas con is_home=1 en eliminatorias).
WC_HOSTS = ("United States", "Canada", "Mexico")

# Referencia y vida media (días) del peso temporal exponencial.
DECAY_REF_DATE = "2024-06-01"
DECAY_HALF_LIFE_DAYS = 1277
