# Mundial 2026 — Predictor con Machine Learning

¿Y si pudiéramos predecir el Mundial 2026 utilizando un modelo de Machine Learning?

Este proyecto entrena un modelo XGBoost sobre datos históricos de partidos internacionales para estimar los goles esperados (XG) de cada equipo, y después usa esas predicciones para simular el torneo completo de 48 equipos de forma probabilística.

---

## Cómo funciona

El sistema tiene dos fases independientes:

### Fase 1 — Entrenamiento del modelo (`src/xg_preds.py`)

1. Carga el histórico de partidos desde 2016 hasta la fecha del Mundial (`data/models_csv/df_<variante>.csv`).
2. Calcula features por equipo: medias móviles de goles a 5 y 15 partidos, ELO medio rival, PCA de estilo de juego (2 componentes), confederación, e importancia del torneo (1–5, siendo el Mundial = 5).
3. Aplica un peso temporal exponencial que da más importancia a los partidos recientes (semivida ≈ 1.277 días, referencia 1 junio 2024).
4. Entrena un regresor Tweedie con XGBoost usando validación temporal (`TimeSeriesSplit`).
5. Genera una "foto fija" con las métricas del último partido de cada selección antes del Mundial.
6. Exporta predicciones de XG para cada jornada de la fase de grupos a `data/ai_models/xg_preds_J{1,2,3}_<variante>.csv`.

### Fase 2 — Simulación del torneo (`src/simulacion.py` + `src/clases_simulacion.py`)

1. Lee los CSVs de XG generados en la Fase 1.
2. Simula cada partido minuto a minuto usando el XG/90 como tasa de Poisson, con multiplicadores dinámicos según el marcador, el minuto y la fortaleza relativa de cada equipo.
3. Cada partido se itera 30 veces para obtener una distribución de resultados y extraer el más probable.
4. Los partidos de eliminatorias incluyen prórroga y penaltis (75% de éxito por lanzamiento).
5. La clasificación de los mejores 8 terceros sigue las pautas de `data/mejores_terceros.csv`.
6. Exporta los resultados completos a `results/predictions_<variante>.csv`.

---

## Ejecución

```bash
# Dependencias (instalar manualmente)
pip install pandas numpy scikit-learn xgboost

# Reentrenar el modelo y generar predicciones de fase de grupos
python src/xg_preds.py

# Simular el torneo completo (genera los CSVs de resultados)
python src/simulacion.py
```

Los dos scripts deben ejecutarse desde la raíz del repositorio.

---

## Variantes del modelo

Se simulan tres escenarios con distintos ajustes de ELO sobre las puntuaciones base:

| Variante | Descripción |
|---|---|
| **misterclaude** | Puntuaciones ELO sin ajustes (salida pura del modelo) |
| **gemaldini** | +200 ELO a Francia, España, Portugal, Inglaterra y Noruega |
| **dav_gpo** | +150–200 ELO a Argentina, Colombia, Ecuador, Paraguay, Uruguay y Brasil |

Los ajustes reflejan distintas hipótesis sobre qué bloque continental llegará más fuerte al torneo.

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
```

---

## Formato de grupos (Mundial 2026)

El torneo sigue el formato real de 48 equipos: 12 grupos de 4 selecciones (A–L), con los dos primeros de cada grupo y los 8 mejores terceros pasando a una eliminatoria de 32.

Los grupos están hardcodeados en `src/clases_simulacion.py` con los ELO base de cada selección.
