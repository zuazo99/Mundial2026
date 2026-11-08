# Datos de La Liga

0. (Alternativa recomendada en el entorno cloud) `python scripts/fetch_liga_openfootball.py` descarga y limpia
   openfootball/football.json (11 temporadas de Primera y Segunda) y escribe `raw/` + `cleaning_report.json`.
1. `python scripts/download_liga.py` (en local, con red) baja `SP1_<aabb>.csv` de
   football-data.co.uk a `data/liga/raw/` (Primera `SP1_*` y Segunda `SP2_*`; `--no-segunda` para omitir SP2). También sirve copiarlos a mano.
2. `python src/liga/ingest.py` genera:
   - `df_liga.csv` — formato largo (2 filas por partido, ELO propio previo al partido)
   - `pending_<aabb>.csv` — partidos que faltan de la temporada en curso
   - `elo_actual.csv` — ELO actual de cada club

ELO propio (K=20, ventaja de local 65, inicio 1500). Los ascendidos empiezan con el
ELO medio de los descendidos de la temporada anterior. Los nombres de equipo son los
de football-data.

Si hay ficheros `SP2_<aabb>.csv`, los ascendidos se siembran con su ELO en Segunda
(media de Segunda anclada `ELO_SP2_GAP`=100 puntos por debajo de la de Primera, supuesto
ajustable); si un ascendido no aparece en SP2, recibe la media de los descendidos.

3. `python src/liga/model.py` entrena el modelo de clubes y genera `xg_pending_<aabb>.csv`
   (XG esperado de local y visitante de cada partido pendiente) y `results/validation_liga.json`.
4. `python src/liga/simulate.py [n_sims]` simula el resto de la temporada y escribe
   `results/liga_probabilities.json` (campeón, Champions, Europa League, descenso, posiciones).
   Zonas y desempates documentados en la cabecera de `src/liga/simulate.py`.

## Flujo de datos (producción)

```
python scripts/fetch_liga_openfootball.py     # openfootball → data/liga/raw (SP1/SP2 limpios, temporada en curso)
python scripts/build_liga_history.py          # + espejo datasets/football-datasets (1993/94-) → data/liga/raw_hist
python src/liga/ingest.py                     # raw_hist, ELO desde 2016/17 → df_liga.csv, pending_*.csv, elo_actual.csv
python src/liga/model.py && python src/liga/simulate.py && python scripts/export_liga.py
```

`build_liga_history.py` descarga el espejo **fijado a un commit** (`source_pins.json`), verifica el SHA-256 de cada fichero
(falla si cambia, salvo `--accept-changes`), comprueba los alias contra los datos, rechaza pares de nombres parecidos sin
revisar, pone en cuarentena filas de fuente única sospechosas y escribe `history_report.json`. openfootball manda donde existe.

## Categorización de datasets (medida, nov 2026)

| Dataset | Uso | Veredicto | Limpieza necesaria |
|---|---|---|---|
| openfootball/football.json (2016/17-2026/27) | **Temporada en curso** + base de referencia | Imprescindible (única con 2026/27) | `clean_season`: marcador válido, sin fechas futuras, duplicados, alias (10 renombres en 2020/21); faltan 10 (2024/25) y 15 (2025/26) resultados |
| datasets/football-datasets (1993/94-2025/26) | **Histórico** y completar 2024/25 | Útil con controles: 0 desacuerdos en 3.775 partidos comunes con openfootball y 100 % con cache.footballdata (11.944), **pero 15 filas de 2025/26 son 0-0 de relleno** | ISO estricto; 33 alias (31 derivados + `Santander`, `Villareal`); cuarentena de filas de fuente única con ≥ 80 % de 0-0; excepción acotada 1998/99→1999/00; huella SHA-256 y commit fijado |
| …sus columnas de tiros/córners/tarjetas/árbitro | Enriquecimiento | **No usable todavía**: solo desde 2005/06 y no existen para 2026/27 (se predeciría con una variable que falta) | — (pendiente de football-data directo) |
| footballcsv/cache.footballdata (1993/94-2023/24) | **Solo validación cruzada** | Útil como segundo testigo independiente | Nombres abreviados; no se usa como entrada |
| jalapic/engsoccerdata (1928-2024) | Historial entre rivales (futuro) | Aplazado; licencia GPL ≥ 2 a revisar | Nombres antiguos y 27.335 filas sin verificar |
| statsbomb/open-data | Validar xG | Marginal (solo 2015/16 completa) | — |

Decisiones medidas: calentar el ELO desde 1993 **no se adopta** (backtest: el log-loss de campeón empeora +0,010; ver
`scripts/liga_backtest_elo.py`, `results/liga_backtest_elo.csv`); se mantiene la cadena desde 2016/17 (`PRODUCTION_FIRST_SEASON`).

## Fuentes evaluadas (verificadas desde el entorno cloud, nov 2026)

Alcanzables vía `raw.githubusercontent.com` (HTTPS; `github.com` y su API dan 403):

| Fuente | Cobertura La Liga | Contenido | Licencia | Notas |
|---|---|---|---|---|
| [datasets/football-datasets](https://github.com/datasets/football-datasets) (`datasets/la-liga/season-<aabb>.csv`) | 1993/94 → 2025/26, 380 filas por temporada (**15 de 2025/26 son 0-0 de relleno**, ver categorización) | resultado, descanso, árbitro, tiros (HS/AS), a puerta (HST/AST), faltas, córners, tarjetas | PDDL (dominio público) | Espejo de football-data.co.uk. Sin cuotas. **No tiene 2026/27** (a 2026-11-08) |
| [openfootball/football.json](https://github.com/openfootball/football.json) | desde ~2016/17 hasta 2026/27 (en curso) | resultado y descanso | CC0 | La única con la temporada en curso (hasta 2026-09-20); faltan 10 resultados en 2024/25 y 15 en 2025/26 |
| [footballcsv/cache.footballdata](https://github.com/footballcsv/cache.footballdata) | 1993/94 → 2023/24 | resultado y descanso | CC0 | Nombres abreviados de football-data; usada para validación cruzada |
| [jalapic/engsoccerdata](https://github.com/jalapic/engsoccerdata) (`data-raw/spain.csv`) | 1928 → 2024, 27.335 partidos | resultado y descanso | GPL ≥ 2 (licencia del paquete; revisar antes de redistribuir datos) | Historial muy largo entre rivales |
| [statsbomb/open-data](https://github.com/statsbomb/open-data) | 2004/05 → 2020/21, casi todo solo partidos del Barça; 2015/16 completa | eventos y xG por tiro | licencia propia de StatsBomb (atribución) | Útil para validar xG, no como fuente principal |

No alcanzables desde aquí (sin permisos de red): football-data.co.uk (original, con cuotas), clubelo.com, Understat, FBref, football-data.org, API-Football, SportMonks. De pago: Opta.

Seguridad de la cadena de datos: implementada para el espejo (commit fijado en `source_pins.json`, SHA-256 por fichero, fallo ruidoso si cambia). Pendiente para openfootball (hoy se descarga de `master`).
