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

## Fuentes evaluadas (verificadas desde el entorno cloud, nov 2026)

Alcanzables vía `raw.githubusercontent.com` (HTTPS; `github.com` y su API dan 403):

| Fuente | Cobertura La Liga | Contenido | Licencia | Notas |
|---|---|---|---|---|
| [datasets/football-datasets](https://github.com/datasets/football-datasets) (`datasets/la-liga/season-<aabb>.csv`) | 1993/94 → 2025/26, 380/380 partidos con resultado por temporada | resultado, descanso, árbitro, tiros (HS/AS), a puerta (HST/AST), faltas, córners, tarjetas | PDDL (dominio público) | Espejo de football-data.co.uk. Sin cuotas. **No tiene 2026/27** (a 2026-11-08) |
| [openfootball/football.json](https://github.com/openfootball/football.json) | desde ~2016/17 hasta 2026/27 (en curso) | resultado y descanso | CC0 | La única con la temporada en curso (hasta 2026-09-20); faltan 10 resultados en 2024/25 y 15 en 2025/26 |
| [footballcsv/cache.footballdata](https://github.com/footballcsv/cache.footballdata) | 1993/94 → 2023/24 | resultado y descanso | CC0 | Nombres abreviados de football-data; usada para validación cruzada |
| [jalapic/engsoccerdata](https://github.com/jalapic/engsoccerdata) (`data-raw/spain.csv`) | 1928 → 2024, 27.335 partidos | resultado y descanso | GPL ≥ 2 (licencia del paquete; revisar antes de redistribuir datos) | Historial muy largo entre rivales |
| [statsbomb/open-data](https://github.com/statsbomb/open-data) | 2004/05 → 2020/21, casi todo solo partidos del Barça; 2015/16 completa | eventos y xG por tiro | licencia propia de StatsBomb (atribución) | Útil para validar xG, no como fuente principal |

No alcanzables desde aquí (sin permisos de red): football-data.co.uk (original, con cuotas), clubelo.com, Understat, FBref, football-data.org, API-Football, SportMonks. De pago: Opta.

Seguridad de la cadena de datos (pendiente de implementar): fijar las descargas a un commit concreto en lugar de `master`, guardar el SHA-256 de cada fichero en `cleaning_report.json`, y mantener la validación cruzada entre fuentes independientes.
