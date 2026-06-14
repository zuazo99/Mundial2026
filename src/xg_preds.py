import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


def calculate_window(series: pd.Series, window):
    return series.rolling(window=window, min_periods=1).mean().shift(1)

def train_model(name):
    # =====================================================================
    # 1. CONFIGURACIÓN BASE Y DICCIONARIOS
    # =====================================================================
    confed_dictionary = {
        'CONMEBOL': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela'],
        'CONCACAF': ['USA', 'United States', 'Mexico', 'Canada', 'Costa Rica', 'Jamaica', 'Panama', 'Honduras', 'El Salvador', 'Haiti', 'Trinidad and Tobago', 'Guatemala', 'Cuba', 'Curaçao', 'Martinique', 'Guadeloupe', 'Suriname'],
        'UEFA': ['Spain', 'France', 'Germany', 'England', 'Italy', 'Netherlands', 'Portugal', 'Belgium', 'Croatia', 'Denmark', 'Switzerland', 'Serbia', 'Poland', 'Sweden', 'Wales', 'Scotland', 'Czech Republic', 'Austria', 'Hungary', 'Ukraine', 'Turkey', 'Russia', 'Norway', 'Republic of Ireland', 'Northern Ireland', 'Slovakia', 'Romania', 'Greece', 'Bosnia and Herzegovina', 'Finland', 'Iceland', 'Albania', 'Slovenia', 'Montenegro', 'North Macedonia', 'Georgia', 'Israel', 'Bulgaria', 'Armenia', 'Luxembourg', 'Cyprus', 'Kosovo', 'Estonia', 'Lithuania', 'Latvia', 'Moldova', 'Belarus', 'Malta', 'Liechtenstein', 'Andorra', 'San Marino', 'Gibraltar', 'Kazakhstan', 'Faroe Islands', 'Azerbaijan'],
        'CAF': ['Senegal', 'Morocco', 'Nigeria', 'Egypt', 'Ivory Coast', 'Cameroon', 'Ghana', 'Algeria', 'Mali', 'Tunisia', 'Burkina Faso', 'South Africa', 'DR Congo', 'Guinea', 'Cape Verde', 'Zambia', 'Gabon', 'Uganda', 'Equatorial Guinea', 'Gambia', 'Angola', 'Mauritania', 'Namibia', 'Benin', 'Mozambique', 'Togo', 'Tanzania', 'Zimbabwe', 'Malawi', 'Kenya', 'Congo', 'Rwanda', 'Madagascar', 'Central African Republic', 'Sudan', 'Sierra Leone'],
        'AFC': ['Japan', 'Iran', 'South Korea', 'Australia', 'Saudi Arabia', 'Qatar', 'Iraq', 'United Arab Emirates', 'Oman', 'Uzbekistan', 'China', 'China PR', 'Jordan', 'Bahrain', 'Syria', 'Vietnam', 'Palestine', 'Kyrgyzstan', 'India', 'Lebanon', 'Tajikistan', 'Thailand', 'North Korea', 'Philippines', 'Malaysia', 'Kuwait', 'Turkmenistan', 'Hong Kong', 'Indonesia', 'Yemen', 'Afghanistan', 'Singapore', 'Myanmar', 'Maldives', 'Nepal', 'Cambodia'],
        'OFC': ['New Zealand', 'Solomon Islands', 'Fiji', 'New Caledonia', 'Tahiti', 'Vanuatu', 'Papua New Guinea', 'Samoa', 'Tonga', 'Cook Islands']
    }

    # =====================================================================
    # 2. CARGA DE DATOS Y SEPARACIÓN TEMPORAL
    # =====================================================================
    print("1. Cargando datos...")
    df = pd.read_csv(f"data/models_csv/df_{name}.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Ordenamos obligatoriamente por equipo y fecha para que las ventanas se calculen bien
    df.sort_values(by=["team", "date"], inplace=True)

    # Separamos el Pasado (Entrenamiento) del Futuro (Mundial)
    df_historia = df[df["date"] < "2026-06-11"].copy()
    df_mundial = df[df["date"] >= "2026-06-11"].copy()

    # =====================================================================
    # 3. INGENIERÍA DE DATOS (SÓLO EN EL HISTÓRICO)
    # =====================================================================
    print("2. Calculando ventanas, PCA y pesos...")

    # --- Ventanas Móviles ---
    df_historia["gf_prom_5"] = df_historia.groupby("team")["goals"].transform(lambda x: calculate_window(x, 5))
    df_historia["gc_prom_5"] = df_historia.groupby("team")["goals_conceded"].transform(lambda x: calculate_window(x, 5))
    df_historia["elo_prom_5"] = df_historia.groupby("team")["opponent_elo"].transform(lambda x: calculate_window(x, 5))
    df_historia["gf_prom_15"] = df_historia.groupby("team")["goals"].transform(lambda x: calculate_window(x, 15))
    df_historia["gc_prom_15"] = df_historia.groupby("team")["goals_conceded"].transform(lambda x: calculate_window(x, 15))

    # --- Variables de Estilo para PCA ---
    df_historia["total_goals"] = df_historia["goals"] + df_historia["goals_conceded"]
    df_historia["pace_20"] = df_historia.groupby("team")["total_goals"].transform(lambda x: calculate_window(x, 20))

    df_historia["gf_ratio"] = df_historia["goals"] / (df_historia["total_goals"] + 1e-9)
    df_historia["gf_ratio"] = np.where(df_historia["total_goals"] == 0, 0.5, df_historia["gf_ratio"])
    df_historia["gf_ratio_20"] = df_historia.groupby("team")["gf_ratio"].transform(lambda x: calculate_window(x, 20))

    df_historia["clean_ratio"] = (df_historia["goals_conceded"] == 0).astype(int)
    df_historia["clean_20"] = df_historia.groupby("team")["clean_ratio"].transform(lambda x: calculate_window(x, 20))

    df_historia["dif_goals_elo"] = np.where(df_historia["opponent_elo"] > df_historia["elo"], df_historia["goals"] - df_historia["goals_conceded"], np.nan)
    df_historia["underdog_20"] = df_historia.groupby("team")["dif_goals_elo"].transform(lambda x: calculate_window(x, 20))
    df_historia["underdog_20"].fillna(0, inplace=True)

    # --- Cálculo del PCA ---
    columns_pca = ["pace_20", "gf_ratio_20", "clean_20", "underdog_20"]
    no_nan_rows = df_historia[columns_pca].notna().all(axis=1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_historia.loc[no_nan_rows, columns_pca])
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_data)

    df_historia["PCA_1"] = np.nan
    df_historia["PCA_2"] = np.nan
    df_historia.loc[no_nan_rows, "PCA_1"] = pca_data[:, 0]
    df_historia.loc[no_nan_rows, "PCA_2"] = pca_data[:, 1]

    # --- Confederaciones ---
    teams_map = {team: confed for confed, teams in confed_dictionary.items() for team in teams}
    df_historia["confed_text"] = df_historia["team"].map(teams_map).fillna("OTHER")
    df_historia["rival_confed_text"] = df_historia["opponent"].map(teams_map).fillna("OTHER")

    numeric_map = {"UEFA": 1, "CONMEBOL": 2, "CONCACAF": 3, "CAF": 4, "AFC": 5, "OFC": 6, "OTHER": 0}
    df_historia["confed"] = df_historia["confed_text"].map(numeric_map).astype(int)
    df_historia["rival_confed"] = df_historia["rival_confed_text"].map(numeric_map).astype(int)

    # --- Torneos (Optimizado con .loc) ---
    df_historia["tournament_num"] = 1
    df_historia.loc[df_historia["tournament"].isin(["FIFA World Cup", "Copa América", "UEFA Euro", "African Cup of Nations"]), "tournament_num"] = 5
    df_historia.loc[df_historia["tournament"].isin(["UEFA Nations League", "CONMEBOL-UEFA Cup of Champions", "FIFA World Cup qualification", "AFC Asian Cup", "Gold Cup"]), "tournament_num"] = 4
    df_historia.loc[df_historia["tournament"].isin(["UEFA Euro qualification", "AFC Asian Cup qualification", "CONCACAF Nations League", "African Cup of Nations qualification", "Gold Cup qualification"]), "tournament_num"] = 3
    df_historia.loc[df_historia["tournament"].isin(["Friendly", "FIFA series", "CONCACAF Nations League qualification"]), "tournament_num"] = 2

    # --- ¡EL FIX DEL MERGE! Cruce de estadísticas del Rival ---
    # 1. Eliminamos las columnas viejas del rival si existían en el CSV original
    cols_rivales_viejas = ["rival_gf_prom_5", "rival_gc_prom_5", "rival_elo_prom_5", "rival_gf_prom_15", "rival_gc_prom_15", "rival_PCA_1", "rival_PCA_2"]
    df_historia.drop(columns=[c for c in cols_rivales_viejas if c in df_historia.columns], inplace=True, errors='ignore')

    # 2. Preparamos el dataframe del rival limpio
    df_rival_stats = df_historia[["date", "team", "gf_prom_5", "gc_prom_5", "elo_prom_5", "gf_prom_15", "gc_prom_15", "PCA_1", "PCA_2"]].copy()
    df_rival_stats.columns = ["date", "opponent", "rival_gf_prom_5", "rival_gc_prom_5", "rival_elo_prom_5", "rival_gf_prom_15", "rival_gc_prom_15", "rival_PCA_1", "rival_PCA_2"]

    # 3. Cruzamos
    df_historia = pd.merge(df_historia, df_rival_stats, on=["date", "opponent"], how="left")

    # --- Pesos Temporales (Time Decay) ---
    reference_date = pd.to_datetime("2024-06-01")
    df_historia["days_away"] = (reference_date - df_historia["date"]).dt.days.clip(lower=0)
    lambda_decay = np.log(2) / 1277 
    df_historia["date_weight"] = np.exp(-lambda_decay * df_historia["days_away"])

    # =====================================================================
    # 4. ENTRENAMIENTO DE XGBOOST
    # =====================================================================
    print("3. Entrenando el motor XGBoost...")
    features = [
        'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
        'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2', 
        'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15', 'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2'
    ]
    target = "goals"

    # Rellenar NaNs con la media y quitar filas sin goles para entrenar
    df_historia[features] = df_historia[features].fillna(df_historia[features].mean())
    df_train = df_historia.dropna(subset=[target]).copy()

    # Validación "cruzada"

    # tscv = TimeSeriesSplit(n_splits=5)

    # parametros = {
    #     'max_depth': [3, 4, 5],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'n_estimators': [100, 300, 500],
    #     'subsample': [0.7, 0.8, 1.0],
    #     'tweedie_variance_power': [1.3, 1.5, 1.7]
    # }

    # optimizador = RandomizedSearchCV(
    #     estimator=xgb.XGBRegressor(objective='reg:tweedie', random_state=42),
    #     param_distributions=parametros,
    #     n_iter=20,
    #     cv=tscv,
    #     scoring='neg_mean_absolute_error',
    #     random_state=42
    # )

    # optimizador.fit(df_train[features], df_train[target], sample_weight=df_train["date_weight"])

    # print("Mejores parámetros:", optimizador.best_params_)

    # Modelo original
    # model = xgb.XGBRegressor(
    #     objective='reg:tweedie',
    #     tweedie_variance_power=1.5,
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=4,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )

    # Hiperparámetros buenos
    model = xgb.XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=1.3,
        n_estimators=500,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(df_train[features], df_train[target], sample_weight=df_train["date_weight"])

    # Persistir el modelo para el endpoint serverless de knockouts
    os.makedirs("api/models", exist_ok=True)
    model.save_model(f"api/models/xgb_{name}.json")

    # =====================================================================
    # 5. INFERENCIA DEL MUNDIAL (LA FOTO FIJA)
    # =====================================================================
    print("4. Generando predicciones para el Mundial 2026...")

    # Sacamos la FOTO FIJA (El último partido de cada equipo antes del Mundial)
    df_historia.sort_values(by="date", inplace=True)
    foto_fija_equipos = df_historia.groupby("team")[['gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2', 'confed']].last().reset_index()

    foto_fija_rivales = foto_fija_equipos.copy()
    foto_fija_rivales.columns = ['opponent', 'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15', 'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2', 'rival_confed']

    # Dividimos el calendario del mundial que nos llegó del CSV en 3 jornadas
    jornadas = {
        "J1": df_mundial[df_mundial["date"] < "2026-06-18"].copy(),
        "J2": df_mundial[(df_mundial["date"] >= "2026-06-18") & (df_mundial["date"] < "2026-06-24")].copy(),
        "J3": df_mundial[df_mundial["date"] >= "2026-06-24"].copy()
    }

    for nombre_jornada, df_jornada in jornadas.items():
        if len(df_jornada) == 0:
            continue
            
        # ==========================================
        # ¡EL FIX! Borramos las columnas métricas vacías que vienen del CSV 
        # para que el merge no cree las odiosas terminaciones _x e _y
        # ==========================================
        cols_a_borrar = [c for c in foto_fija_equipos.columns if c != 'team'] + \
                        [c for c in foto_fija_rivales.columns if c != 'opponent']
        
        df_jornada.drop(columns=[c for c in cols_a_borrar if c in df_jornada.columns], inplace=True, errors='ignore')
        
        # Ahora inyectamos la foto fija del equipo y del rival de forma limpia
        df_jornada = pd.merge(df_jornada, foto_fija_equipos, on="team", how="left")
        df_jornada = pd.merge(df_jornada, foto_fija_rivales, on="opponent", how="left")
        
        # Contexto del torneo (El Mundial es nivel 5)
        df_jornada["tournament_num"] = 5
        
        df_jornada[features] = df_jornada[features].fillna(df_historia[features].mean())
        
        # Predicción
        df_jornada['xg_estimated'] = model.predict(df_jornada[features]).round(2)
        
        # Guardamos los resultados
        cols_export = ["date", "team", "opponent", "xg_estimated"]
        df_jornada.sort_values(by=["date", "team"], inplace=True)
        df_jornada[cols_export].to_csv(f"data/ai_models/xg_preds_{nombre_jornada}_{name}.csv", index=False)
        df_jornada.to_csv(f"data/ai_models/xg_preds_{nombre_jornada}_{name}_complete.csv", index=False)
        print(f"✅ Predicciones para {nombre_jornada} guardadas con éxito.")

    print("\n¡Simulación completa! Revisa la carpeta 'data'.")

    return model