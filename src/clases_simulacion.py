import pandas as pd
import random as rd
from xg_preds import train_model

class Team:
    # Añadir stats de ventana como atributos
    def __init__(self, name, elo, group, df_metrics):
        self.name = name
        self.elo = elo
        self.group = group
        self.points = 0
        self.dg = 0
        self.gf = 0
        self.compute_metrics(df_metrics=df_metrics)

    def reset_points(self):
        self.points = 0

    def update_result(self, g1, g2):
        self.points += 3 if g1 > g2 else (1 if g1 == g2 else 0)
        self.gf += g1
        self.dg += (g1 - g2)

    # Con esto y cruzar, crear dataframe
    def compute_metrics(self, df_metrics: pd.DataFrame):
        df_metrics = df_metrics.loc[df_metrics["team"] == self.name]
        self.gf_prom_5 = df_metrics["gf_prom_5"].item()
        self.gc_prom_5 = df_metrics["gc_prom_5"].item()
        self.gf_prom_15 = df_metrics["gf_prom_15"].item()
        self.gc_prom_15 = df_metrics["gc_prom_15"].item()
        self.elo_prom_5 = df_metrics["elo_prom_5"].item()
        self.pca_1 = df_metrics["PCA_1"].item()
        self.pca_2 = df_metrics["PCA_2"].item()
        self.confed = df_metrics["confed"].item()


class Match:
    def __init__(self, s1, s2, xg1, xg2, ko=False):
        self.s1: Team = s1
        self.s2: Team = s2
        self.xg1_min = xg1 / 90
        self.xg2_min = xg2 / 90
        self.results = {}
        self.num_iterations = 30 # 30
        self.ko = ko

    # def update_elo(self, g1, g2):
    #     w_home = 1.0 if g1 > g2 else (0.5 if g1 == g2 else 0.0)
    #     w_away = 1.0 - w_home

    #     elo_home = self.s1.elo
    #     elo_away = self.s2.elo

    #     we_home = 1 / (1 + 10 ** (((elo_away - (elo_home))) / 400))
    #     we_away = 1 - we_home 

    #     k = 60

    #     elo1 = round(k * (w_home - we_home), 0)
    #     elo2 = round(k * (w_away - we_away), 0)

    #     self.s1.elo += elo1
    #     self.s2.elo += elo2

    #     # print(f"Cambios de ELO - {self.s1.name}: {elo1} - {self.s2.name}: {elo2}")

    def get_multiplier(self, minute, g1, g2):
        better_team = 1 if self.s1.elo >= self.s2.elo else 2

        mult1 = 1.0
        mult2 = 1.0

        if better_team == 1:
            if minute > 59:
                # Con empate -> desempata equipo bueno
                if g1 == g2:
                    mult1 = 1.1
                # Con derrota -> equipo bueno ataca, peor se encierra
                elif g2 > g1:
                    mult1 = 1.2
                    mult2 = 0.7
                # Con mucha ventaja -> peor equipo gol del honor
                elif g1 > g2 + 2:
                    mult2 = 1.1
            elif minute > 45:
                # Con derrota -> equipo bueno ataca, peor se encierra
                if g2 > g1:
                    mult1 = 1.1
        else:
            if minute > 59:
                # Con empate -> desempata equipo bueno
                if g1 == g2:
                    mult2 = 1.1
                # Con derrota -> equipo bueno ataca, peor se encierra
                elif g1 > g2:
                    mult2 = 1.2
                    mult1 = 0.7
                # Con mucha ventaja -> peor equipo gol del honor
                elif g2 > g1 + 2:
                    mult1 = 1.1
            elif minute > 45:
                # Con derrota -> equipo bueno ataca, peor se encierra
                if g1 > g2:
                    mult1 = 1.1

        goal_diff = g1 - g2
        mult1 = 1.3
        mult2 = 1.3

        if goal_diff > 0:
            mult2 = 1.4
            if minute > 80:
                # mult1 = 0.9
                mult2 = 1.7
        elif goal_diff < 0:
            mult1 = 1.4
            if minute > 80:
                mult1 = 1.7
                # mult2 = 0.9
        else:
            if minute > 45:
                if better_team == 1:
                    mult1 = 1.6
                else:
                    mult2 = 1.6

        return mult1, mult2

    def simulate_match(self):
        g1 = 0
        g2 = 0
        # Añadir conteo de iteración para xg adaptativo
        # Añadir ELO para locales
        for min in range(95): # 90 mins + added time
            mult1, mult2 = self.get_multiplier(min, g1, g2)
            if rd.random() < (self.xg1_min * mult1):
                g1 += 1
            if rd.random() < (self.xg2_min * mult2):
                g2 += 1
        if self.ko and g1 == g2:
            for min in range(30):
                mult1, mult2 = self.get_multiplier(min, g1, g2)
                if rd.random() < (self.xg1_min * mult1):
                    g1 += 1
                if rd.random() < (self.xg2_min * mult2):
                    g2 += 1
            if g1 == g2:
                p1, p2 = self.simulate_penalties()
                if p1 > p2:
                    g1 += 1
                    # print(f"{self.s1.name} gana en penaltis {p1} - {p2}")
                elif p2 > p1:
                    g2 += 1
                    # print(f"{self.s2.name} gana en penaltis {p2} - {p1}")

        if (g1, g2) not in self.results:
            self.results[(g1, g2)] = 1
        else:
            self.results[(g1, g2)] += 1

    def simulate_penalties(self):
        prob_pen = 0.75
        p1 = 0
        p2 = 0
        for i in range(5):
            if rd.random() < prob_pen:
                p1 += 1
            if p1 + (4 - i) < p2 or p1 > (4 - i) + p2:
                return p1, p2
            if rd.random() < prob_pen:
                p2 += 1
            if p1 + (4 - i) < p2 or p1 > (4 - i) + p2:
                return p1, p2
        while p1 == p2:
            if rd.random() < prob_pen:
                p1 += 1
            if rd.random() < prob_pen:
                p2 += 1
        return p1, p2

    def simulate_common(self) -> dict:
        for _ in range(self.num_iterations):
            self.simulate_match()

        result_max = max(self.results, key=self.results.get)

        if not self.ko:
            self.s1.update_result(result_max[0], result_max[1])
            self.s2.update_result(result_max[1], result_max[0])

        print(f"{self.s1.name} {result_max[0]} - {result_max[1]} {self.s2.name} - {round(self.results[result_max]*100/self.num_iterations, 2)}%")

        return {self.s1: result_max[0], self.s2: result_max[1]}


class Group:
    def __init__(self, s1, s2, s3, s4, letter, name):
        self.s1: Team = s1
        self.s2: Team = s2
        self.s3: Team = s3
        self.s4: Team = s4
        self.letter = letter
        self.matches: list[Match] = []
        self.df_j1 = pd.read_csv(f"data/ai_models/xg_preds_J1_{name}.csv")
        self.df_j2 = pd.read_csv(f"data/ai_models/xg_preds_J2_{name}.csv")
        self.df_j3 = pd.read_csv(f"data/ai_models/xg_preds_J3_{name}.csv")
        self.results: list[dict] = []

    def search_matches_pandas(self):
        s1_j1 = self.df_j1.loc[self.df_j1["team"] == self.s1.name, "xg_estimated"].item()
        s2_j1 = self.df_j1.loc[self.df_j1["team"] == self.s2.name, "xg_estimated"].item()
        s3_j1 = self.df_j1.loc[self.df_j1["team"] == self.s3.name, "xg_estimated"].item()
        s4_j1 = self.df_j1.loc[self.df_j1["team"] == self.s4.name, "xg_estimated"].item()

        s1_j2 = self.df_j2.loc[self.df_j2["team"] == self.s1.name, "xg_estimated"].item()
        s2_j2 = self.df_j2.loc[self.df_j2["team"] == self.s2.name, "xg_estimated"].item()
        s3_j2 = self.df_j2.loc[self.df_j2["team"] == self.s3.name, "xg_estimated"].item()
        s4_j2 = self.df_j2.loc[self.df_j2["team"] == self.s4.name, "xg_estimated"].item()

        s1_j3 = self.df_j3.loc[self.df_j3["team"] == self.s1.name, "xg_estimated"].item()
        s2_j3 = self.df_j3.loc[self.df_j3["team"] == self.s2.name, "xg_estimated"].item()
        s3_j3 = self.df_j3.loc[self.df_j3["team"] == self.s3.name, "xg_estimated"].item()
        s4_j3 = self.df_j3.loc[self.df_j3["team"] == self.s4.name, "xg_estimated"].item()

        self.matches.append(Match(self.s1, self.s2, s1_j1, s2_j1))
        self.matches.append(Match(self.s3, self.s4, s3_j1, s4_j1))

        self.matches.append(Match(self.s1, self.s3, s1_j2, s3_j2))
        self.matches.append(Match(self.s4, self.s2, s4_j2, s2_j2))

        self.matches.append(Match(self.s4, self.s1, s4_j3, s1_j3))
        self.matches.append(Match(self.s2, self.s3, s2_j3, s3_j3))

    def simulate_matches(self):
        self.search_matches_pandas()
        if not self.matches:
            print("No hay partidos para simular")
        else:
            print("------------------------------------------------")
            print(f"Resultados del grupo {self.letter}")
            for match in self.matches:
                result = match.simulate_common()
                teams = list(result.keys())
                scores = list(result.values())
                self.results.append({teams[0].name: scores[0],
                                     teams[1].name: scores[1]})

    def sort_group(self):
        teams = [self.s1, self.s2, self.s3, self.s4]
        
        final_order = self.tie_break(teams)

        print("------------------------------------------------")
        print(f"Clasificación final grupo {self.letter}")
        for i, team in enumerate(final_order):
            print(f"{i+1}º - {team.name} - Points: {team.points} - DG: {team.dg} - GF: {team.gf}")
            
        return final_order

    def tie_break(self, tied_teams: list[Team]) -> list[Team]:
        if len(tied_teams) <= 1:
            return tied_teams

        h2h_stats = {t.name: {'pts': 0, 'dg': 0, 'gf': 0} for t in tied_teams}
        tied_names = [t.name for t in tied_teams]

        for match_dict in self.results:
            teams_in_match = list(match_dict.keys())
            t1_name, t2_name = teams_in_match[0], teams_in_match[1]
            
            if t1_name in tied_names and t2_name in tied_names:
                g1, g2 = match_dict[t1_name], match_dict[t2_name]
                
                h2h_stats[t1_name]['gf'] += g1
                h2h_stats[t2_name]['gf'] += g2
                h2h_stats[t1_name]['dg'] += (g1 - g2)
                h2h_stats[t2_name]['dg'] += (g2 - g1)
                
                if g1 > g2:
                    h2h_stats[t1_name]['pts'] += 3
                elif g1 < g2:
                    h2h_stats[t2_name]['pts'] += 3
                else:
                    h2h_stats[t1_name]['pts'] += 1
                    h2h_stats[t2_name]['pts'] += 1

        criteria = [
            lambda t: t.points,
            lambda t: h2h_stats[t.name]['pts'],
            lambda t: h2h_stats[t.name]['dg'],
            lambda t: h2h_stats[t.name]['gf'], 
            lambda t: t.dg,                    
            lambda t: t.gf                     
        ]

        # 3. Evaluamos cada criterio
        for criterio in criteria:
            groups = {}
            for t in tied_teams:
                value = criterio(t)
                if value not in groups:
                    groups[value] = []
                groups[value].append(t)

            if len(groups) > 1:
                ordered_values = sorted(groups.keys(), reverse=True)
                partial_result = []
                
                for valor in ordered_values:
                    subgrupo_resuelto = self.tie_break(groups[valor])
                    partial_result.extend(subgrupo_resuelto)
                    
                return partial_result

        return tied_teams
    

class Fixture:
    def __init__(self, s1, s2):
        self.s1: Team = s1
        self.s2: Team = s2


class Knockouts:
    def __init__(self, group_orders, best_thirds, third_letters, model, name):
        self.matches: list[Match] = []
        self.fixtures: list[Fixture] = []
        self.winners = {}
        self.losers = {}
        self.model = model
        self.game_id = 1
        self.results = []
        self.name = name
        self.create_knockouts(group_orders, best_thirds, third_letters)

    def create_first_round(self, group_orders, best_thirds: list[Team], third_letters):
        df_thirds = pd.read_csv("data/mejores_terceros.csv", sep=";")
        thirds = df_thirds.loc[df_thirds["Combinación"] == third_letters]

        A1_rival = thirds["1A"].item()[1]
        B1_rival = thirds["1B"].item()[1]
        D1_rival = thirds["1D"].item()[1]
        E1_rival = thirds["1E"].item()[1]
        G1_rival = thirds["1G"].item()[1]
        I1_rival = thirds["1I"].item()[1]
        K1_rival = thirds["1K"].item()[1]
        L1_rival = thirds["1L"].item()[1]
        
        for third in best_thirds:
            if third.group == A1_rival:
                A1_rival = third
            elif third.group == B1_rival:
                B1_rival = third
            elif third.group == D1_rival:
                D1_rival = third
            elif third.group == E1_rival:
                E1_rival = third
            elif third.group == G1_rival:
                G1_rival = third
            elif third.group == I1_rival:
                I1_rival = third
            elif third.group == K1_rival:
                K1_rival = third
            elif third.group == L1_rival:
                L1_rival = third

        A1 = group_orders[0][0]
        A2 = group_orders[0][1]
        B1 = group_orders[1][0]
        B2 = group_orders[1][1]
        C1 = group_orders[2][0]
        C2 = group_orders[2][1]
        D1 = group_orders[3][0]
        D2 = group_orders[3][1]
        E1 = group_orders[4][0]
        E2 = group_orders[4][1]
        F1 = group_orders[5][0]
        F2 = group_orders[5][1]
        G1 = group_orders[6][0]
        G2 = group_orders[6][1]
        H1 = group_orders[7][0]
        H2 = group_orders[7][1]
        I1 = group_orders[8][0]
        I2 = group_orders[8][1]
        J1 = group_orders[9][0]
        J2 = group_orders[9][1]
        K1 = group_orders[10][0]
        K2 = group_orders[10][1]
        L1 = group_orders[11][0]
        L2 = group_orders[11][1]

        # DAY 1
        self.fixtures.append(Fixture(A2, B2))
        # DAY 2
        self.fixtures.append(Fixture(C1, F2))
        self.fixtures.append(Fixture(E1, E1_rival))
        self.fixtures.append(Fixture(F1, C2))
        # DAY 3
        self.fixtures.append(Fixture(E2, I2))
        self.fixtures.append(Fixture(I1, I1_rival))
        self.fixtures.append(Fixture(A1, A1_rival))
        # DAY 4
        self.fixtures.append(Fixture(L1, L1_rival))
        self.fixtures.append(Fixture(G1, G1_rival))
        self.fixtures.append(Fixture(D1, D1_rival))
        # DAY 5
        self.fixtures.append(Fixture(H1, J2))
        self.fixtures.append(Fixture(K2, L2))
        self.fixtures.append(Fixture(B1, B1_rival))
        # DAY 6
        self.fixtures.append(Fixture(D2, G2))
        self.fixtures.append(Fixture(J1, H2))
        self.fixtures.append(Fixture(K1, K1_rival))

        df_pred: pd.DataFrame = self.predict_xg_matches("first_round")

        A1_goals = df_pred.loc[df_pred["team"] == A1.name, "xg_estimated"].item()
        A2_goals = df_pred.loc[df_pred["team"] == A2.name, "xg_estimated"].item()
        B1_goals = df_pred.loc[df_pred["team"] == B1.name, "xg_estimated"].item()
        B2_goals = df_pred.loc[df_pred["team"] == B2.name, "xg_estimated"].item()
        C1_goals = df_pred.loc[df_pred["team"] == C1.name, "xg_estimated"].item()
        C2_goals = df_pred.loc[df_pred["team"] == C2.name, "xg_estimated"].item()
        D1_goals = df_pred.loc[df_pred["team"] == D1.name, "xg_estimated"].item() 
        D2_goals = df_pred.loc[df_pred["team"] == D2.name, "xg_estimated"].item()
        E1_goals = df_pred.loc[df_pred["team"] == E1.name, "xg_estimated"].item() 
        E2_goals = df_pred.loc[df_pred["team"] == E2.name, "xg_estimated"].item() 
        F1_goals = df_pred.loc[df_pred["team"] == F1.name, "xg_estimated"].item() 
        F2_goals = df_pred.loc[df_pred["team"] == F2.name, "xg_estimated"].item() 
        G1_goals = df_pred.loc[df_pred["team"] == G1.name, "xg_estimated"].item() 
        G2_goals = df_pred.loc[df_pred["team"] == G2.name, "xg_estimated"].item() 
        H1_goals = df_pred.loc[df_pred["team"] == H1.name, "xg_estimated"].item() 
        H2_goals = df_pred.loc[df_pred["team"] == H2.name, "xg_estimated"].item() 
        I1_goals = df_pred.loc[df_pred["team"] == I1.name, "xg_estimated"].item() 
        I2_goals = df_pred.loc[df_pred["team"] == I2.name, "xg_estimated"].item() 
        J1_goals = df_pred.loc[df_pred["team"] == J1.name, "xg_estimated"].item() 
        J2_goals = df_pred.loc[df_pred["team"] == J2.name, "xg_estimated"].item() 
        K1_goals = df_pred.loc[df_pred["team"] == K1.name, "xg_estimated"].item() 
        K2_goals = df_pred.loc[df_pred["team"] == K2.name, "xg_estimated"].item() 
        L1_goals = df_pred.loc[df_pred["team"] == L1.name, "xg_estimated"].item() 
        L2_goals = df_pred.loc[df_pred["team"] == L2.name, "xg_estimated"].item() 
        A1_rival_goals = df_pred.loc[df_pred["team"] == A1_rival.name, "xg_estimated"].item()
        B1_rival_goals = df_pred.loc[df_pred["team"] == B1_rival.name, "xg_estimated"].item() 
        D1_rival_goals = df_pred.loc[df_pred["team"] == D1_rival.name, "xg_estimated"].item()
        E1_rival_goals = df_pred.loc[df_pred["team"] == E1_rival.name, "xg_estimated"].item() 
        G1_rival_goals = df_pred.loc[df_pred["team"] == G1_rival.name, "xg_estimated"].item() 
        I1_rival_goals = df_pred.loc[df_pred["team"] == I1_rival.name, "xg_estimated"].item() 
        K1_rival_goals = df_pred.loc[df_pred["team"] == K1_rival.name, "xg_estimated"].item() 
        L1_rival_goals = df_pred.loc[df_pred["team"] == L1_rival.name, "xg_estimated"].item() 
        
        # DAY 1
        self.matches.append(Match(A2, B2, A2_goals, B2_goals, True))
        # DAY 2
        self.matches.append(Match(C1, F2, C1_goals, F2_goals, True))
        self.matches.append(Match(E1, E1_rival, E1_goals, E1_rival_goals, True))
        self.matches.append(Match(F1, C2, F1_goals, C2_goals, True))
        # DAY 3
        self.matches.append(Match(E2, I2, E2_goals, I2_goals, True))
        self.matches.append(Match(I1, I1_rival, I1_goals, I1_rival_goals, True))
        self.matches.append(Match(A1, A1_rival, A1_goals, A1_rival_goals, True))
        # DAY 4
        self.matches.append(Match(L1, L1_rival, L1_goals, L1_rival_goals, True))
        self.matches.append(Match(G1, G1_rival, G1_goals, G1_rival_goals, True))
        self.matches.append(Match(D1, D1_rival, D1_goals, D1_rival_goals, True))
        # DAY 5
        self.matches.append(Match(H1, J2, H1_goals, J2_goals, True))
        self.matches.append(Match(K2, L2, K2_goals, L2_goals, True))
        self.matches.append(Match(B1, B1_rival, B1_goals, B1_rival_goals, True))
        # DAY 6
        self.matches.append(Match(D2, G2, D2_goals, G2_goals, True))
        self.matches.append(Match(J1, H2, J1_goals, H2_goals, True))
        self.matches.append(Match(K1, K1_rival, K1_goals, K1_rival_goals, True))

    # def simulate_first_round(self):
    #     for i, match in enumerate(self.matches):
    #         result = match.simulate_common()
    #         teams = list(result.keys())
    #         scores = list(result.values())
    #         self.winners[str(i+1)] = teams[0] if scores[0] > scores[1] else teams[1]
        
    #     self.matches.clear()

    def create_sweet16(self):
        # DAY 1
        self.fixtures.append(Fixture(self.winners["1"], self.winners["4"]))
        self.fixtures.append(Fixture(self.winners["3"], self.winners["6"]))

        # DAY 2
        self.fixtures.append(Fixture(self.winners["2"], self.winners["5"]))
        self.fixtures.append(Fixture(self.winners["7"], self.winners["8"]))

        # DAY 3
        self.fixtures.append(Fixture(self.winners["11"], self.winners["12"]))
        self.fixtures.append(Fixture(self.winners["10"], self.winners["9"]))

        # DAY 4
        self.fixtures.append(Fixture(self.winners["15"], self.winners["14"]))
        self.fixtures.append(Fixture(self.winners["13"], self.winners["16"]))

        df_pred = self.predict_xg_matches("sweet16")

        goals_1 = df_pred.loc[df_pred["team"] == self.winners["1"].name, "xg_estimated"].item()
        goals_2 = df_pred.loc[df_pred["team"] == self.winners["2"].name, "xg_estimated"].item()
        goals_3 = df_pred.loc[df_pred["team"] == self.winners["3"].name, "xg_estimated"].item()
        goals_4 = df_pred.loc[df_pred["team"] == self.winners["4"].name, "xg_estimated"].item()
        goals_5 = df_pred.loc[df_pred["team"] == self.winners["5"].name, "xg_estimated"].item()
        goals_6 = df_pred.loc[df_pred["team"] == self.winners["6"].name, "xg_estimated"].item()
        goals_7 = df_pred.loc[df_pred["team"] == self.winners["7"].name, "xg_estimated"].item()
        goals_8 = df_pred.loc[df_pred["team"] == self.winners["8"].name, "xg_estimated"].item()
        goals_9 = df_pred.loc[df_pred["team"] == self.winners["9"].name, "xg_estimated"].item()
        goals_10 = df_pred.loc[df_pred["team"] == self.winners["10"].name, "xg_estimated"].item()
        goals_11 = df_pred.loc[df_pred["team"] == self.winners["11"].name, "xg_estimated"].item()
        goals_12 = df_pred.loc[df_pred["team"] == self.winners["12"].name, "xg_estimated"].item()
        goals_13 = df_pred.loc[df_pred["team"] == self.winners["13"].name, "xg_estimated"].item()
        goals_14 = df_pred.loc[df_pred["team"] == self.winners["14"].name, "xg_estimated"].item()
        goals_15 = df_pred.loc[df_pred["team"] == self.winners["15"].name, "xg_estimated"].item()
        goals_16 = df_pred.loc[df_pred["team"] == self.winners["16"].name, "xg_estimated"].item()

        # DAY 1
        self.matches.append(Match(self.winners["1"], self.winners["4"], goals_1, goals_4, True))
        self.matches.append(Match(self.winners["3"], self.winners["6"], goals_3, goals_6, True))

        # DAY 2
        self.matches.append(Match(self.winners["2"], self.winners["5"], goals_2, goals_5, True))
        self.matches.append(Match(self.winners["7"], self.winners["8"], goals_7, goals_8, True))

        # DAY 3
        self.matches.append(Match(self.winners["11"], self.winners["12"], goals_11, goals_12, True))
        self.matches.append(Match(self.winners["10"], self.winners["9"], goals_10, goals_9, True))

        # DAY 4
        self.matches.append(Match(self.winners["15"], self.winners["14"], goals_15, goals_14, True))
        self.matches.append(Match(self.winners["13"], self.winners["16"], goals_13, goals_16, True))

        self.winners.clear()

    # def simulate_sweet16(self):
    #     for i, match in enumerate(self.matches):

    def create_elite8(self):
        # DAY 1
        self.fixtures.append(Fixture(self.winners["1"], self.winners["2"]))

        # DAY 2
        self.fixtures.append(Fixture(self.winners["5"], self.winners["6"]))

        # DAY 3
        self.fixtures.append(Fixture(self.winners["3"], self.winners["4"]))
        self.fixtures.append(Fixture(self.winners["7"], self.winners["8"]))

        df_pred = self.predict_xg_matches("elite8")

        goals_1 = df_pred.loc[df_pred["team"] == self.winners["1"].name, "xg_estimated"].item()
        goals_2 = df_pred.loc[df_pred["team"] == self.winners["2"].name, "xg_estimated"].item()
        goals_3 = df_pred.loc[df_pred["team"] == self.winners["3"].name, "xg_estimated"].item()
        goals_4 = df_pred.loc[df_pred["team"] == self.winners["4"].name, "xg_estimated"].item()
        goals_5 = df_pred.loc[df_pred["team"] == self.winners["5"].name, "xg_estimated"].item()
        goals_6 = df_pred.loc[df_pred["team"] == self.winners["6"].name, "xg_estimated"].item()
        goals_7 = df_pred.loc[df_pred["team"] == self.winners["7"].name, "xg_estimated"].item()
        goals_8 = df_pred.loc[df_pred["team"] == self.winners["8"].name, "xg_estimated"].item()

        # DAY 1
        self.matches.append(Match(self.winners["1"], self.winners["2"], goals_1, goals_2, True))

        # DAY 2
        self.matches.append(Match(self.winners["5"], self.winners["6"], goals_5, goals_6, True))

        # DAY 3
        self.matches.append(Match(self.winners["3"], self.winners["4"], goals_3, goals_4, True))
        self.matches.append(Match(self.winners["7"], self.winners["8"], goals_7, goals_8, True))

        self.winners.clear()

    # def simulate_elite8(self):
    #     pass

    def create_semis(self):
        # DAY 1
        self.fixtures.append(Fixture(self.winners["1"], self.winners["2"]))

        # DAY 2
        self.fixtures.append(Fixture(self.winners["3"], self.winners["4"]))

        df_pred = self.predict_xg_matches("semis")

        goals_1 = df_pred.loc[df_pred["team"] == self.winners["1"].name, "xg_estimated"].item()
        goals_2 = df_pred.loc[df_pred["team"] == self.winners["2"].name, "xg_estimated"].item()
        goals_3 = df_pred.loc[df_pred["team"] == self.winners["3"].name, "xg_estimated"].item()
        goals_4 = df_pred.loc[df_pred["team"] == self.winners["4"].name, "xg_estimated"].item()

        # DAY 1
        self.matches.append(Match(self.winners["1"], self.winners["2"], goals_1, goals_2, True))

        # DAY 2
        self.matches.append(Match(self.winners["3"], self.winners["4"], goals_3, goals_4, True))

        self.winners.clear()

    # def simulate_semis(self):
    #     pass

    # + 3º y 4º puesto
    def create_final(self):
        # 3º 4º
        self.fixtures.append(Fixture(self.losers["1"], self.losers["2"]))

        # FINAL
        self.fixtures.append(Fixture(self.winners["1"], self.winners["2"]))

        df_pred = self.predict_xg_matches("final")

        goals_1 = df_pred.loc[df_pred["team"] == self.winners["1"].name, "xg_estimated"].item()
        goals_2 = df_pred.loc[df_pred["team"] == self.winners["2"].name, "xg_estimated"].item()
        goals_3 = df_pred.loc[df_pred["team"] == self.losers["1"].name, "xg_estimated"].item()
        goals_4 = df_pred.loc[df_pred["team"] == self.losers["2"].name, "xg_estimated"].item()

        # 3º 4º
        self.matches.append(Match(self.losers["1"], self.losers["2"], goals_3, goals_4, True))

        # FINAL
        self.matches.append(Match(self.winners["1"], self.winners["2"], goals_1, goals_2, True))

        self.winners.clear()

    # def simulate_final(self):
    #     pass

    def print_winner(self):
        winner = self.winners["2"]
        print(f"El ganador es {winner.name}")

    def simulate_round(self, name, include_losers=False):
        print("------------------------------------------------")
        print(f"Resultados de {name}")
        for i, match in enumerate(self.matches):
            result = match.simulate_common()
            teams = list(result.keys())
            scores = list(result.values())
            self.winners[str(i+1)] = teams[0] if scores[0] > scores[1] else teams[1]
            if include_losers:
                self.losers[str(i+1)] = teams[1] if scores[0] > scores[1] else teams[0]
            self.results.append({"Team_1": teams[0].name,
                                 "Score_1": scores[0],
                                 "Score_2": scores[1],
                                 "Team_2": teams[1].name})
        
        self.matches.clear()
        self.fixtures.clear()

    def create_knockouts(self, group_orders, best_thirds, third_letters):
        self.create_first_round(group_orders, best_thirds, third_letters)

    def simulate_knockouts(self):
        self.simulate_round("First Round")
        self.create_sweet16()
        self.simulate_round("Sweet 16")
        self.create_elite8()
        self.simulate_round("Elite 8")
        self.create_semis()
        self.simulate_round("Semis", include_losers=True)
        self.create_final()
        self.simulate_round("Final")

    def predict_xg_matches(self, round):  
        features = [
            'elo', 'opponent_elo', 'is_home', 'tournament_num', 'confed', 'rival_confed',
            'gf_prom_5', 'gc_prom_5', 'elo_prom_5', 'gf_prom_15', 'gc_prom_15', 'PCA_1', 'PCA_2', 
            'rival_gf_prom_5', 'rival_gc_prom_5', 'rival_elo_prom_5', 'rival_gf_prom_15', 'rival_gc_prom_15', 'rival_PCA_1', 'rival_PCA_2'
        ]
        
        data_rows = []
        for m in self.fixtures:
            is_home_1 = True if m.s1.name in ["United States", "Canada", "Mexico"] else False
            data_rows.append({
                'date': self.game_id, 'team': m.s1.name, 'opponent': m.s2.name, 'elo': m.s1.elo, 'opponent_elo': m.s2.elo, 'is_home': is_home_1, 'tournament_num': 5,
                'confed': m.s1.confed, 'rival_confed': m.s2.confed,
                'gf_prom_5': m.s1.gf_prom_5, 'gc_prom_5': m.s1.gc_prom_5, 'elo_prom_5': m.s1.elo_prom_5,
                'gf_prom_15': m.s1.gf_prom_15, 'gc_prom_15': m.s1.gc_prom_15, 'PCA_1': m.s1.pca_1, 'PCA_2': m.s1.pca_2,
                'rival_gf_prom_5': m.s2.gf_prom_5, 'rival_gc_prom_5': m.s2.gc_prom_5, 'rival_elo_prom_5': m.s2.elo_prom_5,
                'rival_gf_prom_15': m.s2.gf_prom_15, 'rival_gc_prom_15': m.s2.gc_prom_15, 'rival_PCA_1': m.s2.pca_1, 'rival_PCA_2': m.s2.pca_2
            })
            
            is_home_2 = True if m.s1.name in ["United States", "Canada", "Mexico"] else False
            data_rows.append({
                'date': self.game_id, 'team': m.s2.name, 'opponent': m.s1.name, 'elo': m.s2.elo, 'opponent_elo': m.s1.elo, 'is_home': is_home_2, 'tournament_num': 5,
                'confed': m.s2.confed, 'rival_confed': m.s1.confed,
                'gf_prom_5': m.s2.gf_prom_5, 'gc_prom_5': m.s2.gc_prom_5, 'elo_prom_5': m.s2.elo_prom_5,
                'gf_prom_15': m.s2.gf_prom_15, 'gc_prom_15': m.s2.gc_prom_15, 'PCA_1': m.s2.pca_1, 'PCA_2': m.s2.pca_2,
                'rival_gf_prom_5': m.s1.gf_prom_5, 'rival_gc_prom_5': m.s1.gc_prom_5, 'rival_elo_prom_5': m.s1.elo_prom_5,
                'rival_gf_prom_15': m.s1.gf_prom_15, 'rival_gc_prom_15': m.s1.gc_prom_15, 'rival_PCA_1': m.s1.pca_1, 'rival_PCA_2': m.s1.pca_2
            })

            self.game_id += 1

        # df_pred = pd.DataFrame(data_rows, columns=features)
        df_pred = pd.DataFrame(data_rows)

        df_pred['xg_estimated'] = self.model.predict(df_pred[features]).round(2)

        cols_export = ["date", "team", "opponent", "xg_estimated"]
        df_pred.sort_values(by=["date", "team"], inplace=True)
        df_pred[cols_export].to_csv(f"data/xg_preds_{round}_{self.name}.csv", index=False)
        # print(f"✅ Predicciones de xg para {round} guardadas con éxito.")

        return df_pred

    def export_results(self):
        return self.results


class Tournament:
    def __init__(self, name):
        self.group_orders: list[list[Team]] = []
        self.thirds: list[Team] = []
        self.thirds_letters = ""
        self.model = train_model(name=name)
        self.match_history = []
        self.name = name
        self.GROUPS = self.create_groups()

    def create_groups(self):
        df_metrics = pd.read_csv(f"data/ai_models/xg_preds_J1_{self.name}_complete.csv")

        # Group A
        MEXICO = Team(name="Mexico", elo=1818.0, group="A", df_metrics=df_metrics)
        SOUTH_AFRICA = Team(name="South Africa", elo=1515.0, group="A", df_metrics=df_metrics)
        SOUTH_KOREA = Team(name="South Korea", elo=1722.0, group="A", df_metrics=df_metrics)
        CZECH_REPUBLIC = Team(name="Czech Republic", elo=1691.0, group="A", df_metrics=df_metrics)

        self.G1 = Group(MEXICO, SOUTH_AFRICA, SOUTH_KOREA, CZECH_REPUBLIC, "A", self.name)

        # Group B
        CANADA = Team(name="Canada", elo=1730.0, group="B", df_metrics=df_metrics)
        BOSNIA_AND_HERZEGOVINA = Team(name="Bosnia and Herzegovina", elo=1598.0, group="B", df_metrics=df_metrics)
        QATAR = Team(name="Qatar", elo=1468.0, group="B", df_metrics=df_metrics)
        SWITZERLAND = Team(name="Switzerland", elo=1837.0, group="B", df_metrics=df_metrics)

        self.G2 = Group(CANADA, BOSNIA_AND_HERZEGOVINA, QATAR, SWITZERLAND, "B", self.name)

        # Group C
        BRAZIL = Team(name="Brazil", elo=1919.0, group="C", df_metrics=df_metrics)
        MOROCCO = Team(name="Morocco", elo=1807.0, group="C", df_metrics=df_metrics)
        HAITI = Team(name="Haiti", elo=1505.0, group="C", df_metrics=df_metrics)
        SCOTLAND = Team(name="Scotland", elo=1718.0, group="C", df_metrics=df_metrics)

        self.G3 = Group(BRAZIL, MOROCCO, HAITI, SCOTLAND, "C", self.name)

        # Group D
        # Subir elo de Estados Unidos
        UNITED_STATES = Team(name="United States", elo=1674.0, group="D", df_metrics=df_metrics)
        PARAGUAY = Team(name="Paraguay", elo=1810.0, group="D", df_metrics=df_metrics)
        AUSTRALIA = Team(name="Australia", elo=1735.0, group="D", df_metrics=df_metrics)
        TURKEY = Team(name="Turkey", elo=1858.0, group="D", df_metrics=df_metrics)

        self.G4 = Group(UNITED_STATES, PARAGUAY, AUSTRALIA, TURKEY, "D", self.name)

        # Group E
        GERMANY = Team(name="Germany", elo=1891.0, group="E", df_metrics=df_metrics)
        CURACAO = Team(name="Curaçao", elo=1236.0, group="E", df_metrics=df_metrics)
        IVORY_COAST = Team(name="Ivory Coast", elo=1633.0, group="E", df_metrics=df_metrics)
        ECUADOR = Team(name="Ecuador", elo=1892.0, group="E", df_metrics=df_metrics)

        self.G5 = Group(GERMANY, CURACAO, IVORY_COAST, ECUADOR, "E", self.name)

        # Group F
        NETHERLANDS = Team(name="Netherlands", elo=1904.0, group="F", df_metrics=df_metrics)
        JAPAN = Team(name="Japan", elo=1821.0, group="F", df_metrics=df_metrics)
        SWEDEN = Team(name="Sweden", elo=1704.0, group="F", df_metrics=df_metrics)
        TUNISIA = Team(name="Tunisia", elo=1586.0, group="F", df_metrics=df_metrics)

        self.G6 = Group(NETHERLANDS, JAPAN, SWEDEN, TUNISIA, "F", self.name)

        # Group G
        BELGIUM = Team(name="Belgium", elo=1835.0, group="G", df_metrics=df_metrics)
        EGYPT = Team(name="Egypt", elo=1656.0, group="G", df_metrics=df_metrics)
        IRAN = Team(name="Iran", elo=1730.0, group="G", df_metrics=df_metrics)
        NEW_ZEALAND = Team(name="New Zealand", elo=1528.0, group="G", df_metrics=df_metrics)

        self.G7 = Group(BELGIUM, EGYPT, IRAN, NEW_ZEALAND, "G", self.name)

        # Group H
        SPAIN = Team(name="Spain", elo=2071.0, group="H", df_metrics=df_metrics)
        CAPE_VERDE = Team(name="Cape Verde", elo=1531.0, group="H", df_metrics=df_metrics)
        SAUDI_ARABIA = Team(name="Saudi Arabia", elo=1550.0, group="H", df_metrics=df_metrics)
        URUGUAY = Team(name="Uruguay", elo=1855.0, group="H", df_metrics=df_metrics)

        self.G8 = Group(SPAIN, CAPE_VERDE, SAUDI_ARABIA, URUGUAY, "H", self.name)

        # Group I
        FRANCE = Team(name="France", elo=2008.0, group="I", df_metrics=df_metrics)
        SENEGAL = Team(name="Senegal", elo=1777.0, group="I", df_metrics=df_metrics)
        IRAQ = Team(name="Iraq", elo=1608.0, group="I", df_metrics=df_metrics)
        NORWAY = Team(name="Norway", elo=1806.0, group="I", df_metrics=df_metrics)

        self.G9 = Group(FRANCE, SENEGAL, IRAQ, NORWAY, "I", self.name)

        # Group J
        ARGENTINA = Team(name="Argentina", elo=2048.0, group="J", df_metrics=df_metrics)
        ALGERIA = Team(name="Algeria", elo=1696.0, group="J", df_metrics=df_metrics)
        AUSTRIA = Team(name="Austria", elo=1785.0, group="J", df_metrics=df_metrics)
        JORDAN = Team(name="Jordan", elo=1616.0, group="J", df_metrics=df_metrics)

        self.G10 = Group(ARGENTINA, ALGERIA, AUSTRIA, JORDAN, "J", self.name)

        # Group K
        PORTUGAL = Team(name="Portugal", elo=1914.0, group="K", df_metrics=df_metrics)
        DR_CONGO = Team(name="DR Congo", elo=1603.0, group="K", df_metrics=df_metrics)
        UZBEKISTAN = Team(name="Uzbekistan", elo=1673.0, group="K", df_metrics=df_metrics)
        COLOMBIA = Team(name="Colombia", elo=1907.0, group="K", df_metrics=df_metrics)

        self.G11 = Group(PORTUGAL, DR_CONGO, UZBEKISTAN, COLOMBIA, "K", self.name)

        # Group L
        ENGLAND = Team(name="England", elo=1931.0, group="L", df_metrics=df_metrics)
        CROATIA = Team(name="Croatia", elo=1874.0, group="L", df_metrics=df_metrics)
        GHANA = Team(name="Ghana", elo=1472.0, group="L", df_metrics=df_metrics)
        PANAMA = Team(name="Panama", elo=1690.0, group="L", df_metrics=df_metrics)

        self.G12 = Group(ENGLAND, CROATIA, GHANA, PANAMA, "L", self.name)

        groups = [self.G1, self.G2, self.G3, self.G4,
                       self.G5, self.G6, self.G7, self.G8,
                       self.G9, self.G10, self.G11, self.G12]
        
        return groups

    def simulate_groups(self):
        for group in self.GROUPS:
            group.simulate_matches()
            group_order = group.sort_group()
            self.group_orders.append(group_order)

            for result in group.results:
                teams = list(result.keys())
                scores = list(result.values())
                self.match_history.append({"Team_1": teams[0],
                                           "Score_1": scores[0],
                                           "Score_2": scores[1],
                                           "Team_2": teams[1]})


    def best_thirds(self):
        group_thirds = [teams[2] for teams in self.group_orders]
        self.thirds = sorted(group_thirds, key=lambda x: [x.points, x.dg, x.gf], reverse=True)[:8]
        letters = sorted([third.group for third in self.thirds])
        for letter in letters:
            self.thirds_letters += letter

        print("------------------------------------------------")
        print("Mejores terceros ordenados")
        for i, third in enumerate(self.thirds):
            print(f"{i+1}º - {third.name} - Points: {third.points} - DG: {third.dg} - GF: {third.gf}")

    # Añadir equipos al csv para calcular goles esperados, coger los datos del with opps
    def create_knockouts(self):
        self.knockouts = Knockouts(self.group_orders, self.thirds, self.thirds_letters, self.model, self.name)

    def simulate_knockouts(self):
        self.knockouts.simulate_knockouts()
        self.knockouts.print_winner()
    
    def export_results(self):
        results_knockouts = self.knockouts.export_results()
        self.match_history.extend(results_knockouts)

        df_export = pd.DataFrame(self.match_history, columns=["Team_1", "Score_1", "Score_2", "Team_2"])
        df_export.to_csv(f"results/predictions_{self.name}.csv", index=False)

    def simulate_tournament(self):
        self.simulate_groups()
        self.best_thirds()
        self.create_knockouts()
        self.simulate_knockouts()
        self.export_results()