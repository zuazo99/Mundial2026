from clases_simulacion import Tournament


# Torneo completo
if __name__ == "__main__":
    # Mister Claude (normal)
    tournament = Tournament(name="misterclaude")
    tournament.simulate_tournament()
    tournament.simulate_real_bracket()

    # Gemaldini (+200 elo a Francia, España, Portugal, Inglaterra, Noruega, +100 a resto Europa normal, +50 a Germany, Netherlands)
    tournament = Tournament(name="gemaldini")
    tournament.simulate_tournament()
    tournament.simulate_real_bracket()

    # Dav-GPO (+50 a Panamá, +100 a Brazil, +150 a Uruguay y México, +200 a Argentina, Colombia, Ecuador y Paraguay)
    tournament = Tournament(name="dav_gpo")
    tournament.simulate_tournament()
    tournament.simulate_real_bracket()