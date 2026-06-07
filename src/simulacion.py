from src.clases_simulacion import Tournament


# # Group A
# MEXICO = Team(name="Mexico", elo=1818.0, group="A")
# SOUTH_AFRICA = Team(name="South Africa", elo=1515.0, group="A")
# SOUTH_KOREA = Team(name="South Korea", elo=1722.0, group="A")
# CZECH_REPUBLIC = Team(name="Czech Republic", elo=1691.0, group="A")

# G1 = Group(MEXICO, SOUTH_AFRICA, SOUTH_KOREA, CZECH_REPUBLIC, "A")

# # Group B
# CANADA = Team(name="Canada", elo=1730.0, group="B")
# BOSNIA_AND_HERZEGOVINA = Team(name="Bosnia and Herzegovina", elo=1598.0, group="B")
# QATAR = Team(name="Qatar", elo=1468.0, group="B")
# SWITZERLAND = Team(name="Switzerland", elo=1837.0, group="B")

# G2 = Group(CANADA, BOSNIA_AND_HERZEGOVINA, QATAR, SWITZERLAND, "B")

# # Group C
# BRAZIL = Team(name="Brazil", elo=1919.0, group="C")
# MOROCCO = Team(name="Morocco", elo=1807.0, group="C")
# HAITI = Team(name="Haiti", elo=1505.0, group="C")
# SCOTLAND = Team(name="Scotland", elo=1718.0, group="C")

# G3 = Group(BRAZIL, MOROCCO, HAITI, SCOTLAND, "C")

# # Group D
# # Subir elo de Estados Unidos
# UNITED_STATES = Team(name="United States", elo=1674.0, group="D")
# PARAGUAY = Team(name="Paraguay", elo=1810.0, group="D")
# AUSTRALIA = Team(name="Australia", elo=1735.0, group="D")
# TURKEY = Team(name="Turkey", elo=1858.0, group="D")

# G4 = Group(UNITED_STATES, PARAGUAY, AUSTRALIA, TURKEY, "D")

# # Group E
# GERMANY = Team(name="Germany", elo=1891.0, group="E")
# CURACAO = Team(name="Curaçao", elo=1236.0, group="E")
# IVORY_COAST = Team(name="Ivory Coast", elo=1633.0, group="E")
# ECUADOR = Team(name="Ecuador", elo=1892.0, group="E")

# G5 = Group(GERMANY, CURACAO, IVORY_COAST, ECUADOR, "E")

# # Group F
# NETHERLANDS = Team(name="Netherlands", elo=1904.0, group="F")
# JAPAN = Team(name="Japan", elo=1821.0, group="F")
# SWEDEN = Team(name="Sweden", elo=1704.0, group="F")
# TUNISIA = Team(name="Tunisia", elo=1586.0, group="F")

# G6 = Group(NETHERLANDS, JAPAN, SWEDEN, TUNISIA, "F")

# # Group G
# BELGIUM = Team(name="Belgium", elo=1835.0, group="G")
# EGYPT = Team(name="Egypt", elo=1656.0, group="G")
# IRAN = Team(name="Iran", elo=1730.0, group="G")
# NEW_ZEALAND = Team(name="New Zealand", elo=1528.0, group="G")

# G7 = Group(BELGIUM, EGYPT, IRAN, NEW_ZEALAND, "G")

# # Group H
# SPAIN = Team(name="Spain", elo=2071.0, group="H")
# CAPE_VERDE = Team(name="Cape Verde", elo=1531.0, group="H")
# SAUDI_ARABIA = Team(name="Saudi Arabia", elo=1550.0, group="H")
# URUGUAY = Team(name="Uruguay", elo=1855.0, group="H")

# G8 = Group(SPAIN, CAPE_VERDE, SAUDI_ARABIA, URUGUAY, "H")

# # Group I
# FRANCE = Team(name="France", elo=2008.0, group="I")
# SENEGAL = Team(name="Senegal", elo=1777.0, group="I")
# IRAQ = Team(name="Iraq", elo=1608.0, group="I")
# NORWAY = Team(name="Norway", elo=1806.0, group="I")

# G9 = Group(FRANCE, SENEGAL, IRAQ, NORWAY, "I")

# # Group J
# ARGENTINA = Team(name="Argentina", elo=2048.0, group="J")
# ALGERIA = Team(name="Algeria", elo=1696.0, group="J")
# AUSTRIA = Team(name="Austria", elo=1785.0, group="J")
# JORDAN = Team(name="Jordan", elo=1616.0, group="J")

# G10 = Group(ARGENTINA, ALGERIA, AUSTRIA, JORDAN, "J")

# # Group K
# PORTUGAL = Team(name="Portugal", elo=1914.0, group="K")
# DR_CONGO = Team(name="DR Congo", elo=1603.0, group="K")
# UZBEKISTAN = Team(name="Uzbekistan", elo=1673.0, group="K")
# COLOMBIA = Team(name="Colombia", elo=1907.0, group="K")

# G11 = Group(PORTUGAL, DR_CONGO, UZBEKISTAN, COLOMBIA, "K")

# # Group L
# ENGLAND = Team(name="England", elo=1931.0, group="L")
# CROATIA = Team(name="Croatia", elo=1874.0, group="L")
# GHANA = Team(name="Ghana", elo=1472.0, group="L")
# PANAMA = Team(name="Panama", elo=1690.0, group="L")

# G12 = Group(ENGLAND, CROATIA, GHANA, PANAMA, "L")

# GROUPS = [G1, G2, G3, G4,
#           G5, G6, G7, G8,
#           G9, G10, G11, G12]

# for group in GROUPS:
#     group.simulate_matches()
#     group.sort_group()

# Torneo completo
if __name__ == "__main__":
    # Mister Claude (normal)
    tournament = Tournament(name="misterclaude")
    tournament.simulate_tournament()

    # Gemaldini (+200 elo a Francia, España, Portugal, Inglaterra, Noruega, +100 a resto Europa normal, +50 a Germany, Netherlands)
    tournament = Tournament(name="gemaldini")
    tournament.simulate_tournament()

    # Dav-GPO (+50 a Panamá, +100 a Brazil, +150 a Uruguay y México, +200 a Argentina, Colombia, Ecuador y Paraguay)
    tournament = Tournament(name="dav_gpo")
    tournament.simulate_tournament()