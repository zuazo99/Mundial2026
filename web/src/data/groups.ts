// Group compositions, fixture order, and ELO ratings
// Mirrors clases_simulacion.py:723-818

export interface GroupDef {
  letter: string;
  // s1–s4 slot order matters: determines which team plays which on each matchday
  // J1: s1 vs s2, s3 vs s4  |  J2: s1 vs s3, s4 vs s2  |  J3: s4 vs s1, s2 vs s3
  s1: string; s2: string; s3: string; s4: string;
}

export const GROUPS: GroupDef[] = [
  { letter: "A", s1: "Mexico",    s2: "South Africa",          s3: "South Korea",   s4: "Czech Republic"       },
  { letter: "B", s1: "Canada",    s2: "Bosnia and Herzegovina", s3: "Qatar",         s4: "Switzerland"          },
  { letter: "C", s1: "Brazil",    s2: "Morocco",               s3: "Haiti",         s4: "Scotland"             },
  { letter: "D", s1: "United States", s2: "Paraguay",          s3: "Australia",     s4: "Turkey"               },
  { letter: "E", s1: "Germany",   s2: "Curaçao",               s3: "Ivory Coast",   s4: "Ecuador"              },
  { letter: "F", s1: "Netherlands", s2: "Japan",               s3: "Sweden",        s4: "Tunisia"              },
  { letter: "G", s1: "Belgium",   s2: "Egypt",                 s3: "Iran",          s4: "New Zealand"          },
  { letter: "H", s1: "Spain",     s2: "Cape Verde",            s3: "Saudi Arabia",  s4: "Uruguay"              },
  { letter: "I", s1: "France",    s2: "Senegal",               s3: "Iraq",          s4: "Norway"               },
  { letter: "J", s1: "Argentina", s2: "Algeria",               s3: "Austria",       s4: "Jordan"               },
  { letter: "K", s1: "Portugal",  s2: "DR Congo",              s3: "Uzbekistan",    s4: "Colombia"             },
  { letter: "L", s1: "England",   s2: "Croatia",               s3: "Ghana",         s4: "Panama"               },
];

// Base ELO ratings (misterclaude variant — no adjustments)
export const BASE_ELO: Record<string, number> = {
  "Mexico": 1818, "South Africa": 1515, "South Korea": 1722, "Czech Republic": 1691,
  "Canada": 1730, "Bosnia and Herzegovina": 1598, "Qatar": 1468, "Switzerland": 1837,
  "Brazil": 1919, "Morocco": 1807, "Haiti": 1505, "Scotland": 1718,
  "United States": 1674, "Paraguay": 1810, "Australia": 1735, "Turkey": 1858,
  "Germany": 1891, "Curaçao": 1236, "Ivory Coast": 1633, "Ecuador": 1892,
  "Netherlands": 1904, "Japan": 1821, "Sweden": 1704, "Tunisia": 1586,
  "Belgium": 1835, "Egypt": 1656, "Iran": 1730, "New Zealand": 1528,
  "Spain": 2071, "Cape Verde": 1531, "Saudi Arabia": 1550, "Uruguay": 1855,
  "France": 2008, "Senegal": 1777, "Iraq": 1608, "Norway": 1806,
  "Argentina": 2048, "Algeria": 1696, "Austria": 1785, "Jordan": 1616,
  "Portugal": 1914, "DR Congo": 1603, "Uzbekistan": 1673, "Colombia": 1907,
  "England": 1931, "Croatia": 1874, "Ghana": 1472, "Panama": 1690,
};

// ELO adjustments per variant (added on top of BASE_ELO)
export const ELO_ADJUSTMENTS: Record<string, Record<string, number>> = {
  misterclaude: {},
  gemaldini: {
    "France": 200, "Spain": 200, "Portugal": 200, "England": 200, "Norway": 200,
  },
  dav_gpo: {
    "Argentina": 200, "Colombia": 200, "Ecuador": 200, "Paraguay": 200,
    "Uruguay": 150, "Mexico": 150, "Brazil": 100, "Panama": 50,
  },
};

export function getElo(team: string, variant: string): number {
  const base = BASE_ELO[team] ?? 1500;
  const adj = ELO_ADJUSTMENTS[variant]?.[team] ?? 0;
  return base + adj;
}

// ISO 3166-1 alpha-2 codes for flag-icons CSS library
export const TEAM_ISO: Record<string, string> = {
  "Argentina": "ar", "Australia": "au", "Austria": "at", "Algeria": "dz",
  "Belgium": "be", "Bosnia and Herzegovina": "ba", "Brazil": "br",
  "Canada": "ca", "Cape Verde": "cv", "Colombia": "co", "Croatia": "hr",
  "Czech Republic": "cz", "Curaçao": "cw", "DR Congo": "cd",
  "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng", "France": "fr",
  "Germany": "de", "Ghana": "gh", "Haiti": "ht", "Iran": "ir",
  "Iraq": "iq", "Ivory Coast": "ci", "Japan": "jp", "Jordan": "jo",
  "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl", "New Zealand": "nz",
  "Norway": "no", "Panama": "pa", "Paraguay": "py", "Portugal": "pt",
  "Qatar": "qa", "Saudi Arabia": "sa", "Scotland": "gb-sct",
  "Senegal": "sn", "South Africa": "za", "South Korea": "kr",
  "Spain": "es", "Sweden": "se", "Switzerland": "ch", "Tunisia": "tn",
  "Turkey": "tr", "United States": "us", "Uruguay": "uy", "Uzbekistan": "uz",
};
