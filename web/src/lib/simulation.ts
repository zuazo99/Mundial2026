/**
 * Browser simulation engine — TypeScript port of the Python simulation logic.
 *
 * Key implementation note: Python's get_multiplier() has dead code in lines 67–105
 * that is unconditionally overwritten by mult1=1.3/mult2=1.3 in line 107.
 * This port implements ONLY the actually-executed second block.
 */

import type { Variant, MatchResult } from '../data/predictions';
import type { XGEntry } from '../data/xg';
import type { TeamStats } from '../data/teamStats';
import { GROUPS, getElo } from '../data/groups';
import { mejoresTerceros } from '../data/mejoresTerceros';
import { computeGroupStandings, getBestThirds, type GroupStandings } from './computeStandings';
import { knockoutXg } from '../data/knockoutXg';
import { rho } from '../data/rho';

const NUM_ITERATIONS = 30;

// ─── Dixon-Coles correction ───────────────────────────────────────────────────
// tau reshapes the simulated score distribution toward the DC-corrected joint
// pmf (rho < 0 → draws and 1-1 weighted up, 1-0/0-1 down). Applied as an
// importance weight on the 90' regulation score; non low-scoring cells = 1.
function dcTau(g1: number, g2: number, xg1: number, xg2: number): number {
  if (rho === 0) return 1;
  if (g1 === 0 && g2 === 0) return Math.max(0, 1 - xg1 * xg2 * rho);
  if (g1 === 1 && g2 === 0) return Math.max(0, 1 + xg2 * rho);
  if (g1 === 0 && g2 === 1) return Math.max(0, 1 + xg1 * rho);
  if (g1 === 1 && g2 === 1) return Math.max(0, 1 - rho);
  return 1;
}

// ─── Multiplier logic (port of the actually-executed code in clases_simulacion.py) ───

function getMultiplier(
  minute: number,
  g1: number,
  g2: number,
  betterTeam: 1 | 2,
): [number, number] {
  let m1 = 1.3, m2 = 1.3;
  const diff = g1 - g2;
  if (diff > 0) {
    m2 = minute > 80 ? 1.7 : 1.4;
  } else if (diff < 0) {
    m1 = minute > 80 ? 1.7 : 1.4;
  } else if (minute > 45) {
    if (betterTeam === 1) m1 = 1.6; else m2 = 1.6;
  }
  return [m1, m2];
}

// ─── Single match iteration ───────────────────────────────────────────────────

function playMinutes(
  xg1: number,
  xg2: number,
  minutes: number,
  betterTeam: 1 | 2,
  startG1 = 0,
  startG2 = 0,
): [number, number] {
  const xg1min = xg1 / 90;
  const xg2min = xg2 / 90;
  let g1 = startG1, g2 = startG2;
  for (let m = 0; m < minutes; m++) {
    const [mult1, mult2] = getMultiplier(m, g1, g2, betterTeam);
    if (Math.random() < xg1min * mult1) g1++;
    if (Math.random() < xg2min * mult2) g2++;
  }
  return [g1, g2];
}

function simulatePenalties(): [number, number] {
  const prob = 0.75;
  let p1 = 0, p2 = 0;
  for (let i = 0; i < 5; i++) {
    if (Math.random() < prob) p1++;
    if (p1 + (4 - i) < p2 || p1 > (4 - i) + p2) return [p1, p2];
    if (Math.random() < prob) p2++;
    if (p1 + (4 - i) < p2 || p1 > (4 - i) + p2) return [p1, p2];
  }
  while (p1 === p2) {
    if (Math.random() < prob) p1++;
    if (Math.random() < prob) p2++;
  }
  return [p1, p2];
}

/** Returns [finalG1, finalG2, regulationG1, regulationG2]. */
function simulateOnce(
  xg1: number, xg2: number, ko: boolean, betterTeam: 1 | 2
): [number, number, number, number] {
  let [g1, g2] = playMinutes(xg1, xg2, 95, betterTeam);
  const [reg1, reg2] = [g1, g2];
  if (ko && g1 === g2) {
    [g1, g2] = playMinutes(xg1, xg2, 30, betterTeam, g1, g2);
    if (g1 === g2) {
      const [p1, p2] = simulatePenalties();
      if (p1 > p2) g1++; else g2++;
    }
  }
  return [g1, g2, reg1, reg2];
}

/** Runs NUM_ITERATIONS simulations and returns the Dixon-Coles weighted modal result. */
function simulateMatch(
  xg1: number, xg2: number, ko: boolean, team1Elo: number, team2Elo: number
): [number, number] {
  const betterTeam: 1 | 2 = team1Elo >= team2Elo ? 1 : 2;
  const freq = new Map<string, number>();
  for (let i = 0; i < NUM_ITERATIONS; i++) {
    const [g1, g2, reg1, reg2] = simulateOnce(xg1, xg2, ko, betterTeam);
    // Importance-weight each sample by the DC tau of its regulation score.
    const w = dcTau(reg1, reg2, xg1, xg2);
    const key = `${g1}-${g2}`;
    freq.set(key, (freq.get(key) ?? 0) + w);
  }
  let best = "0-0", bestCount = -1;
  for (const [k, v] of freq) {
    if (v > bestCount) { best = k; bestCount = v; }
  }
  const [g1, g2] = best.split("-").map(Number);
  return [g1, g2];
}

// ─── XG for knockout stage — pre-computed lookup table ───────────────────────
// knockoutXg contains XG predictions from the real XGBoost model for every
// possible matchup (48×47 ordered pairs × 3 variants), generated at build time
// by scripts/export_data.py. Falls back to ELO formula for unknown teams.

function eloToXG(eloTeam: number, eloOpponent: number): number {
  const winProb = 1 / (1 + Math.pow(10, -(eloTeam - eloOpponent) / 400));
  return Math.max(0.4, 1.15 + (winProb - 0.5) * 1.6);
}

function getKnockoutXG(team1: string, team2: string, variant: Variant): { xg1: number; xg2: number } {
  const xg1 = knockoutXg[variant]?.[team1]?.[team2]
    ?? eloToXG(getElo(team1, variant), getElo(team2, variant));
  const xg2 = knockoutXg[variant]?.[team2]?.[team1]
    ?? eloToXG(getElo(team2, variant), getElo(team1, variant));
  return { xg1, xg2 };
}

// ─── XG lookup from group stage CSV data ─────────────────────────────────────

function lookupXG(entries: XGEntry[], team: string): number {
  return entries.find(e => e.team === team)?.xg ?? 1.2;
}

// ─── Group stage simulation ───────────────────────────────────────────────────

export interface SimMatchResult extends MatchResult {
  xg1: number;
  xg2: number;
}

export interface SimGroupResult {
  letter: string;
  matches: SimMatchResult[];
  standings: GroupStandings;
}

function simulateGroups(
  xgData: { J1: XGEntry[]; J2: XGEntry[]; J3: XGEntry[] },
  variant: Variant,
): SimGroupResult[] {
  return GROUPS.map((g) => {
    const fixtures: Array<{ team1: string; team2: string; j: "J1" | "J2" | "J3" }> = [
      { team1: g.s1, team2: g.s2, j: "J1" }, { team1: g.s3, team2: g.s4, j: "J1" },
      { team1: g.s1, team2: g.s3, j: "J2" }, { team1: g.s4, team2: g.s2, j: "J2" },
      { team1: g.s4, team2: g.s1, j: "J3" }, { team1: g.s2, team2: g.s3, j: "J3" },
    ];

    const matches: SimMatchResult[] = fixtures.map(({ team1, team2, j }) => {
      const xg1 = lookupXG(xgData[j], team1);
      const xg2 = lookupXG(xgData[j], team2);
      const elo1 = getElo(team1, variant);
      const elo2 = getElo(team2, variant);
      const [score1, score2] = simulateMatch(xg1, xg2, false, elo1, elo2);
      return { team1, score1, team2, score2, xg1, xg2 };
    });

    const standings = computeGroupStandings(
      g,
      matches.map(m => ({ team1: m.team1, score1: m.score1, team2: m.team2, score2: m.score2 }))
    );
    return { letter: g.letter, matches, standings };
  });
}

// ─── Knockout bracket assembly ────────────────────────────────────────────────

export interface KnockoutResult {
  round: string;
  label: string;
  matches: SimMatchResult[];
}

function simulateKnockoutMatch(
  team1: string, team2: string,
  variant: Variant,
  xg1: number, xg2: number,
): SimMatchResult {
  const elo1 = getElo(team1, variant);
  const elo2 = getElo(team2, variant);
  const [score1, score2] = simulateMatch(xg1, xg2, true, elo1, elo2);
  return { team1, score1, team2, score2, xg1, xg2 };
}

// ─── Full tournament simulation ───────────────────────────────────────────────

export interface FullSimResult {
  groups: SimGroupResult[];
  knockouts: KnockoutResult[];
  champion: string;
  runnerUp: string;
  thirdPlace: string;
}

export function simulateTournament(
  variant: Variant,
  xgData: { J1: XGEntry[]; J2: XGEntry[]; J3: XGEntry[] },
  stats: Record<string, TeamStats>,
  onProgress?: (stage: string, pct: number) => void,
): FullSimResult {
  // 1. Group stage (local, XG pre-calculado del modelo Python)
  onProgress?.("Fase de grupos", 0);
  const groups = simulateGroups(xgData, variant);
  onProgress?.("Fase de grupos", 100);

  // 2. Determine qualifiers
  const groupWinners  = groups.map(g => g.standings.teams[0].team);
  const groupRunners  = groups.map(g => g.standings.teams[1].team);
  const allStandings  = groups.map(g => g.standings);
  const bestThirdsObj = getBestThirds(allStandings);
  const bestThirds    = bestThirdsObj.map(t => t.team);

  // 3. Build the R32 bracket using mejoresTerceros lookup
  const thirdLetters = [...bestThirdsObj.map(t => {
    const gIdx = groups.findIndex(g => g.standings.teams[2].team === t.team);
    return GROUPS[gIdx].letter;
  })].sort().join("");

  const thirdMap: Record<string, string> = {};
  const lookup = mejoresTerceros[thirdLetters];
  if (lookup) {
    for (const [slot, groupLetter] of Object.entries(lookup)) {
      const letter = (groupLetter as string).slice(1); // "3E" → "E"
      const gIdx = groups.findIndex(g => g.letter === letter);
      if (gIdx >= 0) thirdMap[slot] = groups[gIdx].standings.teams[2].team;
    }
  } else {
    const slotOrder = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"];
    slotOrder.forEach((slot, i) => { thirdMap[slot] = bestThirds[i] ?? "TBD"; });
  }

  // R32 fixture order mirrors Python's create_first_round() in clases_simulacion.py
  const A1=groupWinners[0],A2=groupRunners[0],B1=groupWinners[1],B2=groupRunners[1];
  const C1=groupWinners[2],C2=groupRunners[2],D1=groupWinners[3],D2=groupRunners[3];
  const E1=groupWinners[4],E2=groupRunners[4],F1=groupWinners[5],F2=groupRunners[5];
  const G1=groupWinners[6],G2=groupRunners[6],H1=groupWinners[7],H2=groupRunners[7];
  const I1=groupWinners[8],I2=groupRunners[8],J1=groupWinners[9],J2=groupRunners[9];
  const K1=groupWinners[10],K2=groupRunners[10],L1=groupWinners[11],L2=groupRunners[11];

  const r32Fixtures: [string, string][] = [
    [A2, B2], [C1, F2], [E1, thirdMap["1E"] ?? bestThirds[0]], [F1, C2],
    [E2, I2], [I1, thirdMap["1I"] ?? bestThirds[1]], [A1, thirdMap["1A"] ?? bestThirds[2]],
    [L1, thirdMap["1L"] ?? bestThirds[3]], [G1, thirdMap["1G"] ?? bestThirds[4]],
    [D1, thirdMap["1D"] ?? bestThirds[5]], [H1, J2], [K2, L2],
    [B1, thirdMap["1B"] ?? bestThirds[6]], [D2, G2], [J1, H2],
    [K1, thirdMap["1K"] ?? bestThirds[7]],
  ];

  // 4. Knockout rounds — XG del lookup pre-calculado (modelo Python real), Poisson local
  onProgress?.("Ronda de 32", 0);
  const r32Matches = r32Fixtures.map(([t1, t2]) => {
    const { xg1, xg2 } = getKnockoutXG(t1, t2, variant);
    return simulateKnockoutMatch(t1, t2, variant, xg1, xg2);
  });
  const r32Winners = r32Matches.map(m => m.score1 > m.score2 ? m.team1 : m.team2);
  onProgress?.("Ronda de 32", 100);

  const s16Fixtures: [string, string][] = [
    [r32Winners[0], r32Winners[3]], [r32Winners[2], r32Winners[5]],
    [r32Winners[1], r32Winners[4]], [r32Winners[6], r32Winners[7]],
    [r32Winners[10], r32Winners[11]], [r32Winners[9], r32Winners[8]],
    [r32Winners[14], r32Winners[13]], [r32Winners[12], r32Winners[15]],
  ];
  onProgress?.("Octavos de Final", 0);
  const s16Matches = s16Fixtures.map(([t1, t2]) => {
    const { xg1, xg2 } = getKnockoutXG(t1, t2, variant);
    return simulateKnockoutMatch(t1, t2, variant, xg1, xg2);
  });
  const s16Winners = s16Matches.map(m => m.score1 > m.score2 ? m.team1 : m.team2);
  onProgress?.("Octavos de Final", 100);

  const e8Fixtures: [string, string][] = [
    [s16Winners[0], s16Winners[1]], [s16Winners[4], s16Winners[5]],
    [s16Winners[2], s16Winners[3]], [s16Winners[6], s16Winners[7]],
  ];
  onProgress?.("Cuartos de Final", 0);
  const e8Matches = e8Fixtures.map(([t1, t2]) => {
    const { xg1, xg2 } = getKnockoutXG(t1, t2, variant);
    return simulateKnockoutMatch(t1, t2, variant, xg1, xg2);
  });
  const e8Winners = e8Matches.map(m => m.score1 > m.score2 ? m.team1 : m.team2);
  onProgress?.("Cuartos de Final", 100);

  onProgress?.("Semifinales", 0);
  const semiMatches = [
    (() => { const { xg1, xg2 } = getKnockoutXG(e8Winners[0], e8Winners[1], variant); return simulateKnockoutMatch(e8Winners[0], e8Winners[1], variant, xg1, xg2); })(),
    (() => { const { xg1, xg2 } = getKnockoutXG(e8Winners[2], e8Winners[3], variant); return simulateKnockoutMatch(e8Winners[2], e8Winners[3], variant, xg1, xg2); })(),
  ];
  const semiWinners = semiMatches.map(m => m.score1 > m.score2 ? m.team1 : m.team2);
  const semiLosers  = semiMatches.map(m => m.score1 > m.score2 ? m.team2 : m.team1);
  onProgress?.("Semifinales", 100);

  onProgress?.("Final", 0);
  const { xg1: txg1, xg2: txg2 } = getKnockoutXG(semiLosers[0], semiLosers[1], variant);
  const thirdMatch = simulateKnockoutMatch(semiLosers[0], semiLosers[1], variant, txg1, txg2);
  const { xg1: fxg1, xg2: fxg2 } = getKnockoutXG(semiWinners[0], semiWinners[1], variant);
  const finalMatch  = simulateKnockoutMatch(semiWinners[0], semiWinners[1], variant, fxg1, fxg2);
  onProgress?.("Final", 100);

  const champion   = finalMatch.score1 > finalMatch.score2 ? finalMatch.team1 : finalMatch.team2;
  const runnerUp   = finalMatch.score1 > finalMatch.score2 ? finalMatch.team2 : finalMatch.team1;
  const thirdPlace = thirdMatch.score1 > thirdMatch.score2 ? thirdMatch.team1 : thirdMatch.team2;

  return {
    groups,
    knockouts: [
      { round: "r32",     label: "Ronda de 32",       matches: r32Matches  },
      { round: "sweet16", label: "Octavos de Final",   matches: s16Matches  },
      { round: "elite8",  label: "Cuartos de Final",   matches: e8Matches   },
      { round: "semis",   label: "Semifinales",        matches: semiMatches },
      { round: "third",   label: "3er y 4to Puesto",   matches: [thirdMatch] },
      { round: "final",   label: "Final",              matches: [finalMatch] },
    ],
    champion,
    runnerUp,
    thirdPlace,
  };
}
