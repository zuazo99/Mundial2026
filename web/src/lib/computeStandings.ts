// Computes group standings from match results.
// Tie-break criteria mirror Python's tie_break() in clases_simulacion.py:
//   1. Total points  2. H2H points  3. H2H GD  4. H2H GF  5. Overall GD  6. Overall GF

import type { MatchResult } from '../data/predictions';
import { GROUPS, type GroupDef } from '../data/groups';

export interface TeamStanding {
  team: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  gf: number;
  gc: number;
  gd: number;
  pts: number;
  pos?: number;  // 1–4 after sorting
}

export interface GroupStandings {
  letter: string;
  teams: TeamStanding[];
  /** matches[j] = [match0, match1] for jornada j (0-indexed) */
  matchdays: MatchResult[][];
}

// Recursively orders a set of teams, mirroring Python's tie_break() in
// clases_simulacion.py. Head-to-head stats are recomputed over ONLY the teams
// still tied at the current level; whenever a criterion splits the set, each
// resulting subgroup is re-ranked from scratch (recursion), so H2H never leaks
// in results against teams outside the tie.
function rankTeams(tied: TeamStanding[], matches: MatchResult[]): TeamStanding[] {
  if (tied.length <= 1) return tied;

  const tiedNames = tied.map(t => t.team);
  const h2h: Record<string, { pts: number; gd: number; gf: number }> = {};
  for (const t of tiedNames) h2h[t] = { pts: 0, gd: 0, gf: 0 };

  for (const m of matches) {
    if (tiedNames.includes(m.team1) && tiedNames.includes(m.team2)) {
      h2h[m.team1].gf += m.score1;
      h2h[m.team2].gf += m.score2;
      h2h[m.team1].gd += (m.score1 - m.score2);
      h2h[m.team2].gd += (m.score2 - m.score1);
      if (m.score1 > m.score2) { h2h[m.team1].pts += 3; }
      else if (m.score1 < m.score2) { h2h[m.team2].pts += 3; }
      else { h2h[m.team1].pts += 1; h2h[m.team2].pts += 1; }
    }
  }

  // Criteria in priority order: total points, H2H points, H2H GD, H2H GF,
  // overall GD, overall GF.
  const criteria: ((t: TeamStanding) => number)[] = [
    t => t.pts,
    t => h2h[t.team].pts,
    t => h2h[t.team].gd,
    t => h2h[t.team].gf,
    t => t.gd,
    t => t.gf,
  ];

  for (const criterio of criteria) {
    const buckets = new Map<number, TeamStanding[]>();
    for (const t of tied) {
      const value = criterio(t);
      if (!buckets.has(value)) buckets.set(value, []);
      buckets.get(value)!.push(t);
    }

    if (buckets.size > 1) {
      const result: TeamStanding[] = [];
      for (const value of [...buckets.keys()].sort((x, y) => y - x)) {
        result.push(...rankTeams(buckets.get(value)!, matches));
      }
      return result;
    }
  }

  // Every criterion left them tied — preserve input order.
  return tied;
}

export function computeGroupStandings(
  groupDef: GroupDef,
  groupRows: MatchResult[],   // 6 rows in fixture order
): GroupStandings {
  // Initialise standings
  const teamsArr = [groupDef.s1, groupDef.s2, groupDef.s3, groupDef.s4];
  const st: Record<string, TeamStanding> = {};
  for (const t of teamsArr) {
    st[t] = { team: t, played: 0, won: 0, drawn: 0, lost: 0, gf: 0, gc: 0, gd: 0, pts: 0 };
  }

  const updateStanding = (t: string, gf: number, gc: number) => {
    const s = st[t];
    s.played++; s.gf += gf; s.gc += gc; s.gd += (gf - gc);
    if (gf > gc) { s.won++; s.pts += 3; }
    else if (gf === gc) { s.drawn++; s.pts += 1; }
    else { s.lost++; }
  };

  for (const m of groupRows) {
    updateStanding(m.team1, m.score1, m.score2);
    updateStanding(m.team2, m.score2, m.score1);
  }

  // Order with the recursive tie-break (mirrors Python's tie_break)
  const sorted = rankTeams(teamsArr.map(t => st[t]), groupRows);
  sorted.forEach((s, i) => { s.pos = i + 1; });

  // Split into matchdays: rows [0,1]=J1, [2,3]=J2, [4,5]=J3
  const matchdays = [
    groupRows.slice(0, 2),
    groupRows.slice(2, 4),
    groupRows.slice(4, 6),
  ];

  return {
    letter: groupDef.letter,
    teams: sorted,
    matchdays,
  };
}

/**
 * Compute standings for all 12 groups from the flat 72-row grupo array.
 * grupos[0..71]: groups in order A→L, 6 rows each.
 */
export function computeAllStandings(grupRows: MatchResult[]): GroupStandings[] {
  return GROUPS.map((g, i) =>
    computeGroupStandings(g, grupRows.slice(i * 6, i * 6 + 6))
  );
}

/**
 * Returns the best 8 third-place teams sorted by pts → gd → gf.
 */
export function getBestThirds(standings: GroupStandings[]): TeamStanding[] {
  const thirds = standings.map(g => g.teams[2]);
  return [...thirds]
    .sort((a, b) => {
      if (b.pts !== a.pts) return b.pts - a.pts;
      if (b.gd !== a.gd)  return b.gd - a.gd;
      return b.gf - a.gf;
    })
    .slice(0, 8);
}
