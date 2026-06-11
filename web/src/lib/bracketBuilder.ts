// Converts flat knockout result rows into a structured bracket for display.
// Row ordering follows the Python Knockouts class fixture creation order.

import type { MatchResult, TournamentRounds } from '../data/predictions';

export interface BracketMatch {
  id: string;
  team1: string; score1: number;
  team2: string; score2: number;
  winner: string;
  isPenaltyWin: boolean;
}

export interface BracketRound {
  name: string;
  label: string;
  matches: BracketMatch[];
}

function toBracketMatch(m: MatchResult, id: string): BracketMatch {
  const winner = m.score1 > m.score2 ? m.team1 : m.team2;
  // Penalty wins: detected by 1-goal difference in knockout (heuristic — exact P tracking
  // would require modifying the Python export; this covers the common case)
  const isPenaltyWin = Math.abs(m.score1 - m.score2) === 1 &&
    (m.score1 + m.score2) >= 3;
  return { id, team1: m.team1, score1: m.score1, team2: m.team2, score2: m.score2, winner, isPenaltyWin };
}

export function buildBracket(rounds: TournamentRounds): BracketRound[] {
  const make = (rows: MatchResult[], roundKey: string, label: string): BracketRound => ({
    name: roundKey,
    label,
    matches: rows.map((m, i) => toBracketMatch(m, `${roundKey}-${i}`)),
  });

  return [
    make(rounds.r32,     "r32",     "Ronda de 32"),
    make(rounds.sweet16, "sweet16", "Octavos de Final"),
    make(rounds.elite8,  "elite8",  "Cuartos de Final"),
    make(rounds.semis,   "semis",   "Semifinales"),
    make(rounds.third,   "third",   "3er y 4to Puesto"),
    make(rounds.final,   "final",   "Final"),
  ];
}

/** Returns the champion team name from a rounds object. */
export function getChampion(rounds: TournamentRounds): string {
  const f = rounds.final[0];
  return f.score1 > f.score2 ? f.team1 : f.team2;
}

/** Returns the runner-up team name. */
export function getRunnerUp(rounds: TournamentRounds): string {
  const f = rounds.final[0];
  return f.score1 > f.score2 ? f.team2 : f.team1;
}

/** Returns all teams that appear in the bracket in their furthest round. */
export function getTeamJourney(rounds: TournamentRounds, team: string): string[] {
  const roundOrder = ["r32", "sweet16", "elite8", "semis", "final"] as const;
  const reached: string[] = [];
  for (const round of roundOrder) {
    const matches = rounds[round];
    for (const m of matches) {
      if (m.team1 === team || m.team2 === team) {
        reached.push(round);
        break;
      }
    }
  }
  return reached;
}
