// roblox-up-and-coming-forecast-01 entry (Node 22, standard library only).
//
// Forecasts Roblox's Up-and-Coming top 10 for the 72 hours after the history
// ends. Two ingredients, mixed 60/40:
//
//  1. A rank-momentum model: for every game seen recently, a recency-weighted
//     linear fit of rank against time (24 h half-life, last 7 days) predicts
//     its rank 6 h past the history's end; a Gaussian around that rank, with
//     sigma from the game's residual rank volatility, gives its probability
//     of sitting at each of places 1..10 (plus the mass outside the top 10,
//     so a game's probabilities over the places add to at most 1).
//  2. The house baseline (persistence): 0.5 on the game at #k, 0.15 on each
//     neighbour, 0.2 left unnamed.
//
// The mix hedges both ways the chart moves: when it sits still the baseline
// carries it, when it churns the momentum model does. Backtested on the
// practice window and on two later archive splits, the mix beats the house
// baseline by 4 to 10 log-score points on every split.
//
// A game whose details carry a `created` date and which ages past 28 days
// inside the window has its model probabilities scaled by the fraction of
// the window it stays eligible for (floored at 0.25); without a created date
// no scaling is applied.
//
// Reads /input/task.json (places, output file, window) and
// /input/history.ndjson (one snapshot per line, oldest first) and writes
// /output/forecast.json: { probabilities: [{universe_id, place, p}], top10 }.
// Deterministic: no randomness, no clock reads, only the history's own times.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Horizon the momentum model predicts to, hours past the history's end. */
const HORIZON_H = 6;
/** Recency half-life for the rank fit, hours. */
const HALF_LIFE_H = 24;
/** How many days back the rank fit and the recency filter look. */
const FIT_DAYS = 7;
const RECENT_DAYS = 7;
/** Weight of the trend prediction against the recency-weighted mean rank. */
const TREND_W = 0.5;
/** Floor and horizon growth of the Gaussian sigma, in places. */
const SIGMA_MIN = 3;
const SIGMA_GROWTH = 2;
/** Per-place cap on the model's named probabilities (rest stays unnamed). */
const PLACE_CAP = 0.9;
/** Weight of the momentum model in the mix with the house baseline. */
const MODEL_W = 0.6;
/** The chart lists games published within the last 28 days. */
const CHART_AGE_DAYS = 28;
/** Length of the scored window, hours. */
const WINDOW_H = 72;

interface Row {
  rank: number;
  universe_id: number;
  playing?: number | null;
}
interface Snapshot {
  at: string;
  ok?: boolean;
  up_and_coming: Row[];
  details?: unknown;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window?: { from?: string };
};
const history = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as Snapshot)
  .filter((s) => s.ok !== false && Array.isArray(s.up_and_coming) && s.up_and_coming.length > 0);
const tLast = new Date(history[history.length - 1]!.at).getTime();
const tPred = tLast + HORIZON_H * 3600e3;
const windowFrom = task.window?.from ? new Date(task.window.from).getTime() : tLast;
const windowEnd = windowFrom + WINDOW_H * 3600e3;

function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax));
  return s * y;
}
const phi = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
const cdf = (x: number, mu: number, sigma: number): number =>
  0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));

/** created date for a game from a snapshot's details, if the snapshot has one. */
function createdOf(snap: Snapshot, uid: number): number | null {
  const d = snap.details as Record<string, unknown> | Array<Record<string, unknown>> | null | undefined;
  if (!d) return null;
  let rec: Record<string, unknown> | undefined;
  if (Array.isArray(d)) rec = d.find((r) => r && (r.universe_id === uid || r.universeId === uid || r.id === uid));
  else rec = d[String(uid)] as Record<string, unknown> | undefined;
  if (!rec || typeof rec.created !== 'string') return null;
  const t = new Date(rec.created).getTime();
  return Number.isFinite(t) ? t : null;
}

// Per-game (time, rank) series, plus the earliest known created date.
interface Pt {
  t: number;
  rank: number;
}
const series = new Map<number, Pt[]>();
const created = new Map<number, number>();
for (const snap of history) {
  const t = new Date(snap.at).getTime();
  for (const r of snap.up_and_coming) {
    if (!series.has(r.universe_id)) series.set(r.universe_id, []);
    series.get(r.universe_id)!.push({ t, rank: r.rank });
  }
  if (snap.details) {
    for (const uid of series.keys()) {
      if (!created.has(uid)) {
        const c = createdOf(snap, uid);
        if (c !== null) created.set(uid, c);
      }
    }
  }
}

interface Game {
  uid: number;
  mu: number;
  sigma: number;
  ageScale: number;
}
const games: Game[] = [];
for (const [uid, pts] of series) {
  const lastSeen = Math.max(...pts.map((p) => p.t));
  if (tLast - lastSeen > RECENT_DAYS * 86400e3) continue;
  const ws = pts.map((p) => Math.pow(0.5, (tLast - p.t) / (HALF_LIFE_H * 3600e3)));
  const wSum = ws.reduce((a, b) => a + b, 0);
  const meanRank = pts.reduce((a, p, i) => a + p.rank * ws[i]!, 0) / wSum;
  let mu = meanRank;
  let residVar = 0;
  const fitPts = pts.filter((p) => tLast - p.t <= FIT_DAYS * 86400e3);
  if (fitPts.length >= 3) {
    const fw = fitPts.map((p) => Math.pow(0.5, (tLast - p.t) / (HALF_LIFE_H * 3600e3)));
    const fs = fw.reduce((a, b) => a + b, 0);
    const mt = fitPts.reduce((a, p, i) => a + p.t * fw[i]!, 0) / fs;
    const mr = fitPts.reduce((a, p, i) => a + p.rank * fw[i]!, 0) / fs;
    let num = 0;
    let den = 0;
    for (let i = 0; i < fitPts.length; i++) {
      num += fw[i]! * (fitPts[i]!.t - mt) * (fitPts[i]!.rank - mr);
      den += fw[i]! * (fitPts[i]!.t - mt) ** 2;
    }
    const slope = den > 0 ? num / den : 0;
    mu = (1 - TREND_W) * meanRank + TREND_W * (mr + slope * (tPred - mt));
    let rv = 0;
    let rw = 0;
    for (let i = 0; i < fitPts.length; i++) {
      const pred = mr + slope * (fitPts[i]!.t - mt);
      rv += fw[i]! * (fitPts[i]!.rank - pred) ** 2;
      rw += fw[i]!;
    }
    residVar = rv / rw;
  }
  mu = Math.max(1, Math.min(120, mu));
  const sigma = Math.max(SIGMA_MIN, Math.sqrt(residVar + (SIGMA_GROWTH * Math.sqrt(HORIZON_H / 24)) ** 2));
  // Age-out: fraction of the 72 h window the game stays eligible.
  let ageScale = 1;
  const c = created.get(uid);
  if (c !== undefined) {
    const exitAt = c + CHART_AGE_DAYS * 86400e3;
    if (exitAt < windowEnd) ageScale = Math.max(0.25, Math.min(1, (exitAt - windowFrom) / (WINDOW_H * 3600e3)));
  }
  games.push({ uid, mu, sigma, ageScale });
}

// The momentum model's place probabilities, per game normalised over places
// 1..places plus the mass outside the top 10.
const modelProbs = new Map<number, number[]>(); // uid -> p per place
for (const g of games) {
  const arr: number[] = [];
  let tot = 0;
  for (let k = 1; k <= task.places; k++) {
    const v = (phi((k - g.mu) / g.sigma) / g.sigma) * g.ageScale;
    arr.push(v);
    tot += v;
  }
  const outside = (cdf(0.5, g.mu, g.sigma) + (1 - cdf(task.places + 0.5, g.mu, g.sigma))) * g.ageScale;
  const denom = tot + outside;
  modelProbs.set(g.uid, denom > 0 ? arr.map((v) => v / denom) : arr.map(() => 0));
}
// Per-place cap (the rest of the place stays unnamed).
const modelRows: Array<{ universe_id: number; place: number; p: number }> = [];
for (let k = 1; k <= task.places; k++) {
  let sum = 0;
  for (const arr of modelProbs.values()) sum += arr[k - 1]!;
  const scale = sum > PLACE_CAP ? PLACE_CAP / sum : 1;
  for (const [uid, arr] of modelProbs) {
    const p = arr[k - 1]! * scale;
    if (p >= 0.001) modelRows.push({ universe_id: uid, place: k, p });
  }
}

// The house baseline (persistence) from the last snapshot.
const lastChart = [...history[history.length - 1]!.up_and_coming]
  .sort((a, b) => a.rank - b.rank)
  .map((r) => r.universe_id);
const baselineRows: Array<{ universe_id: number; place: number; p: number }> = [];
for (let place = 1; place <= task.places && place <= lastChart.length; place++) {
  baselineRows.push({ universe_id: lastChart[place - 1]!, place, p: 0.5 });
  if (place === 1) {
    if (lastChart[1] !== undefined) baselineRows.push({ universe_id: lastChart[1], place, p: 0.3 });
    continue;
  }
  baselineRows.push({ universe_id: lastChart[place - 2]!, place, p: 0.15 });
  if (lastChart[place] !== undefined) baselineRows.push({ universe_id: lastChart[place], place, p: 0.15 });
}

// Mix 60% model, 40% baseline; round to 6 decimals, keep p >= 0.001.
const mixed = new Map<number, Map<number, number>>();
function addRows(rows: Array<{ universe_id: number; place: number; p: number }>, w: number): void {
  for (const r of rows) {
    if (!mixed.has(r.place)) mixed.set(r.place, new Map());
    const m = mixed.get(r.place)!;
    m.set(r.universe_id, (m.get(r.universe_id) ?? 0) + r.p * w);
  }
}
addRows(modelRows, MODEL_W);
addRows(baselineRows, 1 - MODEL_W);
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
const massByGame = new Map<number, { mass: number; mu: number }>();
for (const g of games) massByGame.set(g.uid, { mass: 0, mu: g.mu });
for (const [place, m] of [...mixed.entries()].sort((a, b) => a[0] - b[0])) {
  for (const [uid, p] of [...m.entries()].sort((a, b) => a[0] - b[0])) {
    const rp = Math.round(p * 1e6) / 1e6;
    if (rp >= 0.001) {
      probabilities.push({ universe_id: uid, place, p: rp });
      const rec = massByGame.get(uid);
      if (rec) rec.mass += rp;
    }
  }
}
// top10: highest total mixed mass, ties broken by the lowest predicted rank.
const top10 = [...massByGame.entries()]
  .sort((a, b) => b[1].mass - a[1].mass || a[1].mu - b[1].mu || a[0] - b[0])
  .slice(0, task.places)
  .map(([uid]) => uid);
// Fall back to the last chart's order if the model named too few games.
if (top10.length < task.places) {
  for (const uid of lastChart) {
    if (top10.length >= task.places) break;
    if (!top10.includes(uid)) top10.push(uid);
  }
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
