import clickhouse from '../db/clickhouse';
import { getBotBalance } from './bot-state';
import type { Trade, DailyPnl, StrategyPerformance, AuditEvent, AdvancedMetrics, SkipMetrics } from '../types/trade';

// ClickHouse JSONEachRow returns ALL numeric columns as strings.
// This helper converts known numeric fields to actual numbers so
// downstream code can do arithmetic safely without `Number()` wrappers.
const TRADE_NUMERIC_FIELDS = new Set([
  'entry_price', 'exit_price', 'position_size', 'pnl', 'fee_cost',
  'window_minute', 'cum_return_pct', 'confidence',
  'clob_entry_price', 'sigmoid_entry_price', 'bet_to_win_ratio',
]);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function coerceTradeNumerics<T>(row: T): T {
  const out = { ...row } as any;
  for (const key of TRADE_NUMERIC_FIELDS) {
    if (key in out) {
      const val = Number(out[key]);
      out[key] = Number.isFinite(val) ? val : 0;
    }
  }
  return out as T;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function coerceRows<T>(rows: T[], numericFields?: Set<string>): T[] {
  if (!numericFields) return rows.map(coerceTradeNumerics);
  return rows.map(row => {
    const out = { ...row } as any;
    for (const key of numericFields) {
      if (key in out) {
        const val = Number(out[key]);
        out[key] = Number.isFinite(val) ? val : 0;
      }
    }
    return out as T;
  });
}

export async function getRecentTrades(limit: number = 10, strategy?: string, offset: number = 0): Promise<Trade[]> {
  const where = strategy ? `WHERE strategy = {strategy:String}` : '';
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.trades ${where} ORDER BY entry_time DESC LIMIT {limit:UInt32} OFFSET {offset:UInt32}`,
    query_params: { limit, offset, ...(strategy ? { strategy } : {}) },
    format: 'JSONEachRow',
  });
  return coerceRows(await result.json<Trade>());
}

export async function getTradeCount(strategy?: string): Promise<number> {
  const where = strategy ? `WHERE strategy = {strategy:String}` : '';
  const result = await clickhouse.query({
    query: `SELECT count() AS cnt FROM mvhe.trades ${where}`,
    query_params: strategy ? { strategy } : {},
    format: 'JSONEachRow',
  });
  const rows = await result.json<{ cnt: string }>();
  return rows.length > 0 ? Number(rows[0].cnt) : 0;
}

export async function getTradesInRange(start: string, end: string, strategy?: string): Promise<Trade[]> {
  const stratWhere = strategy ? ` AND strategy = {strategy:String}` : '';
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.trades WHERE entry_time >= {start:String} AND entry_time <= {end:String}${stratWhere} ORDER BY entry_time DESC`,
    query_params: { start, end, ...(strategy ? { strategy } : {}) },
    format: 'JSONEachRow',
  });
  return coerceRows(await result.json<Trade>());
}

const DAILY_NUMERIC = new Set(['trade_count', 'total_pnl', 'total_fees', 'win_count']);

export async function getDailyPnl(date: string): Promise<DailyPnl[]> {
  const result = await clickhouse.query({
    query: `
      SELECT
        strategy,
        count() AS trade_count,
        sum(pnl) AS total_pnl,
        sum(fee_cost) AS total_fees,
        countIf(pnl > 0) AS win_count
      FROM mvhe.trades
      WHERE toDate(entry_time) = {date:String}
      GROUP BY strategy
    `,
    query_params: { date },
    format: 'JSONEachRow',
  });
  return coerceRows(await result.json<DailyPnl>(), DAILY_NUMERIC);
}

const PERF_NUMERIC = new Set([
  'trade_count', 'win_count', 'total_pnl', 'total_fees', 'avg_pnl',
  'best_trade', 'worst_trade', 'avg_position_size', 'avg_confidence',
]);

export async function getStrategyPerformance(strategy: string): Promise<StrategyPerformance | null> {
  const result = await clickhouse.query({
    query: `
      SELECT
        strategy,
        count() AS trade_count,
        countIf(pnl > 0) AS win_count,
        sum(pnl) AS total_pnl,
        sum(fee_cost) AS total_fees,
        avg(pnl) AS avg_pnl,
        max(pnl) AS best_trade,
        min(pnl) AS worst_trade,
        avg(position_size) AS avg_position_size,
        avg(confidence) AS avg_confidence
      FROM mvhe.trades
      WHERE strategy = {strategy:String}
      GROUP BY strategy
    `,
    query_params: { strategy },
    format: 'JSONEachRow',
  });
  const rows = coerceRows(await result.json<StrategyPerformance>(), PERF_NUMERIC);
  return rows.length > 0 ? rows[0] : null;
}

const EQ_NUMERIC = new Set(['cumulative_pnl']);

export async function getEquityCurve(strategy?: string): Promise<Array<{ time: string; cumulative_pnl: number }>> {
  const where = strategy ? `WHERE strategy = {strategy:String}` : '';
  const result = await clickhouse.query({
    query: `
      SELECT
        entry_time AS time,
        sum(pnl) OVER (ORDER BY entry_time) AS cumulative_pnl
      FROM mvhe.trades
      ${where}
      ORDER BY entry_time ASC
    `,
    query_params: strategy ? { strategy } : {},
    format: 'JSONEachRow',
  });
  return coerceRows(await result.json<{ time: string; cumulative_pnl: number }>(), EQ_NUMERIC);
}

export async function getAdvancedMetrics(strategy: string): Promise<AdvancedMetrics | null> {
  // Sortino uses target=0 (not mean) — downside_dev is sqrt of mean squared negative PnL
  const result = await clickhouse.query({
    query: `
      SELECT
        count() AS total_trades,
        sumIf(pnl, pnl > 0) AS gross_profit,
        abs(sumIf(pnl, pnl < 0)) AS gross_loss,
        avg(pnl) AS avg_pnl,
        stddevPop(pnl) AS pnl_stddev,
        sqrt(sumIf(pow(pnl, 2), pnl < 0) / greatest(countIf(pnl < 0), 1)) AS downside_dev,
        avg(dateDiff('minute', entry_time, exit_time)) AS avg_hold_time_minutes,
        countIf(pnl > 0) AS win_count,
        min(toDate(entry_time)) AS first_date,
        max(toDate(entry_time)) AS last_date
      FROM mvhe.trades
      WHERE strategy = {strategy:String}
    `,
    query_params: { strategy },
    format: 'JSONEachRow',
  });
  type RawRow = {
    total_trades: string;
    gross_profit: string;
    gross_loss: string;
    avg_pnl: string;
    pnl_stddev: string;
    downside_dev: string;
    avg_hold_time_minutes: string;
    win_count: string;
    first_date: string;
    last_date: string;
  };
  const rows = await result.json<RawRow>();
  if (rows.length === 0 || Number(rows[0].total_trades) === 0) return null;
  const r = rows[0];

  const grossProfit = Number(r.gross_profit);
  const grossLoss = Number(r.gross_loss);
  const avgPnl = Number(r.avg_pnl);
  const totalTrades = Number(r.total_trades);
  const winCount = Number(r.win_count);

  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
  const stddev = Number(r.pnl_stddev);
  const sharpe = stddev > 0 ? avgPnl / stddev : avgPnl > 0 ? Infinity : 0;
  const downsideDev = Number(r.downside_dev);
  const sortino = downsideDev > 0 ? avgPnl / downsideDev : avgPnl > 0 ? Infinity : 0;
  const winRate = totalTrades > 0 ? winCount / totalTrades : 0;
  const avgWin = winCount > 0 ? grossProfit / winCount : 0;
  const lossCount = totalTrades - winCount;
  const avgLoss = lossCount > 0 ? grossLoss / lossCount : 0;
  const expectancy = winRate * avgWin - (1 - winRate) * avgLoss;

  // Days span for avg trades per day
  const firstDate = new Date(r.first_date);
  const lastDate = new Date(r.last_date);
  const daySpan = Math.max(1, Math.ceil((lastDate.getTime() - firstDate.getTime()) / 86400000) + 1);
  const avgTradesPerDay = totalTrades / daySpan;

  // Max drawdown via equity curve
  const eqResult = await clickhouse.query({
    query: `
      SELECT sum(pnl) OVER (ORDER BY entry_time) AS cumulative_pnl
      FROM mvhe.trades
      WHERE strategy = {strategy:String}
      ORDER BY entry_time ASC
    `,
    query_params: { strategy },
    format: 'JSONEachRow',
  });
  const eqRows = coerceRows(await eqResult.json<{ cumulative_pnl: number }>(), EQ_NUMERIC);
  // Use equity (initial_balance + cumulative_pnl) to compute drawdown percentage
  // Read real initial_balance from Redis (live mode), fall back to 100 for live
  const bal = await getBotBalance();
  const initialBalance = bal?.initial_balance && Number(bal.initial_balance) > 0
    ? Number(bal.initial_balance) : 100;
  let peakEquity = initialBalance;
  let maxDrawdown = 0;
  for (const row of eqRows) {
    const equity = initialBalance + row.cumulative_pnl;
    if (equity > peakEquity) peakEquity = equity;
    const dd = peakEquity > 0 ? (peakEquity - equity) / peakEquity : 0;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  return {
    profit_factor: Number.isFinite(profitFactor) ? Number(profitFactor.toFixed(2)) : 9999,
    sharpe_ratio: Number.isFinite(sharpe) ? Number(sharpe.toFixed(2)) : 9999,
    sortino_ratio: Number.isFinite(sortino) ? Number(sortino.toFixed(2)) : 9999,
    max_drawdown_pct: Number((maxDrawdown * 100).toFixed(2)),
    avg_hold_time_minutes: Number(Number(r.avg_hold_time_minutes || 0).toFixed(1)),
    expectancy: Number(expectancy.toFixed(4)),
    total_trades: totalTrades,
    avg_trades_per_day: Number(avgTradesPerDay.toFixed(1)),
  };
}

export async function getTradesWithSignals(strategy: string): Promise<Array<Trade & { signal_details: string }>> {
  const result = await clickhouse.query({
    query: `SELECT *, signal_details FROM mvhe.trades WHERE strategy = {strategy:String} AND signal_details != '' ORDER BY entry_time DESC LIMIT 1000`,
    query_params: { strategy },
    format: 'JSONEachRow',
  });
  return coerceRows(await result.json<Trade & { signal_details: string }>());
}

export async function getAuditEvents(limit: number = 50): Promise<AuditEvent[]> {
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.audit_events ORDER BY timestamp DESC LIMIT {limit:UInt32}`,
    query_params: { limit },
    format: 'JSONEachRow',
  });
  return result.json<AuditEvent>();
}

export async function getRecentSignalActivity(limit: number = 20): Promise<SignalActivityRow[]> {
  const result = await clickhouse.query({
    query: `
      SELECT eval_id, timestamp, market_id, minute, outcome, reason,
             direction, confidence, votes_yes, votes_no, votes_neutral, detail
      FROM mvhe.signal_evaluations
      ORDER BY timestamp DESC
      LIMIT {limit:UInt32}
    `,
    query_params: { limit },
    format: 'JSONEachRow',
  });
  return result.json<SignalActivityRow>();
}

export interface SignalActivityRow {
  eval_id: string;
  timestamp: string;
  market_id: string;
  minute: number;
  outcome: string;
  reason: string;
  direction: string;
  confidence: number;
  votes_yes: number;
  votes_no: number;
  votes_neutral: number;
  detail: string;
}

export async function getSkipMetrics(strategy: string, days: number = 7): Promise<SkipMetrics> {
  // Query 1: Summary + reasons breakdown
  const summaryResult = await clickhouse.query({
    query: `
      SELECT outcome, reason, count() as cnt
      FROM mvhe.signal_evaluations
      WHERE strategy = {strategy:String}
        AND timestamp >= now() - INTERVAL {days:UInt32} DAY
      GROUP BY outcome, reason
    `,
    query_params: { strategy, days },
    format: 'JSONEachRow',
  });
  type SummaryRow = { outcome: string; reason: string; cnt: string };
  const summaryRows = await summaryResult.json<SummaryRow>();

  // Query 2: By minute breakdown
  const minuteResult = await clickhouse.query({
    query: `
      SELECT minute, outcome, count() as cnt
      FROM mvhe.signal_evaluations
      WHERE strategy = {strategy:String}
        AND timestamp >= now() - INTERVAL {days:UInt32} DAY
      GROUP BY minute, outcome
      ORDER BY minute
    `,
    query_params: { strategy, days },
    format: 'JSONEachRow',
  });
  type MinuteRow = { minute: string; outcome: string; cnt: string };
  const minuteRows = await minuteResult.json<MinuteRow>();

  // Assemble totals
  let totalSkips = 0;
  let totalEntries = 0;
  const reasonCounts = new Map<string, number>();

  for (const row of summaryRows) {
    const cnt = Number(row.cnt);
    if (row.outcome === 'skip') {
      totalSkips += cnt;
      reasonCounts.set(row.reason, (reasonCounts.get(row.reason) || 0) + cnt);
    } else if (row.outcome === 'entry') {
      totalEntries += cnt;
    }
  }

  const totalEvaluations = totalSkips + totalEntries;
  const skipRate = totalEvaluations > 0 ? (totalSkips / totalEvaluations) * 100 : 0;

  // Build reasons array
  const reasons = Array.from(reasonCounts.entries())
    .map(([reason, count]) => ({
      reason,
      count,
      pct: totalSkips > 0 ? Number(((count / totalSkips) * 100).toFixed(1)) : 0,
    }))
    .sort((a, b) => b.count - a.count);

  // Build by_minute array
  const minuteMap = new Map<number, { skips: number; entries: number }>();
  for (const row of minuteRows) {
    const m = Number(row.minute);
    const entry = minuteMap.get(m) || { skips: 0, entries: 0 };
    const cnt = Number(row.cnt);
    if (row.outcome === 'skip') entry.skips += cnt;
    else if (row.outcome === 'entry') entry.entries += cnt;
    minuteMap.set(m, entry);
  }

  const byMinute = Array.from(minuteMap.entries())
    .map(([minute, { skips, entries }]) => {
      const total = skips + entries;
      return {
        minute,
        skips,
        entries,
        skip_rate: total > 0 ? Number(((skips / total) * 100).toFixed(1)) : 0,
      };
    })
    .sort((a, b) => a.minute - b.minute);

  return {
    total_evaluations: totalEvaluations,
    total_skips: totalSkips,
    total_entries: totalEntries,
    skip_rate: Number(skipRate.toFixed(1)),
    reasons,
    by_minute: byMinute,
  };
}
