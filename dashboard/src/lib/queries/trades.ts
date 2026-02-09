import clickhouse from '../db/clickhouse';
import type { Trade, DailyPnl, StrategyPerformance, AuditEvent, AdvancedMetrics } from '../types/trade';

export async function getRecentTrades(limit: number = 10, strategy?: string): Promise<Trade[]> {
  const where = strategy ? `WHERE strategy = {strategy:String}` : '';
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.trades ${where} ORDER BY entry_time DESC LIMIT {limit:UInt32}`,
    query_params: { limit, ...(strategy ? { strategy } : {}) },
    format: 'JSONEachRow',
  });
  return result.json<Trade>();
}

export async function getTradesInRange(start: string, end: string): Promise<Trade[]> {
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.trades WHERE entry_time >= {start:String} AND entry_time <= {end:String} ORDER BY entry_time DESC`,
    query_params: { start, end },
    format: 'JSONEachRow',
  });
  return result.json<Trade>();
}

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
  return result.json<DailyPnl>();
}

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
  const rows = await result.json<StrategyPerformance>();
  return rows.length > 0 ? rows[0] : null;
}

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
  return result.json<{ time: string; cumulative_pnl: number }>();
}

export async function getAdvancedMetrics(strategy: string): Promise<AdvancedMetrics | null> {
  const result = await clickhouse.query({
    query: `
      SELECT
        count() AS total_trades,
        sumIf(pnl, pnl > 0) AS gross_profit,
        abs(sumIf(pnl, pnl < 0)) AS gross_loss,
        avg(pnl) AS avg_pnl,
        stddevPop(pnl) AS pnl_stddev,
        sqrt(sumIf(pow(pnl - avg_pnl_sub.m, 2), pnl < avg_pnl_sub.m) / greatest(countIf(pnl < avg_pnl_sub.m), 1)) AS downside_dev,
        avg(dateDiff('minute', entry_time, exit_time)) AS avg_hold_time_minutes,
        countIf(pnl > 0) AS win_count,
        min(toDate(entry_time)) AS first_date,
        max(toDate(entry_time)) AS last_date
      FROM mvhe.trades
      CROSS JOIN (SELECT avg(pnl) AS m FROM mvhe.trades WHERE strategy = {strategy:String}) AS avg_pnl_sub
      WHERE strategy = {strategy:String}
    `,
    query_params: { strategy },
    format: 'JSONEachRow',
  });
  type RawRow = {
    total_trades: number;
    gross_profit: number;
    gross_loss: number;
    avg_pnl: number;
    pnl_stddev: number;
    downside_dev: number;
    avg_hold_time_minutes: number;
    win_count: number;
    first_date: string;
    last_date: string;
  };
  const rows = await result.json<RawRow>();
  if (rows.length === 0 || rows[0].total_trades === 0) return null;
  const r = rows[0];

  const profitFactor = Number(r.gross_loss) > 0 ? Number(r.gross_profit) / Number(r.gross_loss) : Number(r.gross_profit) > 0 ? Infinity : 0;
  const stddev = Number(r.pnl_stddev);
  const sharpe = stddev > 0 ? Number(r.avg_pnl) / stddev : 0;
  const downsideDev = Number(r.downside_dev);
  const sortino = downsideDev > 0 ? Number(r.avg_pnl) / downsideDev : 0;
  const winRate = Number(r.win_count) / Number(r.total_trades);
  const avgWin = Number(r.win_count) > 0 ? Number(r.gross_profit) / Number(r.win_count) : 0;
  const lossCount = Number(r.total_trades) - Number(r.win_count);
  const avgLoss = lossCount > 0 ? Number(r.gross_loss) / lossCount : 0;
  const expectancy = winRate * avgWin - (1 - winRate) * avgLoss;

  // Days span for avg trades per day
  const firstDate = new Date(r.first_date);
  const lastDate = new Date(r.last_date);
  const daySpan = Math.max(1, Math.ceil((lastDate.getTime() - firstDate.getTime()) / 86400000) + 1);
  const avgTradesPerDay = Number(r.total_trades) / daySpan;

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
  const eqRows = await eqResult.json<{ cumulative_pnl: number }>();
  // Use equity (initial_balance + cumulative_pnl) to compute drawdown percentage
  const initialBalance = 10000;
  let peakEquity = initialBalance;
  let maxDrawdown = 0;
  for (const row of eqRows) {
    const equity = initialBalance + Number(row.cumulative_pnl);
    if (equity > peakEquity) peakEquity = equity;
    const dd = peakEquity > 0 ? (peakEquity - equity) / peakEquity : 0;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  return {
    profit_factor: Number(profitFactor.toFixed(2)),
    sharpe_ratio: Number(sharpe.toFixed(2)),
    sortino_ratio: Number(sortino.toFixed(2)),
    max_drawdown_pct: Number((maxDrawdown * 100).toFixed(2)),
    avg_hold_time_minutes: Number(Number(r.avg_hold_time_minutes).toFixed(1)),
    expectancy: Number(expectancy.toFixed(4)),
    total_trades: Number(r.total_trades),
    avg_trades_per_day: Number(avgTradesPerDay.toFixed(1)),
  };
}

export async function getTradesWithSignals(strategy: string): Promise<Array<Trade & { signal_details: string }>> {
  const result = await clickhouse.query({
    query: `SELECT *, signal_details FROM mvhe.trades WHERE strategy = {strategy:String} AND signal_details != '' ORDER BY entry_time DESC LIMIT 1000`,
    query_params: { strategy },
    format: 'JSONEachRow',
  });
  return result.json<Trade & { signal_details: string }>();
}

export async function getAuditEvents(limit: number = 50): Promise<AuditEvent[]> {
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.audit_events ORDER BY timestamp DESC LIMIT {limit:UInt32}`,
    query_params: { limit },
    format: 'JSONEachRow',
  });
  return result.json<AuditEvent>();
}
