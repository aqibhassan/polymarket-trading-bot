import clickhouse from '../db/clickhouse';
import type { Trade, DailyPnl, StrategyPerformance, AuditEvent } from '../types/trade';

export async function getRecentTrades(limit: number = 10): Promise<Trade[]> {
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.trades ORDER BY entry_time DESC LIMIT {limit:UInt32}`,
    query_params: { limit },
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

export async function getEquityCurve(): Promise<Array<{ time: string; cumulative_pnl: number }>> {
  const result = await clickhouse.query({
    query: `
      SELECT
        entry_time AS time,
        sum(pnl) OVER (ORDER BY entry_time) AS cumulative_pnl
      FROM mvhe.trades
      ORDER BY entry_time ASC
    `,
    format: 'JSONEachRow',
  });
  return result.json<{ time: string; cumulative_pnl: number }>();
}

export async function getAuditEvents(limit: number = 50): Promise<AuditEvent[]> {
  const result = await clickhouse.query({
    query: `SELECT * FROM mvhe.audit_events ORDER BY timestamp DESC LIMIT {limit:UInt32}`,
    query_params: { limit },
    format: 'JSONEachRow',
  });
  return result.json<AuditEvent>();
}
