import pool from '../db/timescale';

export interface Candle {
  time: string;
  exchange: string;
  symbol: string;
  interval: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function getRecentCandles(symbol: string = 'BTCUSDT', limit: number = 100): Promise<Candle[]> {
  const { rows } = await pool.query<Candle>(
    'SELECT time, exchange, symbol, interval, open, high, low, close, volume FROM candles WHERE symbol = $1 ORDER BY time DESC LIMIT $2',
    [symbol, limit],
  );
  return rows;
}

export async function getCandlesInRange(symbol: string, start: string, end: string): Promise<Candle[]> {
  const { rows } = await pool.query<Candle>(
    'SELECT time, exchange, symbol, interval, open, high, low, close, volume FROM candles WHERE symbol = $1 AND time >= $2 AND time <= $3 ORDER BY time DESC',
    [symbol, start, end],
  );
  return rows;
}
