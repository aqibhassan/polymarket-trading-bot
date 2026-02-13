export interface BWOTradeRecord {
  window_ts: string;
  window_ts_unix: number;
  market_id: string;
  resolution: string;
  btc_return_pct: number;
  cont_prob: number;
  early_direction: number;
  btc_direction: string;
  entered: boolean;
  side: string;
  entry_price: number;
  settlement: number;
  shares: number;
  pnl_gross: number;
  fee: number;
  pnl_net: number;
  bankroll_after: number;
  correct: boolean;
  skip_reason: string;
}

export interface BWOSummary {
  updated_at: string;
  strategy: string;
  bankroll: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  total_pnl: number;
  ev_per_trade: number;
  total_windows: number;
  skip_rate: number;
}

export interface BWOEquityPoint {
  time: string;
  bankroll: number;
  cumulative_pnl: number;
}

export interface BWODailyPnl {
  date: string;
  pnl: number;
  trades: number;
  wins: number;
  losses: number;
}

export interface BWOSkipReason {
  reason: string;
  count: number;
  pct: number;
}

export interface BWOSkipMetrics {
  total_windows: number;
  total_skips: number;
  total_entries: number;
  skip_rate: number;
  reasons: BWOSkipReason[];
}

export interface BWOAdvancedMetrics {
  profit_factor: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  max_consec_losses: number;
  total_fees: number;
  best_trade: number;
  worst_trade: number;
  expectancy: number;
  total_trades: number;
  avg_trades_per_day: number;
}

export interface BWOConfidenceBucket {
  bucket: string;
  total: number;
  wins: number;
  win_rate: number;
  avg_pnl: number;
}

export interface BWOEntryPriceBucket {
  bucket: string;
  total: number;
  wins: number;
  win_rate: number;
  avg_pnl: number;
}

export interface BWOHealthStatus {
  server_ok: boolean;
  paper_trader_active: boolean;
  jsonl_age_s: number;
  total_records: number;
  last_modified: string;
}

export interface BWORollingWinRate {
  index: number;
  win_rate: number;
  time: string;
}
