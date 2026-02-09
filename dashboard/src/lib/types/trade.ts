export interface Trade {
  trade_id: string;
  market_id: string;
  strategy: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  position_size: number;
  pnl: number;
  fee_cost: number;
  entry_time: string;
  exit_time: string;
  exit_reason: string;
  window_minute: number;
  cum_return_pct: number;
  confidence: number;
}

export interface DailyPnl {
  strategy: string;
  trade_count: number;
  total_pnl: number;
  total_fees: number;
  win_count: number;
}

export interface StrategyPerformance {
  strategy: string;
  trade_count: number;
  win_count: number;
  total_pnl: number;
  total_fees: number;
  avg_pnl: number;
  best_trade: number;
  worst_trade: number;
  avg_position_size: number;
  avg_confidence: number;
}

export interface AdvancedMetrics {
  profit_factor: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  avg_hold_time_minutes: number;
  expectancy: number;
  total_trades: number;
  avg_trades_per_day: number;
}

export interface SignalComboWinRate {
  combo: string;
  total: number;
  wins: number;
  win_rate: number;
  avg_pnl: number;
  total_pnl: number;
}

export interface AuditEvent {
  event_id: string;
  order_id: string;
  event_type: string;
  market_id: string;
  strategy: string;
  details: string;
  timestamp: string;
}
