export interface BotHeartbeat {
  timestamp: string;
  uptime_s: number;
  mode: string;
  strategy: string;
}

export interface BotBalance {
  balance: string;
  initial_balance: string;
  pnl: string;
  pnl_pct: number;
}

export interface BotPosition {
  market_id: string;
  side: string;
  entry_price: string;
  size: string;
  entry_time: string;
}

export interface BotWindow {
  start_ts: number;
  minute: number;
  cum_return_pct: number;
  yes_price: string;
  btc_close: string;
}

export interface BotDaily {
  date: string;
  trade_count: number;
  daily_pnl: string;
  win_count: number;
  loss_count: number;
}

export interface WsStatus {
  binance: boolean;
  polymarket: boolean;
}

export interface LastTrade {
  trade_id: string;
  direction: string;
  entry_price: string;
  exit_price: string;
  pnl: string;
  exit_reason: string;
  window_minute: number;
  confidence: number;
  timestamp: string;
}

export interface SignalVote {
  name: string;
  direction: string;
  strength: number;
}

export interface SignalBreakdown {
  timestamp: string;
  minute: number;
  votes: SignalVote[];
  overall_confidence: number;
  direction: string;
  entry_generated: boolean;
}

export interface SizingDetails {
  kelly_fraction: string;
  recommended_size: string;
  max_allowed: string;
  capped_reason: string;
  balance: string;
  entry_price: string;
  estimated_win_prob: string;
}

export interface BotState {
  heartbeat: BotHeartbeat | null;
  balance: BotBalance | null;
  position: BotPosition | null;
  window: BotWindow | null;
  daily: BotDaily | null;
  ws_status: WsStatus | null;
  last_trade: LastTrade | null;
  signals: SignalBreakdown | null;
  sizing: SizingDetails | null;
}

export interface HealthStatus {
  bot_alive: boolean;
  heartbeat_age_s: number | null;
  redis: boolean;
  clickhouse: boolean;
  timescaledb: boolean;
  mode: string | null;
  strategy: string | null;
}
