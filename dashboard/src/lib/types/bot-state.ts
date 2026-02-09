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

export interface BotState {
  heartbeat: BotHeartbeat | null;
  balance: BotBalance | null;
  position: BotPosition | null;
  window: BotWindow | null;
  daily: BotDaily | null;
  ws_status: WsStatus | null;
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
