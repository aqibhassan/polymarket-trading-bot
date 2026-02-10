import redis from '../db/redis';
import type {
  BotHeartbeat,
  BotBalance,
  BotPosition,
  BotWindow,
  BotDaily,
  WsStatus,
  BotState,
  LastTrade,
  SignalBreakdown,
  SizingDetails,
  SignalActivityEvent,
} from '../types/bot-state';

async function getKey<T>(key: string): Promise<T | null> {
  const raw = await redis.get(`mvhe:${key}`);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function getBotHeartbeat(): Promise<BotHeartbeat | null> {
  return getKey<BotHeartbeat>('bot:heartbeat');
}

export async function getBotBalance(): Promise<BotBalance | null> {
  return getKey<BotBalance>('bot:balance');
}

export async function getBotPosition(): Promise<BotPosition | null> {
  return getKey<BotPosition>('bot:position');
}

export async function getBotWindow(): Promise<BotWindow | null> {
  return getKey<BotWindow>('bot:window');
}

export async function getBotDaily(): Promise<BotDaily | null> {
  return getKey<BotDaily>('bot:daily');
}

export async function getWsStatus(): Promise<WsStatus | null> {
  return getKey<WsStatus>('bot:ws_status');
}

export async function getLastTrade(): Promise<LastTrade | null> {
  return getKey<LastTrade>('bot:last_trade');
}

export async function getSignals(): Promise<SignalBreakdown | null> {
  return getKey<SignalBreakdown>('bot:signals');
}

export async function getSizing(): Promise<SizingDetails | null> {
  return getKey<SizingDetails>('bot:sizing');
}

export async function getSignalActivity(): Promise<SignalActivityEvent[]> {
  const raw = await redis.lrange('mvhe:bot:signal_activity', 0, 19);
  if (!raw || raw.length === 0) return [];
  return raw.map((item) => {
    try {
      return JSON.parse(item) as SignalActivityEvent;
    } catch {
      return null;
    }
  }).filter((x): x is SignalActivityEvent => x !== null);
}

export async function getKillSwitch(): Promise<boolean> {
  const raw = await redis.get('mvhe:kill_switch');
  return raw !== null && raw !== '' && raw !== '0' && raw !== 'false';
}

export async function getAllBotState(): Promise<BotState> {
  const [heartbeat, balance, position, window, daily, ws_status, last_trade, signals, sizing, signal_activity] = await Promise.all([
    getBotHeartbeat(),
    getBotBalance(),
    getBotPosition(),
    getBotWindow(),
    getBotDaily(),
    getWsStatus(),
    getLastTrade(),
    getSignals(),
    getSizing(),
    getSignalActivity(),
  ]);

  return { heartbeat, balance, position, window, daily, ws_status, last_trade, signals, sizing, signal_activity };
}
