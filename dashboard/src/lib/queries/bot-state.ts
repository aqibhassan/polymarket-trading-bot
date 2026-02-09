import redis from '../db/redis';
import type {
  BotHeartbeat,
  BotBalance,
  BotPosition,
  BotWindow,
  BotDaily,
  WsStatus,
  BotState,
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

export async function getKillSwitch(): Promise<boolean> {
  const raw = await redis.get('mvhe:kill_switch');
  return raw !== null && raw !== '' && raw !== '0' && raw !== 'false';
}

export async function getAllBotState(): Promise<BotState> {
  const [heartbeat, balance, position, window, daily, ws_status] = await Promise.all([
    getBotHeartbeat(),
    getBotBalance(),
    getBotPosition(),
    getBotWindow(),
    getBotDaily(),
    getWsStatus(),
  ]);

  return { heartbeat, balance, position, window, daily, ws_status };
}
