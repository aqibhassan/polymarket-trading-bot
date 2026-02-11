import { NextResponse } from 'next/server';
import redis from '@/lib/db/redis';
import pool from '@/lib/db/timescale';
import clickhouse from '@/lib/db/clickhouse';
import { getBotHeartbeat, getBotPosition } from '@/lib/queries/bot-state';
import type { HealthStatus } from '@/lib/types/bot-state';

export async function GET() {
  const status: HealthStatus = {
    bot_alive: false,
    heartbeat_age_s: null,
    uptime_s: null,
    idle_since_s: null,
    last_trade_time: null,
    has_open_position: false,
    redis: false,
    clickhouse: false,
    timescaledb: false,
    mode: null,
    strategy: null,
  };

  // Check Redis
  try {
    await redis.ping();
    status.redis = true;
  } catch { /* connection failed */ }

  // Check TimescaleDB
  try {
    await pool.query('SELECT 1');
    status.timescaledb = true;
  } catch { /* connection failed */ }

  // Check ClickHouse
  try {
    await clickhouse.query({ query: 'SELECT 1', format: 'JSONEachRow' });
    status.clickhouse = true;
  } catch { /* connection failed */ }

  // Check bot heartbeat + uptime
  try {
    const heartbeat = await getBotHeartbeat();
    if (heartbeat) {
      const age = (Date.now() - new Date(heartbeat.timestamp).getTime()) / 1000;
      status.bot_alive = age < 120;
      status.heartbeat_age_s = Math.round(age);
      status.uptime_s = Math.round(heartbeat.uptime_s);
      status.mode = heartbeat.mode;
      status.strategy = heartbeat.strategy;
    }
  } catch { /* heartbeat unavailable */ }

  // Check open position
  try {
    const pos = await getBotPosition();
    status.has_open_position = pos !== null;
  } catch { /* position unavailable */ }

  // Get last trade time from ClickHouse → compute idle time
  try {
    const result = await clickhouse.query({
      query: 'SELECT max(exit_time) AS last_exit, count() AS cnt FROM mvhe.trades',
      format: 'JSONEachRow',
    });
    const rows = await result.json<{ last_exit: string; cnt: string }>();
    if (rows.length > 0 && Number(rows[0].cnt) > 0 && rows[0].last_exit) {
      status.last_trade_time = rows[0].last_exit;
      const lastExit = new Date(rows[0].last_exit).getTime();
      if (!isNaN(lastExit) && lastExit > 0) {
        status.idle_since_s = Math.round((Date.now() - lastExit) / 1000);
      }
    }
  } catch { /* query failed */ }

  const httpStatus = status.redis && status.clickhouse && status.timescaledb ? 200 : 503;
  return NextResponse.json(status, { status: httpStatus });
}

export const dynamic = 'force-dynamic';
