import { NextResponse } from 'next/server';
import redis from '@/lib/db/redis';
import pool from '@/lib/db/timescale';
import clickhouse from '@/lib/db/clickhouse';
import { getBotHeartbeat } from '@/lib/queries/bot-state';
import type { HealthStatus } from '@/lib/types/bot-state';

export async function GET() {
  const status: HealthStatus = {
    bot_alive: false,
    heartbeat_age_s: null,
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

  // Check bot heartbeat
  try {
    const heartbeat = await getBotHeartbeat();
    if (heartbeat) {
      const age = (Date.now() - new Date(heartbeat.timestamp).getTime()) / 1000;
      status.bot_alive = age < 120;
      status.heartbeat_age_s = Math.round(age);
      status.mode = heartbeat.mode;
      status.strategy = heartbeat.strategy;
    }
  } catch { /* heartbeat unavailable */ }

  const httpStatus = status.redis && status.clickhouse && status.timescaledb ? 200 : 503;
  return NextResponse.json(status, { status: httpStatus });
}

export const dynamic = 'force-dynamic';
