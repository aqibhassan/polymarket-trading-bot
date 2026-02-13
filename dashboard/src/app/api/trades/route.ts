import { NextRequest, NextResponse } from 'next/server';
import { getRecentTrades, getTradesInRange, getTradeCount } from '@/lib/queries/trades';
import { validLimit, validIsoDatetime, validStrategyOrUndefined } from '@/lib/validation';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = validLimit(searchParams.get('limit'), 10, 1000);
    const offset = Math.max(0, parseInt(searchParams.get('offset') || '0', 10) || 0);
    const start = validIsoDatetime(searchParams.get('start'));
    const end = validIsoDatetime(searchParams.get('end'));
    const strategy = validStrategyOrUndefined(searchParams.get('strategy'));

    if (start && end) {
      const trades = await getTradesInRange(start, end, strategy);
      return NextResponse.json({ trades, total: trades.length });
    }

    const [trades, total] = await Promise.all([
      getRecentTrades(limit, strategy, offset),
      getTradeCount(strategy),
    ]);

    return NextResponse.json({ trades, total });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch trades' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
