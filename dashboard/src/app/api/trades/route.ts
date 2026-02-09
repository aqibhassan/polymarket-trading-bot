import { NextRequest, NextResponse } from 'next/server';
import { getRecentTrades, getTradesInRange } from '@/lib/queries/trades';
import { validLimit, validIsoDatetime, validStrategyOrUndefined } from '@/lib/validation';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = validLimit(searchParams.get('limit'), 10, 1000);
    const start = validIsoDatetime(searchParams.get('start'));
    const end = validIsoDatetime(searchParams.get('end'));
    const strategy = validStrategyOrUndefined(searchParams.get('strategy'));

    const trades = start && end
      ? await getTradesInRange(start, end)
      : await getRecentTrades(limit, strategy);

    return NextResponse.json({ trades });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch trades' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
