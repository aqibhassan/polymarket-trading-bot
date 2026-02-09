import { NextRequest, NextResponse } from 'next/server';
import { getRecentTrades, getTradesInRange } from '@/lib/queries/trades';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 1000);
    const start = searchParams.get('start');
    const end = searchParams.get('end');

    const trades = start && end
      ? await getTradesInRange(start, end)
      : await getRecentTrades(limit);

    return NextResponse.json({ trades });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch trades' }, { status: 500 });
  }
}
