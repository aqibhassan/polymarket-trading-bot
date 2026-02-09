import { NextRequest, NextResponse } from 'next/server';
import { getRecentCandles } from '@/lib/queries/candles';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';
    const limit = parseInt(searchParams.get('limit') || '100');
    const candles = await getRecentCandles(symbol, limit);
    return NextResponse.json({ candles });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch candles' }, { status: 500 });
  }
}
