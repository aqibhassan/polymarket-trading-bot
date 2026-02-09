import { NextRequest, NextResponse } from 'next/server';
import { getRecentCandles } from '@/lib/queries/candles';
import { validLimit, validSymbol } from '@/lib/validation';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = validSymbol(searchParams.get('symbol'));
    const limit = validLimit(searchParams.get('limit'), 100, 10000);
    const candles = await getRecentCandles(symbol, limit);
    return NextResponse.json({ candles });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch candles' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
