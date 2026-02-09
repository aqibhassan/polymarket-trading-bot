import { NextRequest, NextResponse } from 'next/server';
import { getDailyPnl } from '@/lib/queries/trades';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get('date') || new Date().toISOString().split('T')[0];
    const data = await getDailyPnl(date);
    return NextResponse.json({ date, data });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch daily PnL' }, { status: 500 });
  }
}
