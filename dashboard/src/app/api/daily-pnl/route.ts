import { NextRequest, NextResponse } from 'next/server';
import { getDailyPnl } from '@/lib/queries/trades';
import { validIsoDate } from '@/lib/validation';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const date = validIsoDate(searchParams.get('date'))
      || new Date().toISOString().split('T')[0];
    const data = await getDailyPnl(date);
    return NextResponse.json({ date, data });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch daily PnL' }, { status: 500 });
  }
}
