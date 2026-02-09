import { NextRequest, NextResponse } from 'next/server';
import { getEquityCurve } from '@/lib/queries/trades';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const strategy = searchParams.get('strategy') || undefined;
    const data = await getEquityCurve(strategy);
    return NextResponse.json({ data });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch equity curve' }, { status: 500 });
  }
}
