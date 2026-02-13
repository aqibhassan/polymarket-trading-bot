import { NextRequest, NextResponse } from 'next/server';
import { getEquityCurve } from '@/lib/queries/trades';
import { validStrategyOrUndefined } from '@/lib/validation';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const strategy = validStrategyOrUndefined(searchParams.get('strategy'));
    const data = await getEquityCurve(strategy);
    return NextResponse.json({ data });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch equity curve' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
