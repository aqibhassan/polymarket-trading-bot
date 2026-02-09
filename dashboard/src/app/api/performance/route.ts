import { NextRequest, NextResponse } from 'next/server';
import { getStrategyPerformance } from '@/lib/queries/trades';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const strategy = searchParams.get('strategy') || 'momentum_confirmation';
    const data = await getStrategyPerformance(strategy);
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: 'Failed to fetch performance' }, { status: 500 });
  }
}
