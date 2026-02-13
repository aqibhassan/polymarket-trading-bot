import { NextResponse } from 'next/server';
import { getSkipMetrics } from '@/lib/queries/trades';
import { validStrategy, validLimit } from '@/lib/validation';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const strategy = validStrategy(searchParams.get('strategy'));
    const days = validLimit(searchParams.get('days'), 7, 90);
    const metrics = await getSkipMetrics(strategy, days);
    return NextResponse.json(metrics);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch skip metrics' },
      { status: 500 },
    );
  }
}

export const dynamic = 'force-dynamic';
