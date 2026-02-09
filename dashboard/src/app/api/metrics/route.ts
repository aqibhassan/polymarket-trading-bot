import { NextResponse } from 'next/server';
import { getAdvancedMetrics } from '@/lib/queries/trades';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const strategy = searchParams.get('strategy') || 'momentum_confirmation';

  try {
    const metrics = await getAdvancedMetrics(strategy);
    return NextResponse.json(metrics || {});
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch metrics' },
      { status: 500 },
    );
  }
}

export const dynamic = 'force-dynamic';
