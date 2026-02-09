import { NextResponse } from 'next/server';
import { getAdvancedMetrics } from '@/lib/queries/trades';
import { validStrategy } from '@/lib/validation';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const strategy = validStrategy(searchParams.get('strategy'));

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
