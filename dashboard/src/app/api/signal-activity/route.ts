import { NextResponse } from 'next/server';
import { getRecentSignalActivity } from '@/lib/queries/trades';
import { validLimit } from '@/lib/validation';
import type { SignalActivityEvent } from '@/lib/types/bot-state';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = validLimit(searchParams.get('limit'), 20, 100);
    const rows = await getRecentSignalActivity(limit);

    const events: SignalActivityEvent[] = rows.map((r) => ({
      id: r.eval_id,
      timestamp: r.timestamp,
      minute: Number(r.minute),
      market_id: r.market_id,
      outcome: r.outcome as 'entry' | 'skip' | 'rejected',
      reason: r.reason,
      direction: r.direction,
      confidence: Number(r.confidence),
      votes: {
        yes: Number(r.votes_yes),
        no: Number(r.votes_no),
        neutral: Number(r.votes_neutral),
      },
      detail: r.detail,
    }));

    return NextResponse.json({ events });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch signal activity' },
      { status: 500 },
    );
  }
}

export const dynamic = 'force-dynamic';
