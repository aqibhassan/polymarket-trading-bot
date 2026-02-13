import { NextRequest, NextResponse } from 'next/server';
import { getAuditEvents } from '@/lib/queries/trades';
import { validLimit } from '@/lib/validation';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = validLimit(searchParams.get('limit'), 50, 1000);
    const events = await getAuditEvents(limit);
    return NextResponse.json({ events });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch audit events' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
