import { NextResponse } from 'next/server';
import { getKillSwitch } from '@/lib/queries/bot-state';

export async function GET() {
  try {
    const active = await getKillSwitch();
    return NextResponse.json({ active });
  } catch {
    return NextResponse.json({ error: 'Failed to check kill switch' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
