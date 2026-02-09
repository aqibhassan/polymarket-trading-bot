import { NextResponse } from 'next/server';
import { getBotPosition } from '@/lib/queries/bot-state';

export async function GET() {
  try {
    const position = await getBotPosition();
    return NextResponse.json(position);
  } catch {
    return NextResponse.json({ error: 'Failed to fetch position' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
