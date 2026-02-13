import { NextResponse } from 'next/server';
import { getBotWindow } from '@/lib/queries/bot-state';

export async function GET() {
  try {
    const window = await getBotWindow();
    return NextResponse.json(window);
  } catch {
    return NextResponse.json({ error: 'Failed to fetch window state' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
