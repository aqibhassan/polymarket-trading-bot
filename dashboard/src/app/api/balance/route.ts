import { NextResponse } from 'next/server';
import { getBotBalance } from '@/lib/queries/bot-state';

export async function GET() {
  try {
    const balance = await getBotBalance();
    return NextResponse.json(balance);
  } catch {
    return NextResponse.json({ error: 'Failed to fetch balance' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
