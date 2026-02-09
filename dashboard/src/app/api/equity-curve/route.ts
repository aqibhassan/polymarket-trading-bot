import { NextResponse } from 'next/server';
import { getEquityCurve } from '@/lib/queries/trades';

export async function GET() {
  try {
    const data = await getEquityCurve();
    return NextResponse.json({ data });
  } catch {
    return NextResponse.json({ error: 'Failed to fetch equity curve' }, { status: 500 });
  }
}
