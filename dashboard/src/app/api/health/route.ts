import { NextResponse } from 'next/server';

const BWO_API = process.env.BWO_API_URL || 'http://localhost:8100';

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const res = await fetch(`${BWO_API}/api/health?${searchParams}`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: 'Failed to fetch from data server' }, { status: 502 });
  }
}
