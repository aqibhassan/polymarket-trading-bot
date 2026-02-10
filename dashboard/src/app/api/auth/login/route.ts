import { NextRequest, NextResponse } from 'next/server';
import { createSession, getDashboardPassword, sessionCookieOptions } from '@/lib/auth';
import { timingSafeEqual } from 'crypto';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { password } = body;

    if (!password || typeof password !== 'string') {
      return NextResponse.json({ error: 'Password required' }, { status: 400 });
    }

    const expected = getDashboardPassword();

    // Timing-safe comparison
    const a = Buffer.from(password);
    const b = Buffer.from(expected);
    const valid = a.length === b.length && timingSafeEqual(a, b);

    if (!valid) {
      return NextResponse.json({ error: 'Invalid password' }, { status: 401 });
    }

    const token = await createSession();
    const response = NextResponse.json({ ok: true });
    response.cookies.set(sessionCookieOptions(token));
    return response;
  } catch {
    return NextResponse.json({ error: 'Internal error' }, { status: 500 });
  }
}
