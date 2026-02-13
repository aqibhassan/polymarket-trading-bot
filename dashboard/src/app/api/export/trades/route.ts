import { NextResponse } from 'next/server';

const BWO_API = process.env.BWO_API_URL || 'http://localhost:8100';

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const format = searchParams.get('format') || 'json';
    const res = await fetch(`${BWO_API}/api/export/trades?format=${format}`);

    if (format === 'csv') {
      const text = await res.text();
      return new Response(text, {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': 'attachment; filename="bwo_trades.csv"',
        },
      });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: 'Failed to export' }, { status: 502 });
  }
}
