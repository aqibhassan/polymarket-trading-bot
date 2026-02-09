import { NextResponse } from 'next/server';
import { getTradesWithSignals } from '@/lib/queries/trades';
import { validStrategy } from '@/lib/validation';
import type { SignalComboWinRate } from '@/lib/types/trade';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const strategy = validStrategy(searchParams.get('strategy'));

  try {
    const trades = await getTradesWithSignals(strategy);

    // Group by active signal combination
    const comboMap = new Map<string, { wins: number; total: number; pnls: number[] }>();

    for (const t of trades) {
      let comboKey: string;
      try {
        // signal_details is a JSON string with vote info
        const details = JSON.parse(t.signal_details);
        if (Array.isArray(details)) {
          // Extract active signal names (non-neutral votes)
          const active = details
            .filter((v: { direction?: string }) => v.direction && v.direction !== 'neutral')
            .map((v: { name?: string }) => v.name || 'unknown')
            .sort();
          comboKey = active.length > 0 ? active.join(' + ') : 'none';
        } else {
          comboKey = 'unknown';
        }
      } catch {
        // Fallback: try pipe-delimited format from votes string
        const parts = t.signal_details.split('|').map((s: string) => s.trim()).filter(Boolean);
        const active = parts
          .filter((p: string) => !p.toLowerCase().includes('neutral'))
          .map((p: string) => p.split(':')[0]?.trim() || p)
          .sort();
        comboKey = active.length > 0 ? active.join(' + ') : 'none';
      }

      const entry = comboMap.get(comboKey) || { wins: 0, total: 0, pnls: [] };
      entry.total++;
      entry.pnls.push(Number(t.pnl));
      if (Number(t.pnl) > 0) entry.wins++;
      comboMap.set(comboKey, entry);
    }

    const combos: SignalComboWinRate[] = Array.from(comboMap.entries())
      .map(([combo, data]) => ({
        combo,
        total: data.total,
        wins: data.wins,
        win_rate: data.total > 0 ? Number(((data.wins / data.total) * 100).toFixed(1)) : 0,
        avg_pnl: data.pnls.length > 0 ? Number((data.pnls.reduce((a, b) => a + b, 0) / data.pnls.length).toFixed(4)) : 0,
        total_pnl: Number(data.pnls.reduce((a, b) => a + b, 0).toFixed(4)),
      }))
      .sort((a, b) => b.total - a.total);

    return NextResponse.json({ combos });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch signal combos' },
      { status: 500 },
    );
  }
}

export const dynamic = 'force-dynamic';
