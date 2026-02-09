'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { SignalComboWinRate } from '@/lib/types/trade';

interface SignalComboChartProps {
  combos: SignalComboWinRate[];
}

function winRateColor(rate: number): string {
  if (rate >= 70) return '#10b981';
  if (rate >= 50) return '#3b82f6';
  if (rate >= 30) return '#f59e0b';
  return '#ef4444';
}

export function SignalComboChart({ combos }: SignalComboChartProps) {
  if (combos.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Win Rate by Signal Combination</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[250px] text-zinc-500">
            No signal combination data yet
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium text-zinc-400">Win Rate by Signal Combination</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Horizontal bar chart */}
        <ResponsiveContainer width="100%" height={Math.max(200, combos.length * 40)}>
          <BarChart data={combos} layout="vertical" margin={{ left: 120 }}>
            <XAxis type="number" domain={[0, 100]} stroke="#71717a" fontSize={11} tickFormatter={(v: number) => `${v}%`} />
            <YAxis type="category" dataKey="combo" stroke="#71717a" fontSize={10} width={110} />
            <Tooltip
              contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
              formatter={(v: number, _: string, props: { payload?: SignalComboWinRate }) => [
                `${v}% (${props.payload?.total ?? 0} trades)`,
                'Win Rate',
              ]}
            />
            <Bar dataKey="win_rate" radius={[0, 4, 4, 0]}>
              {combos.map((entry, i) => (
                <Cell key={i} fill={winRateColor(entry.win_rate)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Summary table */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="text-left py-2 pr-4">Combination</th>
                <th className="text-right py-2 px-2">Trades</th>
                <th className="text-right py-2 px-2">Win Rate</th>
                <th className="text-right py-2 px-2">Avg P&L</th>
                <th className="text-right py-2 pl-2">Total P&L</th>
              </tr>
            </thead>
            <tbody>
              {combos.map((c) => (
                <tr key={c.combo} className="border-b border-zinc-800/50">
                  <td className="py-1.5 pr-4 text-zinc-300">{c.combo}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-zinc-400">{c.total}</td>
                  <td className="py-1.5 px-2 text-right font-mono" style={{ color: winRateColor(c.win_rate) }}>
                    {c.win_rate.toFixed(1)}%
                  </td>
                  <td className={`py-1.5 px-2 text-right font-mono ${c.avg_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    ${c.avg_pnl.toFixed(4)}
                  </td>
                  <td className={`py-1.5 pl-2 text-right font-mono ${c.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    ${c.total_pnl.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
