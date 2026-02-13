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
import type { BWOEntryPriceBucket } from '@/lib/types/bwo';

interface EntryPriceAnalysisProps {
  data: BWOEntryPriceBucket[];
}

function getColor(winRate: number): string {
  if (winRate > 70) return '#10b981';
  if (winRate >= 50) return '#f59e0b';
  return '#ef4444';
}

export function EntryPriceAnalysis({ data }: EntryPriceAnalysisProps) {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Win Rate by Entry Price</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[250px] text-zinc-500">
            No data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium text-zinc-400">Win Rate by Entry Price</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <XAxis dataKey="bucket" stroke="#71717a" fontSize={11} />
            <YAxis
              stroke="#71717a"
              fontSize={11}
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
              labelStyle={{ color: '#a1a1aa' }}
              formatter={(value: number, _: string, props: { payload?: BWOEntryPriceBucket }) => {
                const p = props.payload;
                return [
                  `${value.toFixed(1)}% (${p?.total ?? 0} trades, avg $${p?.avg_pnl?.toFixed(2) ?? '0.00'})`,
                  'Win Rate',
                ];
              }}
            />
            <Bar dataKey="win_rate" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={getColor(entry.win_rate)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
