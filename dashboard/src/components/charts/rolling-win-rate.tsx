'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { BWORollingWinRate } from '@/lib/types/bwo';

interface RollingWinRateProps {
  data: BWORollingWinRate[];
}

export function RollingWinRate({ data }: RollingWinRateProps) {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Rolling 20-Trade Win Rate</CardTitle>
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
        <CardTitle className="text-sm font-medium text-zinc-400">Rolling 20-Trade Win Rate</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="index"
              stroke="#71717a"
              fontSize={11}
              label={{ value: 'Trade #', position: 'insideBottom', offset: -2, style: { fill: '#71717a', fontSize: 10 } }}
            />
            <YAxis
              stroke="#71717a"
              fontSize={11}
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
              labelStyle={{ color: '#a1a1aa' }}
              formatter={(value: number) => [`${value.toFixed(1)}%`, 'Win Rate']}
              labelFormatter={(label: number) => `Trade #${label}`}
            />
            <ReferenceLine
              y={80}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{ value: 'Target 80%', position: 'right', style: { fill: '#f59e0b', fontSize: 10 } }}
            />
            <Line
              type="monotone"
              dataKey="win_rate"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#10b981' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
