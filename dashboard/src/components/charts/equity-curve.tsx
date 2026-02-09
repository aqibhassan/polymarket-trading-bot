'use client';

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';

interface EquityCurveProps {
  data: Array<{ time: string; cumulative_pnl: number }>;
  height?: number;
}

export function EquityCurve({ data, height = 300 }: EquityCurveProps) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-[300px] text-zinc-500">
        No trade data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis
          dataKey="time"
          stroke="#71717a"
          fontSize={12}
          tickFormatter={(val: string) => new Date(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        />
        <YAxis stroke="#71717a" fontSize={12} tickFormatter={(val: number) => `$${val.toFixed(2)}`} />
        <Tooltip
          contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
          labelStyle={{ color: '#a1a1aa' }}
          formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
          labelFormatter={(label: string) => new Date(label).toLocaleString()}
        />
        <ReferenceLine y={0} stroke="#52525b" strokeDasharray="3 3" />
        <Line
          type="monotone"
          dataKey="cumulative_pnl"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: '#10b981' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
