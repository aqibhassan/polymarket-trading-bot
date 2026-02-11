'use client';

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';

interface EquityCurveProps {
  data: Array<{ time: string; cumulative_pnl: number }>;
  height?: number;
}

export function EquityCurve({ data, height = 300 }: EquityCurveProps) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center text-zinc-500" style={{ height }}>
        No trade data available
      </div>
    );
  }

  // Smart x-axis: show HH:MM for short spans (<3 days), date for longer spans
  const firstTime = data[0]?.time ? new Date(data[0].time).getTime() : 0;
  const lastTime = data[data.length - 1]?.time ? new Date(data[data.length - 1].time).getTime() : 0;
  const spanDays = (lastTime - firstTime) / 86400000;
  const showTime = spanDays < 3;

  const formatXAxis = (val: string) => {
    const d = new Date(val);
    if (showTime) {
      return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' });
    }
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis
          dataKey="time"
          stroke="#71717a"
          fontSize={12}
          tickFormatter={formatXAxis}
        />
        <YAxis stroke="#71717a" fontSize={12} tickFormatter={(val: number) => `$${val.toFixed(2)}`} />
        <Tooltip
          contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
          labelStyle={{ color: '#a1a1aa' }}
          formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
          labelFormatter={(label: string) => {
            const d = new Date(label);
            return d.toLocaleString('en-GB', { timeZone: 'UTC' }) + ' UTC';
          }}
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
