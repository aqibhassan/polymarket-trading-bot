'use client';

import { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { StatCard } from '@/components/charts/stat-card';
import { EquityCurve } from '@/components/charts/equity-curve';
import { TradesTable } from '@/components/charts/trades-table';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { StrategyPerformance, Trade } from '@/lib/types/trade';

export default function PerformancePage() {
  const [perf, setPerf] = useState<StrategyPerformance | null>(null);
  const [equityData, setEquityData] = useState<Array<{ time: string; cumulative_pnl: number }>>([]);
  const [trades, setTrades] = useState<Trade[]>([]);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [perfRes, equityRes, tradesRes] = await Promise.all([
          fetch('/api/performance?strategy=momentum_confirmation'),
          fetch('/api/equity-curve'),
          fetch('/api/trades?limit=100'),
        ]);
        const perfData = await perfRes.json();
        const equityJson = await equityRes.json();
        const tradesData = await tradesRes.json();

        setPerf(perfData);
        setEquityData(equityJson.data || []);
        setTrades(tradesData.trades || []);
      } catch { /* fetch failed */ }
    };

    fetchAll();
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, []);

  const winRate = perf && perf.trade_count > 0
    ? ((perf.win_count / perf.trade_count) * 100).toFixed(1)
    : '—';

  // Build daily P&L bar chart data from trades
  const dailyPnlMap = new Map<string, number>();
  for (const t of trades) {
    const day = t.entry_time.split(/[T ]/)[0];
    dailyPnlMap.set(day, (dailyPnlMap.get(day) || 0) + Number(t.pnl));
  }
  const dailyPnlData = Array.from(dailyPnlMap.entries())
    .map(([date, pnl]) => ({ date, pnl: Number(pnl.toFixed(2)) }))
    .sort((a, b) => a.date.localeCompare(b.date));

  // Win rate by entry minute
  const minuteWins = new Map<number, { wins: number; total: number }>();
  for (const t of trades) {
    const m = t.window_minute;
    const entry = minuteWins.get(m) || { wins: 0, total: 0 };
    entry.total++;
    if (Number(t.pnl) > 0) entry.wins++;
    minuteWins.set(m, entry);
  }
  const winByMinute = Array.from(minuteWins.entries())
    .map(([minute, { wins, total }]) => ({
      minute: `Min ${minute}`,
      win_rate: total > 0 ? Number(((wins / total) * 100).toFixed(1)) : 0,
      total,
    }))
    .sort((a, b) => a.minute.localeCompare(b.minute));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-zinc-50">Performance Analytics</h1>

      {/* Strategy Summary */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-6">
        <StatCard title="Total P&L" value={perf ? `$${Number(perf.total_pnl).toFixed(2)}` : '—'} trend={perf && Number(perf.total_pnl) >= 0 ? 'up' : 'down'} />
        <StatCard title="Win Rate" value={winRate !== '—' ? `${winRate}%` : '—'} />
        <StatCard title="Best Trade" value={perf ? `$${Number(perf.best_trade).toFixed(2)}` : '—'} trend="up" />
        <StatCard title="Worst Trade" value={perf ? `$${Number(perf.worst_trade).toFixed(2)}` : '—'} trend="down" />
        <StatCard title="Avg Position" value={perf ? `$${Number(perf.avg_position_size).toFixed(2)}` : '—'} />
        <StatCard title="Avg Confidence" value={perf ? `${(Number(perf.avg_confidence) * 100).toFixed(1)}%` : '—'} />
      </div>

      {/* Full Equity Curve */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Equity Curve (All History)</CardTitle>
        </CardHeader>
        <CardContent>
          <EquityCurve data={equityData} height={350} />
        </CardContent>
      </Card>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Daily P&L */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Daily P&L</CardTitle>
          </CardHeader>
          <CardContent>
            {dailyPnlData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={dailyPnlData}>
                  <XAxis dataKey="date" stroke="#71717a" fontSize={11} tickFormatter={(v: string) => v.slice(5)} />
                  <YAxis stroke="#71717a" fontSize={11} tickFormatter={(v: number) => `$${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
                    formatter={(v: number) => [`$${v.toFixed(2)}`, 'P&L']}
                  />
                  <Bar dataKey="pnl">
                    {dailyPnlData.map((entry, i) => (
                      <Cell key={i} fill={entry.pnl >= 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[250px] text-zinc-500">No data</div>
            )}
          </CardContent>
        </Card>

        {/* Win Rate by Minute */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Win Rate by Entry Minute</CardTitle>
          </CardHeader>
          <CardContent>
            {winByMinute.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={winByMinute}>
                  <XAxis dataKey="minute" stroke="#71717a" fontSize={11} />
                  <YAxis stroke="#71717a" fontSize={11} tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
                    formatter={(v: number, _: string, props: { payload?: { total: number } }) => [`${v}% (${props.payload?.total ?? 0} trades)`, 'Win Rate']}
                  />
                  <Bar dataKey="win_rate" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[250px] text-zinc-500">No data</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Trade History */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <TradesTable trades={trades} />
        </CardContent>
      </Card>
    </div>
  );
}
