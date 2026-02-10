'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { StatCard } from '@/components/charts/stat-card';
import { Badge } from '@/components/ui/badge';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { Trade } from '@/lib/types/trade';

const INITIAL_BALANCE = 10000;

export default function RiskPage() {
  const [killSwitch, setKillSwitch] = useState(false);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [maxDrawdownPct, setMaxDrawdownPct] = useState(0);

  const fetchData = useCallback(async () => {
    try {
      const [ksRes, trRes, eqRes] = await Promise.all([
        fetch('/api/kill-switch'),
        fetch('/api/trades?limit=1000'),
        fetch('/api/equity-curve'),
      ]);
      const ksData = await ksRes.json();
      const trData = await trRes.json();
      const eqData = await eqRes.json();
      setKillSwitch(ksData.active || false);
      setTrades(trData.trades || []);

      // Compute peak-to-trough drawdown from ClickHouse equity curve
      const curve: Array<{ cumulative_pnl: number }> = eqData.data || [];
      let peak = INITIAL_BALANCE;
      let maxDd = 0;
      for (const pt of curve) {
        const equity = INITIAL_BALANCE + Number(pt.cumulative_pnl);
        if (equity > peak) peak = equity;
        const dd = peak > 0 ? ((peak - equity) / peak) * 100 : 0;
        if (dd > maxDd) maxDd = dd;
      }
      setMaxDrawdownPct(Number(maxDd.toFixed(2)));
    } catch { /* fetch failed */ }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Calculate max consecutive losses
  let maxConsecLosses = 0;
  let currentStreak = 0;
  for (const t of [...trades].reverse()) {
    if (t.pnl < 0) {
      currentStreak++;
      maxConsecLosses = Math.max(maxConsecLosses, currentStreak);
    } else {
      currentStreak = 0;
    }
  }

  // Drawdown from ClickHouse equity curve (peak-to-trough, accurate across restarts)
  const drawdownLimit = 5; // 5% max daily drawdown

  // Total fees from ClickHouse trades (numeric coercion done in query layer)
  const totalFees = trades.reduce((s, t) => s + (t.fee_cost || 0), 0);

  // Daily fees from trades
  const dailyFeeMap = new Map<string, number>();
  for (const t of trades) {
    const day = (t.entry_time || '').split(/[T ]/)[0];
    if (!day) continue;
    dailyFeeMap.set(day, (dailyFeeMap.get(day) || 0) + (t.fee_cost || 0));
  }
  const feeData = Array.from(dailyFeeMap.entries())
    .map(([date, fees]) => ({ date, fees: Number(fees.toFixed(4)) }))
    .sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold text-zinc-50">Risk Monitor</h1>

      {/* Kill Switch Status */}
      <Card>
        <CardContent className="p-8">
          <div className="flex items-center gap-6">
            <div className={`h-16 w-16 rounded-full flex items-center justify-center ${killSwitch ? 'bg-red-500/20 ring-4 ring-red-500/50' : 'bg-emerald-500/20 ring-4 ring-emerald-500/50'}`}>
              <div className={`h-8 w-8 rounded-full ${killSwitch ? 'bg-red-500 animate-pulse' : 'bg-emerald-500'}`} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-zinc-50">
                Kill Switch: {killSwitch ? 'ACTIVE' : 'Inactive'}
              </h2>
              <p className="text-sm text-zinc-400">
                {killSwitch
                  ? 'Trading halted — daily drawdown limit breached'
                  : 'System operating normally within risk parameters'}
              </p>
            </div>
            <Badge variant={killSwitch ? 'destructive' : 'success'} className="ml-auto text-sm px-4 py-1">
              {killSwitch ? 'HALTED' : 'SAFE'}
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Risk Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard title="Max Consecutive Losses" value={maxConsecLosses} />
        <StatCard title="Max Drawdown" value={`${maxDrawdownPct}%`} subtitle={`of ${drawdownLimit}% limit`} trend={maxDrawdownPct > 2 ? 'down' : 'neutral'} />
        <StatCard title="Total Fees Paid" value={`$${totalFees.toFixed(2)}`} />
      </div>

      {/* Drawdown Gauge */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Max Drawdown vs Limit</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">Peak-to-Trough: {maxDrawdownPct}%</span>
              <span className="text-zinc-400">Limit: {drawdownLimit}%</span>
            </div>
            <div className="h-4 w-full rounded-full bg-zinc-800 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${maxDrawdownPct > 3 ? 'bg-red-500' : maxDrawdownPct > 1.5 ? 'bg-yellow-500' : 'bg-emerald-500'}`}
                style={{ width: `${Math.min(100, (maxDrawdownPct / drawdownLimit) * 100)}%` }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fee Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Daily Fee Costs</CardTitle>
        </CardHeader>
        <CardContent>
          {feeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={feeData}>
                <XAxis dataKey="date" stroke="#71717a" fontSize={11} tickFormatter={(v: string) => v.slice(5)} />
                <YAxis stroke="#71717a" fontSize={11} tickFormatter={(v: number) => `$${v}`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
                  formatter={(v: number) => [`$${v.toFixed(4)}`, 'Fees']}
                />
                <Bar dataKey="fees" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[250px] text-zinc-500">No fee data</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
