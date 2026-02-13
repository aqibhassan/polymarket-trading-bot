'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { StatCard } from '@/components/charts/stat-card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { ErrorBanner } from '@/components/ui/error-banner';
import type { BWOAdvancedMetrics, BWOTradeRecord } from '@/lib/types/bwo';

export default function RiskPage() {
  const [metrics, setMetrics] = useState<BWOAdvancedMetrics | null>(null);
  const [trades, setTrades] = useState<BWOTradeRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [metRes, trRes] = await Promise.all([
        fetch('/api/metrics'),
        fetch('/api/trades?limit=1000'),
      ]);
      const metData = await metRes.json();
      const trData = await trRes.json();
      setMetrics(metData.total_trades != null ? metData : null);
      setTrades(trData.trades || []);
      setError(null);
    } catch {
      setError('Failed to load risk data');
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const maxConsecLosses = metrics?.max_consec_losses ?? 0;
  const maxDrawdownPct = metrics?.max_drawdown_pct ?? 0;
  const totalFees = metrics?.total_fees ?? 0;
  const drawdownLimit = 5;

  // Daily fees from trades
  const dailyFeeMap = new Map<string, number>();
  for (const t of trades) {
    if (!t.entered) continue;
    const day = (t.window_ts || '').split(/[T ]/)[0];
    if (!day) continue;
    dailyFeeMap.set(day, (dailyFeeMap.get(day) || 0) + (t.fee || 0));
  }
  const feeData = Array.from(dailyFeeMap.entries())
    .map(([date, fees]) => ({ date, fees: Number(fees.toFixed(4)) }))
    .sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold text-zinc-50">Risk Monitor</h1>
      {error && <ErrorBanner message={error} onRetry={fetchData} />}

      {/* Risk Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard title="Max Consecutive Losses" value={maxConsecLosses} />
        <StatCard
          title="Max Drawdown"
          value={`${maxDrawdownPct.toFixed(2)}%`}
          subtitle={`of ${drawdownLimit}% limit`}
          trend={maxDrawdownPct > 2 ? 'down' : 'neutral'}
        />
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
              <span className="text-zinc-400">Peak-to-Trough: {maxDrawdownPct.toFixed(2)}%</span>
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
