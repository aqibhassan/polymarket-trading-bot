'use client';

import { useState, useEffect, useCallback } from 'react';
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
import { MetricsGrid } from '@/components/charts/metrics-grid';
import { EquityCurve } from '@/components/charts/equity-curve';
import { SignalComboChart } from '@/components/charts/signal-combo-chart';
import { SkipMetricsPanel } from '@/components/charts/skip-metrics-panel';
import { TradesTable } from '@/components/charts/trades-table';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { StrategyPerformance, Trade, AdvancedMetrics, SignalComboWinRate, SkipMetrics } from '@/lib/types/trade';

/** Safe Number conversion — returns 0 for null/undefined/NaN */
function safeNum(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

export default function PerformancePage() {
  const [perf, setPerf] = useState<StrategyPerformance | null>(null);
  const [equityData, setEquityData] = useState<Array<{ time: string; cumulative_pnl: number }>>([]);
  const [allTrades, setAllTrades] = useState<Trade[]>([]);  // for charts
  const [trades, setTrades] = useState<Trade[]>([]);        // for table (paginated)
  const [tradeTotal, setTradeTotal] = useState(0);
  const [tradePage, setTradePage] = useState(1);
  const TRADES_PER_PAGE = 20;
  const [advancedMetrics, setAdvancedMetrics] = useState<AdvancedMetrics | null>(null);
  const [signalCombos, setSignalCombos] = useState<SignalComboWinRate[]>([]);
  const [skipMetrics, setSkipMetrics] = useState<SkipMetrics | null>(null);
  const [strategy, setStrategy] = useState('singularity');

  // Detect active strategy from health endpoint (lighter than SSE)
  useEffect(() => {
    const detect = async () => {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (data.strategy) setStrategy(data.strategy);
      } catch { /* use default */ }
    };
    detect();
  }, []);

  // Fetch chart/summary data (depends on strategy only, not page)
  useEffect(() => {
    const fetchCharts = async () => {
      try {
        const [perfRes, equityRes, allTradesRes, metricsRes, combosRes, skipRes] = await Promise.all([
          fetch(`/api/performance?strategy=${strategy}`),
          fetch(`/api/equity-curve?strategy=${strategy}`),
          fetch(`/api/trades?limit=1000&strategy=${strategy}`),
          fetch(`/api/metrics?strategy=${strategy}`),
          fetch(`/api/signal-combos?strategy=${strategy}`),
          fetch(`/api/skip-metrics?strategy=${strategy}`),
        ]);
        const perfData = await perfRes.json();
        const equityJson = await equityRes.json();
        const allTradesData = await allTradesRes.json();
        const metricsData = await metricsRes.json();
        const combosData = await combosRes.json();
        const skipData = await skipRes.json();

        setPerf(perfData);
        setEquityData(equityJson.data || []);
        setAllTrades(allTradesData.trades || []);
        setTradeTotal(allTradesData.total ?? 0);
        setAdvancedMetrics(metricsData.total_trades ? metricsData : null);
        setSignalCombos(combosData.combos || []);
        setSkipMetrics(skipData.total_evaluations != null ? skipData : null);
      } catch { /* fetch failed */ }
    };

    fetchCharts();
    const interval = setInterval(fetchCharts, 10000);
    return () => clearInterval(interval);
  }, [strategy]);

  // Fetch paginated table data (depends on strategy + page)
  const fetchPagedTrades = useCallback(async () => {
    try {
      const tradeOffset = (tradePage - 1) * TRADES_PER_PAGE;
      const res = await fetch(`/api/trades?limit=${TRADES_PER_PAGE}&offset=${tradeOffset}&strategy=${strategy}`);
      const data = await res.json();
      setTrades(data.trades || []);
    } catch { /* fetch failed */ }
  }, [strategy, tradePage]);

  useEffect(() => {
    fetchPagedTrades();
    const interval = setInterval(fetchPagedTrades, 10000);
    return () => clearInterval(interval);
  }, [fetchPagedTrades]);

  const winRate = perf && safeNum(perf.trade_count) > 0
    ? ((safeNum(perf.win_count) / safeNum(perf.trade_count)) * 100).toFixed(1)
    : '—';

  // Build daily P&L bar chart data from all trades (not paginated)
  const dailyPnlMap = new Map<string, number>();
  for (const t of allTrades) {
    const day = (t.entry_time || '').split(/[T ]/)[0];
    if (!day) continue;
    dailyPnlMap.set(day, (dailyPnlMap.get(day) || 0) + safeNum(t.pnl));
  }
  const dailyPnlData = Array.from(dailyPnlMap.entries())
    .map(([date, pnl]) => ({ date, pnl: Number(pnl.toFixed(2)) }))
    .sort((a, b) => a.date.localeCompare(b.date));

  // Win rate by entry minute (from all trades)
  const minuteWins = new Map<number, { wins: number; total: number }>();
  for (const t of allTrades) {
    const m = t.window_minute ?? 0;
    const entry = minuteWins.get(m) || { wins: 0, total: 0 };
    entry.total++;
    if (safeNum(t.pnl) > 0) entry.wins++;
    minuteWins.set(m, entry);
  }
  const winByMinute = Array.from(minuteWins.entries())
    .map(([minute, { wins, total }]) => ({
      minute: `Min ${minute}`,
      win_rate: total > 0 ? Number(((wins / total) * 100).toFixed(1)) : 0,
      total,
    }))
    .sort((a, b) => {
      const numA = parseInt(a.minute.replace('Min ', ''), 10);
      const numB = parseInt(b.minute.replace('Min ', ''), 10);
      return numA - numB;
    });

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold text-zinc-50">Performance Analytics</h1>

      {/* Strategy Summary */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard title="Total P&L" value={perf ? `$${safeNum(perf.total_pnl).toFixed(2)}` : '—'} trend={perf && safeNum(perf.total_pnl) >= 0 ? 'up' : 'down'} />
        <StatCard title="Win Rate" value={winRate !== '—' ? `${winRate}%` : '—'} />
        <StatCard title="Best Trade" value={perf ? `$${safeNum(perf.best_trade).toFixed(2)}` : '—'} trend="up" />
        <StatCard title="Worst Trade" value={perf ? `$${safeNum(perf.worst_trade).toFixed(2)}` : '—'} trend="down" />
        <StatCard title="Avg Position" value={perf ? `$${safeNum(perf.avg_position_size).toFixed(2)}` : '—'} />
        <StatCard title="Avg Confidence" value={perf ? `${(safeNum(perf.avg_confidence) * 100).toFixed(1)}%` : '—'} />
      </div>

      {/* Advanced Metrics Grid */}
      <MetricsGrid metrics={advancedMetrics} />

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

      {/* Signal Combo Chart + Table */}
      <SignalComboChart combos={signalCombos} />

      {/* Skip Metrics Analytics */}
      <SkipMetricsPanel metrics={skipMetrics} />

      {/* Trade History */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <TradesTable
            trades={trades}
            page={tradePage}
            totalPages={Math.max(1, Math.ceil(tradeTotal / TRADES_PER_PAGE))}
            onPageChange={setTradePage}
          />
        </CardContent>
      </Card>
    </div>
  );
}
