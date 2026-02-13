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
import { ConfidenceDistribution } from '@/components/charts/confidence-distribution';
import { EntryPriceAnalysis } from '@/components/charts/entry-price-analysis';
import { RollingWinRate } from '@/components/charts/rolling-win-rate';
import { SkipMetricsPanel } from '@/components/charts/skip-metrics-panel';
import { TradesTable } from '@/components/charts/trades-table';
import { ExportButton } from '@/components/charts/export-button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ErrorBanner } from '@/components/ui/error-banner';
import type {
  BWOSummary,
  BWOAdvancedMetrics,
  BWOEquityPoint,
  BWODailyPnl,
  BWOConfidenceBucket,
  BWOEntryPriceBucket,
  BWORollingWinRate,
  BWOSkipMetrics,
  BWOTradeRecord,
} from '@/lib/types/bwo';

export default function PerformancePage() {
  const [summary, setSummary] = useState<BWOSummary | null>(null);
  const [metrics, setMetrics] = useState<BWOAdvancedMetrics | null>(null);
  const [equityData, setEquityData] = useState<BWOEquityPoint[]>([]);
  const [dailyPnl, setDailyPnl] = useState<BWODailyPnl[]>([]);
  const [confidenceDist, setConfidenceDist] = useState<BWOConfidenceBucket[]>([]);
  const [entryPrice, setEntryPrice] = useState<BWOEntryPriceBucket[]>([]);
  const [rollingWR, setRollingWR] = useState<BWORollingWinRate[]>([]);
  const [skipMetrics, setSkipMetrics] = useState<BWOSkipMetrics | null>(null);
  const [trades, setTrades] = useState<BWOTradeRecord[]>([]);
  const [tradeTotal, setTradeTotal] = useState(0);
  const [tradePage, setTradePage] = useState(1);
  const TRADES_PER_PAGE = 20;
  const [error, setError] = useState<string | null>(null);

  const fetchCharts = useCallback(async () => {
    try {
      const [sumRes, metRes, eqRes, dpRes, cdRes, epRes, rwRes, skRes] = await Promise.all([
        fetch('/api/summary'),
        fetch('/api/metrics'),
        fetch('/api/equity-curve'),
        fetch('/api/daily-pnl'),
        fetch('/api/confidence-dist'),
        fetch('/api/entry-price'),
        fetch('/api/rolling-wr'),
        fetch('/api/skip-metrics'),
      ]);
      const sumData = await sumRes.json();
      const metData = await metRes.json();
      const eqData = await eqRes.json();
      const dpData = await dpRes.json();
      const cdData = await cdRes.json();
      const epData = await epRes.json();
      const rwData = await rwRes.json();
      const skData = await skRes.json();

      setSummary(sumData);
      setMetrics(metData.total_trades != null ? metData : null);
      setEquityData(eqData.data || eqData || []);
      setDailyPnl(dpData.data || dpData || []);
      setConfidenceDist(cdData.data || cdData || []);
      setEntryPrice(epData.data || epData || []);
      setRollingWR(rwData.data || rwData || []);
      setSkipMetrics(skData.total_windows != null ? skData : null);
      setError(null);
    } catch {
      setError('Failed to load performance data');
    }
  }, []);

  useEffect(() => {
    fetchCharts();
    const interval = setInterval(fetchCharts, 10000);
    return () => clearInterval(interval);
  }, [fetchCharts]);

  // Paginated trades
  const fetchPagedTrades = useCallback(async () => {
    try {
      const offset = (tradePage - 1) * TRADES_PER_PAGE;
      const res = await fetch(`/api/trades?limit=${TRADES_PER_PAGE}&offset=${offset}`);
      const data = await res.json();
      setTrades(data.trades || []);
      setTradeTotal(data.total ?? 0);
    } catch {
      setError('Failed to load trades');
    }
  }, [tradePage]);

  useEffect(() => {
    fetchPagedTrades();
    const interval = setInterval(fetchPagedTrades, 10000);
    return () => clearInterval(interval);
  }, [fetchPagedTrades]);

  const winRate = summary && summary.total_trades > 0
    ? (summary.win_rate * 100).toFixed(1)
    : '--';

  // Avg position size: avg(shares * entry_price) from summary
  const avgPosition = summary && summary.total_trades > 0
    ? (summary.bankroll / summary.total_trades).toFixed(2)
    : '--';

  // Avg confidence from trades
  const avgConfidence = trades.length > 0
    ? (trades.reduce((s, t) => s + t.cont_prob, 0) / trades.length * 100).toFixed(1)
    : '--';

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold text-zinc-50">Performance Analytics</h1>
      {error && <ErrorBanner message={error} />}

      {/* Summary Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard
          title="Total P&L"
          value={summary ? `${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl.toFixed(2)}` : '--'}
          trend={summary && summary.total_pnl >= 0 ? 'up' : 'down'}
        />
        <StatCard title="Win Rate" value={winRate !== '--' ? `${winRate}%` : '--'} />
        <StatCard
          title="Best Trade"
          value={metrics ? `$${metrics.best_trade.toFixed(2)}` : '--'}
          trend="up"
        />
        <StatCard
          title="Worst Trade"
          value={metrics ? `$${metrics.worst_trade.toFixed(2)}` : '--'}
          trend="down"
        />
        <StatCard title="Avg Position" value={avgPosition !== '--' ? `$${avgPosition}` : '--'} />
        <StatCard title="Avg Confidence" value={avgConfidence !== '--' ? `${avgConfidence}%` : '--'} />
      </div>

      {/* Advanced Metrics Grid */}
      <MetricsGrid metrics={metrics} />

      {/* Full Equity Curve */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Equity Curve (All History)</CardTitle>
        </CardHeader>
        <CardContent>
          <EquityCurve data={equityData} height={350} />
        </CardContent>
      </Card>

      {/* Daily PnL + Confidence Distribution */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Daily P&L</CardTitle>
          </CardHeader>
          <CardContent>
            {dailyPnl.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={dailyPnl}>
                  <XAxis dataKey="date" stroke="#71717a" fontSize={11} tickFormatter={(v: string) => v.slice(5)} />
                  <YAxis stroke="#71717a" fontSize={11} tickFormatter={(v: number) => `$${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '8px' }}
                    formatter={(v: number) => [`$${v.toFixed(2)}`, 'P&L']}
                  />
                  <Bar dataKey="pnl">
                    {dailyPnl.map((entry, i) => (
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

        <ConfidenceDistribution data={confidenceDist} />
      </div>

      {/* Entry Price Analysis + Rolling Win Rate */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <EntryPriceAnalysis data={entryPrice} />
        <RollingWinRate data={rollingWR} />
      </div>

      {/* Skip Metrics */}
      <SkipMetricsPanel metrics={skipMetrics} />

      {/* Trade History */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-zinc-400">Trade History</CardTitle>
            <ExportButton />
          </div>
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
