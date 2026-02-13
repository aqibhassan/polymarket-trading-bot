'use client';

import { useState, useEffect, useCallback } from 'react';
import { useBwoState } from '@/lib/hooks/use-bwo-state';
import { useBwoTrades } from '@/lib/hooks/use-bwo-trades';
import { StatCard } from '@/components/charts/stat-card';
import { EquityCurve } from '@/components/charts/equity-curve';
import { TradesTable } from '@/components/charts/trades-table';
import { Badge } from '@/components/ui/badge';
import { ErrorBanner } from '@/components/ui/error-banner';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DollarSign, TrendingUp, Trophy, BarChart3, Activity } from 'lucide-react';
import type { BWOEquityPoint, BWODailyPnl, BWOSkipMetrics } from '@/lib/types/bwo';

export default function OverviewPage() {
  const { summary, health, connected, error: stateError, refresh } = useBwoState();
  const { trades, page: tradePage, totalPages: tradeTotalPages, total: tradeTotal, setPage: setTradePage } = useBwoTrades(10);
  const [equityData, setEquityData] = useState<BWOEquityPoint[]>([]);
  const [dailyPnl, setDailyPnl] = useState<BWODailyPnl[]>([]);
  const [skipMetrics, setSkipMetrics] = useState<BWOSkipMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchExtra = useCallback(async () => {
    try {
      const [eqRes, dpRes, skRes] = await Promise.all([
        fetch('/api/equity-curve'),
        fetch('/api/daily-pnl'),
        fetch('/api/skip-metrics'),
      ]);
      const eqData = await eqRes.json();
      const dpData = await dpRes.json();
      const skData = await skRes.json();
      setEquityData(eqData.data || eqData || []);
      setDailyPnl(dpData.data || dpData || []);
      setSkipMetrics(skData.total_windows != null ? skData : null);
      setError(null);
    } catch {
      setError('Failed to load chart data');
    }
  }, []);

  useEffect(() => {
    fetchExtra();
    const interval = setInterval(fetchExtra, 10000);
    return () => clearInterval(interval);
  }, [fetchExtra]);

  // Today stats from daily PnL
  const today = new Date().toISOString().split('T')[0];
  const todayData = dailyPnl.find((d) => d.date === today);
  const todayPnl = todayData?.pnl ?? 0;
  const todayTrades = todayData?.trades ?? 0;
  const todayWins = todayData?.wins ?? 0;
  const todayLosses = todayData?.losses ?? 0;
  const todayWR = todayTrades > 0 ? ((todayWins / todayTrades) * 100).toFixed(1) : '--';

  // All-time stats from summary
  const bankroll = summary?.bankroll ?? 0;
  const totalPnl = summary?.total_pnl ?? 0;
  const winRate = summary?.win_rate ?? 0;
  const totalTrades = summary?.total_trades ?? 0;
  const evPerTrade = summary?.ev_per_trade ?? 0;

  const pnlTrend = totalPnl >= 0 ? 'up' as const : 'down' as const;

  // Last 5 trades for mini table
  const last5 = trades.slice(0, 5);

  const displayError = stateError || error;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h1 className="text-xl md:text-2xl font-bold text-zinc-50">BWO 5m Paper Trader</h1>
        </div>
        <Badge variant={connected ? 'success' : 'destructive'}>
          {connected ? 'Connected' : 'Disconnected'}
        </Badge>
      </div>

      {displayError && <ErrorBanner message={displayError} onRetry={refresh} />}

      {/* All Time Stats */}
      <div>
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">All Time</p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Balance"
            value={`$${bankroll.toFixed(2)}`}
            trend={pnlTrend}
            icon={<DollarSign className="h-4 w-4" />}
          />
          <StatCard
            title="Total P&L"
            value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`}
            trend={pnlTrend}
            icon={<TrendingUp className="h-4 w-4" />}
          />
          <StatCard
            title="Win Rate"
            value={totalTrades > 0 ? `${(winRate * 100).toFixed(1)}%` : '--'}
            subtitle={totalTrades > 0 ? `${summary?.wins ?? 0}W / ${summary?.losses ?? 0}L` : ''}
            icon={<Trophy className="h-4 w-4" />}
          />
          <StatCard
            title="Total Trades"
            value={totalTrades}
            icon={<BarChart3 className="h-4 w-4" />}
          />
        </div>
      </div>

      {/* Today Stats */}
      <div>
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2">Today</p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Daily P&L"
            value={`${todayPnl >= 0 ? '+' : ''}$${todayPnl.toFixed(2)}`}
            trend={todayPnl >= 0 ? 'up' : 'down'}
            icon={<TrendingUp className="h-4 w-4" />}
          />
          <StatCard
            title="Win Rate (Today)"
            value={todayWR !== '--' ? `${todayWR}%` : '--'}
            subtitle={todayTrades > 0 ? `${todayWins}W / ${todayLosses}L` : ''}
            icon={<Trophy className="h-4 w-4" />}
          />
          <StatCard
            title="Trades Today"
            value={todayTrades}
            icon={<BarChart3 className="h-4 w-4" />}
          />
          <StatCard
            title="EV/Trade"
            value={totalTrades > 0 ? `$${evPerTrade.toFixed(2)}` : '--'}
            trend={evPerTrade >= 0 ? 'up' : 'down'}
            icon={<Activity className="h-4 w-4" />}
          />
        </div>
      </div>

      {/* Equity Curve */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Equity Curve</CardTitle>
        </CardHeader>
        <CardContent>
          <EquityCurve data={equityData} height={300} />
        </CardContent>
      </Card>

      {/* Two-col: Last 5 trades + Skip Rate */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Last 5 Trades */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Last 5 Trades</CardTitle>
          </CardHeader>
          <CardContent>
            {last5.length > 0 ? (
              <div className="space-y-2">
                {last5.map((t, i) => (
                  <div key={i} className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
                    <div className="flex items-center gap-2">
                      <Badge variant={t.side === 'YES' ? 'success' : 'destructive'} className="text-xs">
                        {t.side}
                      </Badge>
                      <span className="text-xs text-zinc-400 font-mono">${t.entry_price.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs font-mono font-semibold ${t.pnl_net >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {t.pnl_net >= 0 ? '+' : ''}${t.pnl_net.toFixed(2)}
                      </span>
                      <Badge variant={t.correct ? 'success' : 'destructive'} className="text-xs">
                        {t.correct ? 'WIN' : 'LOSS'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-32 text-zinc-500">No trades yet</div>
            )}
          </CardContent>
        </Card>

        {/* Skip Rate Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Skip Rate Summary</CardTitle>
          </CardHeader>
          <CardContent>
            {skipMetrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-zinc-500 uppercase">Total Windows</p>
                    <p className="text-xl font-bold text-zinc-50">{skipMetrics.total_windows}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 uppercase">Entries</p>
                    <p className="text-xl font-bold text-emerald-400">{skipMetrics.total_entries}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 uppercase">Skips</p>
                    <p className="text-xl font-bold text-zinc-400">{skipMetrics.total_skips}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 uppercase">Skip Rate</p>
                    <p className="text-xl font-bold text-zinc-50">{skipMetrics.skip_rate.toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-32 text-zinc-500">No skip data</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Full Trades Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <TradesTable trades={trades} page={tradePage} totalPages={tradeTotalPages} onPageChange={setTradePage} />
        </CardContent>
      </Card>
    </div>
  );
}
