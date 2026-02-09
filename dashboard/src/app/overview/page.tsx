'use client';

import { useState, useEffect, useCallback } from 'react';
import { useBotState } from '@/lib/hooks/use-bot-state';
import { useTrades } from '@/lib/hooks/use-trades';
import { StatCard } from '@/components/charts/stat-card';
import { MarketStatePanel } from '@/components/charts/market-state-panel';
import { SignalPanel } from '@/components/charts/signal-panel';
import { TradeFeed } from '@/components/charts/trade-feed';
import { EquityCurve } from '@/components/charts/equity-curve';
import { PositionCard } from '@/components/charts/position-card';
import { SizingCard } from '@/components/charts/sizing-card';
import { TradesTable } from '@/components/charts/trades-table';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DollarSign, TrendingUp, Trophy, BarChart3 } from 'lucide-react';
import type { DailyPnl } from '@/lib/types/trade';

const INITIAL_BALANCE = 10000;

export default function OverviewPage() {
  const { state, connected } = useBotState();
  const { trades } = useTrades(10, 10000);
  const [equityData, setEquityData] = useState<Array<{ time: string; cumulative_pnl: number }>>([]);
  const [killSwitch, setKillSwitch] = useState(false);
  // ClickHouse fallback for daily stats when Redis is empty/stale
  const [chDaily, setChDaily] = useState<DailyPnl | null>(null);
  const [chTotalPnl, setChTotalPnl] = useState<number>(0);

  const fetchClickhouseFallback = useCallback(async () => {
    try {
      const today = new Date().toISOString().split('T')[0];
      const res = await fetch(`/api/daily-pnl?date=${today}`);
      const data = await res.json();
      const rows: DailyPnl[] = data.data || [];
      if (rows.length > 0) {
        // Sum across strategies for totals
        const combined: DailyPnl = {
          strategy: 'all',
          trade_count: rows.reduce((s, r) => s + Number(r.trade_count), 0),
          total_pnl: rows.reduce((s, r) => s + Number(r.total_pnl), 0),
          total_fees: rows.reduce((s, r) => s + Number(r.total_fees), 0),
          win_count: rows.reduce((s, r) => s + Number(r.win_count), 0),
        };
        setChDaily(combined);
      }
    } catch { /* ignore */ }

    try {
      const eqRes = await fetch('/api/equity-curve');
      const eqData = await eqRes.json();
      const curve = eqData.data || [];
      setEquityData(curve);
      if (curve.length > 0) {
        setChTotalPnl(Number(curve[curve.length - 1].cumulative_pnl));
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    const fetchKillSwitch = async () => {
      try {
        const res = await fetch('/api/kill-switch');
        const data = await res.json();
        setKillSwitch(data.active || false);
      } catch { /* ignore */ }
    };

    fetchClickhouseFallback();
    fetchKillSwitch();
    const interval = setInterval(() => {
      fetchClickhouseFallback();
      fetchKillSwitch();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchClickhouseFallback]);

  // Always use ClickHouse for daily stats — Redis counters reset on bot restart
  // and may undercount trades. ClickHouse has the full picture.
  const dailyPnl = chDaily?.total_pnl ?? 0;
  const tradeCount = chDaily?.trade_count ?? 0;
  const winCount = chDaily?.win_count ?? 0;
  const lossCount = tradeCount - winCount;
  const winRate = tradeCount > 0 ? ((winCount / tradeCount) * 100).toFixed(1) : '—';

  // Balance: always compute from initial + ClickHouse total PnL for accuracy
  // Redis balance resets on restart and misses trades from prior runs
  const displayBalance = INITIAL_BALANCE + chTotalPnl;
  const displayPnl = chTotalPnl;
  const displayPnlPct = (chTotalPnl / INITIAL_BALANCE) * 100;

  const pnlTrend = displayPnl >= 0 ? 'up' as const : 'down' as const;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-zinc-50">Live Overview</h1>
        <div className="flex items-center gap-3">
          {killSwitch && (
            <Badge variant="destructive" className="animate-pulse text-sm px-3 py-1">
              KILL SWITCH ACTIVE
            </Badge>
          )}
          <Badge variant={connected ? 'success' : 'destructive'}>
            {connected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Balance"
          value={`$${displayBalance.toFixed(2)}`}
          trend={pnlTrend}
          trendValue={`${displayPnlPct >= 0 ? '+' : ''}${displayPnlPct.toFixed(2)}%`}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          title="Daily P&L"
          value={`$${dailyPnl.toFixed(2)}`}
          trend={dailyPnl >= 0 ? 'up' : 'down'}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatCard
          title="Win Rate"
          value={winRate !== '—' ? `${winRate}%` : '—'}
          subtitle={`${winCount}W / ${lossCount}L`}
          icon={<Trophy className="h-4 w-4" />}
        />
        <StatCard
          title="Trades Today"
          value={tradeCount}
          icon={<BarChart3 className="h-4 w-4" />}
        />
      </div>

      {/* Market State Panel — full width */}
      <MarketStatePanel window={state.window} />

      {/* Signal Panel + Trade Feed */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <SignalPanel signals={state.signals} />
        <TradeFeed lastTrade={state.last_trade} />
      </div>

      {/* Equity Curve + Position + Sizing */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Equity Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <EquityCurve data={equityData} height={280} />
          </CardContent>
        </Card>
        <PositionCard position={state.position} />
        <SizingCard sizing={state.sizing} />
      </div>

      {/* Recent Trades */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <TradesTable trades={trades} />
        </CardContent>
      </Card>
    </div>
  );
}
