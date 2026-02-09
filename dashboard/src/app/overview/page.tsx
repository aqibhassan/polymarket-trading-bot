'use client';

import { useState, useEffect } from 'react';
import { useBotState } from '@/lib/hooks/use-bot-state';
import { useTrades } from '@/lib/hooks/use-trades';
import { StatCard } from '@/components/charts/stat-card';
import { WindowProgress } from '@/components/charts/window-progress';
import { EquityCurve } from '@/components/charts/equity-curve';
import { PositionCard } from '@/components/charts/position-card';
import { TradesTable } from '@/components/charts/trades-table';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DollarSign, TrendingUp, Trophy, BarChart3 } from 'lucide-react';

export default function OverviewPage() {
  const { state, connected } = useBotState();
  const { trades } = useTrades(10, 15000);
  const [equityData, setEquityData] = useState<Array<{ time: string; cumulative_pnl: number }>>([]);
  const [killSwitch, setKillSwitch] = useState(false);

  useEffect(() => {
    const fetchEquity = async () => {
      try {
        const res = await fetch('/api/equity-curve');
        const data = await res.json();
        setEquityData(data.data || []);
      } catch { /* failed to fetch equity curve */ }
    };

    const fetchKillSwitch = async () => {
      try {
        const res = await fetch('/api/kill-switch');
        const data = await res.json();
        setKillSwitch(data.active || false);
      } catch { /* failed to fetch kill switch */ }
    };

    fetchEquity();
    fetchKillSwitch();
    const interval = setInterval(() => {
      fetchEquity();
      fetchKillSwitch();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const balance = state.balance;
  const daily = state.daily;
  const window = state.window;

  const winRate = daily && daily.trade_count > 0
    ? ((daily.win_count / daily.trade_count) * 100).toFixed(1)
    : '—';

  const pnlTrend = balance
    ? Number(balance.pnl) >= 0 ? 'up' as const : 'down' as const
    : 'neutral' as const;

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
          value={balance ? `$${Number(balance.balance).toFixed(2)}` : '—'}
          trend={pnlTrend}
          trendValue={balance ? `${balance.pnl_pct >= 0 ? '+' : ''}${balance.pnl_pct.toFixed(2)}%` : undefined}
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatCard
          title="Daily P&L"
          value={daily ? `$${Number(daily.daily_pnl).toFixed(2)}` : '—'}
          trend={daily ? (Number(daily.daily_pnl) >= 0 ? 'up' : 'down') : 'neutral'}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatCard
          title="Win Rate"
          value={winRate !== '—' ? `${winRate}%` : '—'}
          subtitle={daily ? `${daily.win_count}W / ${daily.loss_count}L` : undefined}
          icon={<Trophy className="h-4 w-4" />}
        />
        <StatCard
          title="Trades Today"
          value={daily?.trade_count ?? 0}
          icon={<BarChart3 className="h-4 w-4" />}
        />
      </div>

      {/* Window Progress */}
      {window && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm text-zinc-400">BTC Price</p>
                <p className="text-xl font-mono font-bold text-zinc-50">
                  ${Number(window.btc_close).toLocaleString()}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-zinc-400">Cumulative Return</p>
                <p className={`text-xl font-mono font-bold ${window.cum_return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {window.cum_return_pct >= 0 ? '+' : ''}{window.cum_return_pct.toFixed(4)}%
                </p>
              </div>
            </div>
            <WindowProgress minute={window.minute} />
          </CardContent>
        </Card>
      )}

      {/* Equity Curve + Position */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Equity Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <EquityCurve data={equityData} height={280} />
          </CardContent>
        </Card>
        <PositionCard position={state.position} />
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
