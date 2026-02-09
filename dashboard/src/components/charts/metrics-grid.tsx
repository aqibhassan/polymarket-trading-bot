'use client';

import { StatCard } from '@/components/charts/stat-card';
import type { AdvancedMetrics } from '@/lib/types/trade';
import { Activity, TrendingUp, TrendingDown, BarChart3, Clock, Target, Layers, Zap } from 'lucide-react';

interface MetricsGridProps {
  metrics: AdvancedMetrics | null;
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  if (!metrics) {
    return null;
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        title="Profit Factor"
        value={metrics.profit_factor === Infinity ? '\u221E' : metrics.profit_factor.toFixed(2)}
        trend={metrics.profit_factor >= 1.5 ? 'up' : metrics.profit_factor >= 1 ? 'neutral' : 'down'}
        icon={<TrendingUp className="h-4 w-4" />}
      />
      <StatCard
        title="Sharpe Ratio"
        value={metrics.sharpe_ratio.toFixed(2)}
        trend={metrics.sharpe_ratio >= 1 ? 'up' : metrics.sharpe_ratio >= 0 ? 'neutral' : 'down'}
        icon={<Activity className="h-4 w-4" />}
      />
      <StatCard
        title="Sortino Ratio"
        value={metrics.sortino_ratio.toFixed(2)}
        trend={metrics.sortino_ratio >= 1 ? 'up' : metrics.sortino_ratio >= 0 ? 'neutral' : 'down'}
        icon={<Zap className="h-4 w-4" />}
      />
      <StatCard
        title="Max Drawdown"
        value={`${metrics.max_drawdown_pct.toFixed(2)}%`}
        trend={metrics.max_drawdown_pct <= 5 ? 'up' : metrics.max_drawdown_pct <= 15 ? 'neutral' : 'down'}
        icon={<TrendingDown className="h-4 w-4" />}
      />
      <StatCard
        title="Avg Hold Time"
        value={`${metrics.avg_hold_time_minutes.toFixed(1)}m`}
        icon={<Clock className="h-4 w-4" />}
      />
      <StatCard
        title="Expectancy"
        value={`$${metrics.expectancy.toFixed(4)}`}
        trend={metrics.expectancy > 0 ? 'up' : metrics.expectancy === 0 ? 'neutral' : 'down'}
        icon={<Target className="h-4 w-4" />}
      />
      <StatCard
        title="Total Trades"
        value={metrics.total_trades}
        icon={<Layers className="h-4 w-4" />}
      />
      <StatCard
        title="Avg Trades/Day"
        value={metrics.avg_trades_per_day.toFixed(1)}
        icon={<BarChart3 className="h-4 w-4" />}
      />
    </div>
  );
}
