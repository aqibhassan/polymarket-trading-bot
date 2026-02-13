'use client';

import { StatCard } from '@/components/charts/stat-card';
import type { BWOAdvancedMetrics } from '@/lib/types/bwo';
import { Activity, TrendingUp, TrendingDown, BarChart3, Target, Layers, Zap, DollarSign } from 'lucide-react';

interface MetricsGridProps {
  metrics: BWOAdvancedMetrics | null;
}

/** Safe toFixed that handles null/undefined/NaN. */
function safe(val: number | null | undefined, digits: number = 2): string {
  if (val == null || Number.isNaN(val)) return '0';
  if (!Number.isFinite(val)) return '\u221E';
  return val.toFixed(digits);
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  if (!metrics) {
    return null;
  }

  const pf = metrics.profit_factor ?? 0;
  const sharpe = metrics.sharpe_ratio ?? 0;
  const sortino = metrics.sortino_ratio ?? 0;
  const maxDd = metrics.max_drawdown_pct ?? 0;
  const maxConsec = metrics.max_consec_losses ?? 0;
  const totalFees = metrics.total_fees ?? 0;
  const exp = metrics.expectancy ?? 0;
  const tpd = metrics.avg_trades_per_day ?? 0;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        title="Profit Factor"
        value={pf >= 9999 ? '\u221E' : safe(pf)}
        trend={pf >= 1.5 ? 'up' : pf >= 1 ? 'neutral' : 'down'}
        icon={<TrendingUp className="h-4 w-4" />}
      />
      <StatCard
        title="Sharpe Ratio"
        value={sharpe >= 9999 ? '\u221E' : safe(sharpe)}
        trend={sharpe >= 1 ? 'up' : sharpe >= 0 ? 'neutral' : 'down'}
        icon={<Activity className="h-4 w-4" />}
      />
      <StatCard
        title="Sortino Ratio"
        value={sortino >= 9999 ? '\u221E' : safe(sortino)}
        trend={sortino >= 1 ? 'up' : sortino >= 0 ? 'neutral' : 'down'}
        icon={<Zap className="h-4 w-4" />}
      />
      <StatCard
        title="Max Drawdown"
        value={`${safe(maxDd)}%`}
        trend={maxDd <= 5 ? 'up' : maxDd <= 15 ? 'neutral' : 'down'}
        icon={<TrendingDown className="h-4 w-4" />}
      />
      <StatCard
        title="Max Consec Losses"
        value={maxConsec}
        icon={<TrendingDown className="h-4 w-4" />}
      />
      <StatCard
        title="Expectancy"
        value={`$${safe(exp, 4)}`}
        trend={exp > 0 ? 'up' : exp === 0 ? 'neutral' : 'down'}
        icon={<Target className="h-4 w-4" />}
      />
      <StatCard
        title="Total Fees"
        value={`$${safe(totalFees)}`}
        icon={<DollarSign className="h-4 w-4" />}
      />
      <StatCard
        title="Avg Trades/Day"
        value={safe(tpd, 1)}
        icon={<BarChart3 className="h-4 w-4" />}
      />
    </div>
  );
}
