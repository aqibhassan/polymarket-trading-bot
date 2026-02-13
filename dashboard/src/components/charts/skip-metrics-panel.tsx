'use client';

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
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { BWOSkipMetrics } from '@/lib/types/bwo';

interface SkipMetricsPanelProps {
  metrics: BWOSkipMetrics | null;
}

const REASON_DISPLAY: Record<string, string> = {
  low_confidence: 'Low Confidence',
  price_too_high: 'Price Too High',
  noise_move: 'Noise Move',
  ev_negative: 'Negative EV',
  no_depth: 'No Depth',
  entry: 'Entry',
  momentum_veto: 'Momentum Veto',
  clob_stability_skip: 'CLOB Unstable',
  coin_flip_zone_skip: 'Coin-Flip Zone',
  calibration_skip_no_edge: 'No Edge',
  desert_skip_paper: 'Desert CLOB',
  fak_no_real_ask_to_cross: 'No Real Ask',
};

const REASON_COLORS: Record<string, string> = {
  low_confidence: '#3b82f6',   // blue
  price_too_high: '#f59e0b',   // amber
  noise_move: '#71717a',       // gray
  ev_negative: '#ef4444',      // red
  no_depth: '#a855f7',         // purple
  entry: '#10b981',            // green
  momentum_veto: '#ef4444',
  clob_stability_skip: '#f97316',
  coin_flip_zone_skip: '#eab308',
  calibration_skip_no_edge: '#6366f1',
  desert_skip_paper: '#71717a',
  fak_no_real_ask_to_cross: '#8b5cf6',
};

function formatReason(reason: string): string {
  return REASON_DISPLAY[reason] || reason.replace(/_/g, ' ');
}

function getReasonColor(reason: string): string {
  return REASON_COLORS[reason] || '#71717a';
}

export function SkipMetricsPanel({ metrics }: SkipMetricsPanelProps) {
  if (!metrics) {
    return null;
  }

  const reasonData = metrics.reasons.map((r) => ({
    reason: formatReason(r.reason),
    count: r.count,
    pct: r.pct,
    rawReason: r.reason,
  }));

  return (
    <div className="space-y-6">
      {/* Top row: 4 StatCards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Skip Rate"
          value={`${metrics.skip_rate.toFixed(1)}%`}
          trend={metrics.skip_rate > 80 ? 'down' : metrics.skip_rate < 50 ? 'up' : 'neutral'}
        />
        <StatCard
          title="Total Windows"
          value={metrics.total_windows.toLocaleString()}
        />
        <StatCard
          title="Total Entries"
          value={metrics.total_entries.toLocaleString()}
          trend="up"
        />
        <StatCard
          title="Total Skips"
          value={metrics.total_skips.toLocaleString()}
        />
      </div>

      {/* Skip Reasons Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Skip Reasons Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          {reasonData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={reasonData} layout="vertical">
                <XAxis type="number" stroke="#71717a" fontSize={11} />
                <YAxis
                  type="category"
                  dataKey="reason"
                  stroke="#71717a"
                  fontSize={11}
                  width={120}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#18181b',
                    border: '1px solid #27272a',
                    borderRadius: '8px',
                  }}
                  formatter={(v: number, _: string, props: { payload?: { pct: number } }) => [
                    `${v} (${props.payload?.pct ?? 0}%)`,
                    'Count',
                  ]}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {reasonData.map((entry, i) => (
                    <Cell key={i} fill={getReasonColor(entry.rawReason)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[250px] text-zinc-500">No data</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
