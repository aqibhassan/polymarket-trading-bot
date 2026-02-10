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
import type { SkipMetrics } from '@/lib/types/trade';

interface SkipMetricsPanelProps {
  metrics: SkipMetrics | null;
}

const REASON_DISPLAY: Record<string, string> = {
  insufficient_agreement: 'Low Agreement',
  contrarian_filter: 'Contrarian Filter',
  low_confidence: 'Low Confidence',
  no_votes: 'No Votes',
  entry_signal: 'Entry',
};

const REASON_COLORS: Record<string, string> = {
  insufficient_agreement: '#f59e0b', // amber
  low_confidence: '#3b82f6',         // blue
  contrarian_filter: '#a855f7',      // purple
  no_votes: '#71717a',               // gray
  entry_signal: '#10b981',           // green
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

  // Filter by_minute to minutes 8-12 for the chart
  const minuteData = metrics.by_minute
    .filter((m) => m.minute >= 8 && m.minute <= 12)
    .map((m) => ({
      minute: `Min ${m.minute}`,
      skip_rate: m.skip_rate,
      skips: m.skips,
      entries: m.entries,
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
          title="Total Evaluations"
          value={metrics.total_evaluations.toLocaleString()}
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

      {/* Middle: Two side-by-side charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left: Skip Reasons Breakdown */}
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

        {/* Right: Skip Rate by Entry Minute */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-zinc-400">Skip Rate by Entry Minute</CardTitle>
          </CardHeader>
          <CardContent>
            {minuteData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={minuteData}>
                  <XAxis dataKey="minute" stroke="#71717a" fontSize={11} />
                  <YAxis stroke="#71717a" fontSize={11} tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#18181b',
                      border: '1px solid #27272a',
                      borderRadius: '8px',
                    }}
                    formatter={(v: number, _: string, props: { payload?: { skips: number; entries: number } }) => [
                      `${v}% (${props.payload?.skips ?? 0} skips / ${props.payload?.entries ?? 0} entries)`,
                      'Skip Rate',
                    ]}
                  />
                  <Bar dataKey="skip_rate" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[250px] text-zinc-500">No data</div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
