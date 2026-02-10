'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { SignalBreakdown, SignalVote } from '@/lib/types/bot-state';

/** All 5 Singularity signal sources with display names */
const ALL_SIGNALS: { key: string; label: string }[] = [
  { key: 'momentum', label: 'Momentum' },
  { key: 'ofi', label: 'Order Flow (OFI)' },
  { key: 'futures', label: 'Futures Lead' },
  { key: 'vol_regime', label: 'Vol Regime' },
  { key: 'time_of_day', label: 'Time of Day' },
];

interface SignalPanelProps {
  signals: SignalBreakdown | null;
}

function directionBadge(dir: string) {
  if (dir === 'YES') return <Badge variant="success" className="text-[10px] px-1.5">YES</Badge>;
  if (dir === 'NO') return <Badge variant="destructive" className="text-[10px] px-1.5">NO</Badge>;
  if (dir === 'neutral') return <Badge variant="outline" className="text-[10px] px-1.5 text-zinc-400">NTRL</Badge>;
  return <Badge variant="outline" className="text-[10px] px-1.5 text-zinc-600">--</Badge>;
}

function SignalRow({ label, vote }: { label: string; vote: SignalVote | undefined }) {
  const active = vote !== undefined;
  const strength = active ? (vote.strength ?? 0) : 0;
  const direction = active ? vote.direction : '';

  return (
    <div className={`space-y-1 ${active ? '' : 'opacity-40'}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-300 w-28 truncate">{label}</span>
          {directionBadge(direction)}
        </div>
        <span className="text-xs font-mono text-zinc-400">
          {active ? `${strength.toFixed(0)}%` : '—'}
        </span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            direction === 'YES' ? 'bg-emerald-500' :
            direction === 'NO' ? 'bg-red-500' :
            direction === 'neutral' ? 'bg-zinc-500' : 'bg-zinc-800'
          }`}
          style={{ width: `${Math.min(100, Math.max(0, strength))}%` }}
        />
      </div>
    </div>
  );
}

export function SignalPanel({ signals }: SignalPanelProps) {
  const hasVotes = signals && signals.votes.length > 0;

  // Build lookup from vote name → vote data
  const voteMap = new Map<string, SignalVote>();
  if (hasVotes) {
    for (const v of signals.votes) {
      voteMap.set(v.name, v);
    }
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Signal Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        {!hasVotes ? (
          <div className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-zinc-800 rounded-lg gap-2">
            <p className="text-sm text-zinc-500">Waiting for entry zone</p>
            <p className="text-xs text-zinc-600">Signals appear at minute 8-12</p>
          </div>
        ) : (
          <div className="space-y-3">
            {ALL_SIGNALS.map(({ key, label }) => (
              <SignalRow key={key} label={label} vote={voteMap.get(key)} />
            ))}

            <div className="mt-4 pt-3 border-t border-zinc-800">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-zinc-300">Overall</span>
                  {directionBadge(signals.direction)}
                </div>
                <span className="text-sm font-mono font-bold text-zinc-200">
                  {((signals.overall_confidence ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              {signals.entry_generated && (
                <p className="text-[10px] text-emerald-400 mt-1">Entry signal generated</p>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
