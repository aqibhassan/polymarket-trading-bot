'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { SignalBreakdown } from '@/lib/types/bot-state';

interface SignalPanelProps {
  signals: SignalBreakdown | null;
}

function directionBadge(dir: string) {
  if (dir === 'YES') return <Badge variant="success" className="text-[10px] px-1.5">YES</Badge>;
  if (dir === 'NO') return <Badge variant="destructive" className="text-[10px] px-1.5">NO</Badge>;
  return <Badge variant="outline" className="text-[10px] px-1.5">--</Badge>;
}

export function SignalPanel({ signals }: SignalPanelProps) {
  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Signal Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        {!signals || signals.votes.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-zinc-800 rounded-lg gap-2">
            <p className="text-sm text-zinc-500">Waiting for entry zone</p>
            <p className="text-xs text-zinc-600">Signals appear at minute 8-12</p>
          </div>
        ) : (
          <div className="space-y-3">
            {signals.votes.map((vote) => (
              <div key={vote.name} className="space-y-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-zinc-300 w-28 truncate">{vote.name}</span>
                    {directionBadge(vote.direction)}
                  </div>
                  <span className="text-xs font-mono text-zinc-400">{(vote.strength ?? 0).toFixed(0)}%</span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      vote.direction === 'YES' ? 'bg-emerald-500' :
                      vote.direction === 'NO' ? 'bg-red-500' : 'bg-zinc-600'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, vote.strength ?? 0))}%` }}
                  />
                </div>
              </div>
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
