'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { BotPosition } from '@/lib/types/bot-state';

interface PositionCardProps {
  position: BotPosition | null;
}

function safeNum(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function timeSince(iso: string): string {
  if (!iso) return '--';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '--';
  const diff = Math.floor((Date.now() - d.getTime()) / 1000);
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
  return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m`;
}

export function PositionCard({ position }: PositionCardProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Active Position</CardTitle>
      </CardHeader>
      <CardContent>
        {position ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge variant={position.side === 'YES' ? 'success' : 'destructive'}>
                {position.side}
              </Badge>
              <span className="text-xs text-zinc-500">{position.market_id}</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-zinc-500">Entry Price</p>
                <p className="font-mono text-sm text-zinc-200">${safeNum(position.entry_price).toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Size</p>
                <p className="font-mono text-sm text-zinc-200">${safeNum(position.size).toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Entry Time</p>
                <p className="font-mono text-sm text-zinc-200">{new Date(position.entry_time).toLocaleTimeString()}</p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Duration</p>
                <p className="font-mono text-sm text-zinc-200">{timeSince(position.entry_time)}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-24 border-2 border-dashed border-zinc-800 rounded-lg">
            <p className="text-sm text-zinc-500">No active position</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
