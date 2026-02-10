'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { SizingDetails } from '@/lib/types/bot-state';

function safeNum(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

interface SizingCardProps {
  sizing: SizingDetails | null;
}

export function SizingCard({ sizing }: SizingCardProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Position Sizing</CardTitle>
      </CardHeader>
      <CardContent>
        {!sizing ? (
          <div className="flex items-center justify-center h-24 border-2 border-dashed border-zinc-800 rounded-lg">
            <p className="text-sm text-zinc-500">No sizing data</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-zinc-500">Kelly %</p>
                <p className="font-mono text-sm text-zinc-200">
                  {(safeNum(sizing.kelly_fraction) * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Win Prob</p>
                <p className="font-mono text-sm text-zinc-200">
                  {(safeNum(sizing.estimated_win_prob) * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Shares</p>
                <p className="font-mono text-sm text-zinc-200">{safeNum(sizing.recommended_size).toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">USDC Cost</p>
                <p className="font-mono text-sm text-emerald-400">
                  ${(safeNum(sizing.recommended_size) * safeNum(sizing.entry_price)).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Entry Price</p>
                <p className="font-mono text-sm text-zinc-200">${safeNum(sizing.entry_price).toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Max Allowed</p>
                <p className="font-mono text-sm text-zinc-200">{safeNum(sizing.max_allowed).toFixed(2)} shares</p>
              </div>
            </div>

            {/* Visual gauge */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-zinc-500">
                <span>Size vs Max</span>
                <span>{((safeNum(sizing.recommended_size) / Math.max(safeNum(sizing.max_allowed), 0.01)) * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all"
                  style={{
                    width: `${Math.min(100, (safeNum(sizing.recommended_size) / Math.max(safeNum(sizing.max_allowed), 0.01)) * 100)}%`,
                  }}
                />
              </div>
            </div>

            {sizing.capped_reason && sizing.capped_reason !== 'none' && (
              <Badge variant="outline" className="text-[10px]">
                {sizing.capped_reason}
              </Badge>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
