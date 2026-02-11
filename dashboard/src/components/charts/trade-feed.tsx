'use client';

import { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { timeAgo } from '@/lib/format';
import type { LastTrade, BotPosition } from '@/lib/types/bot-state';

interface TradeFeedProps {
  lastTrade: LastTrade | null;
  activePosition?: BotPosition | null;
}

interface FeedEntry extends LastTrade {
  receivedAt: number;
}

export function TradeFeed({ lastTrade, activePosition }: TradeFeedProps) {
  const [feed, setFeed] = useState<FeedEntry[]>([]);
  const seenIds = useRef(new Set<string>());
  const initialized = useRef(false);

  // Seed feed from recent trades API on mount
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    const seedFeed = async () => {
      try {
        const res = await fetch('/api/trades?limit=20');
        const data = await res.json();
        const trades = data.trades || [];
        const entries: FeedEntry[] = trades.map((t: { trade_id: string; direction: string; entry_price: string; exit_price: string; pnl: string; exit_reason: string; window_minute: number; confidence: number; exit_time: string }) => {
          seenIds.current.add(t.trade_id);
          return {
            trade_id: t.trade_id,
            direction: t.direction,
            entry_price: t.entry_price,
            exit_price: t.exit_price,
            pnl: t.pnl,
            exit_reason: t.exit_reason,
            window_minute: t.window_minute,
            confidence: t.confidence,
            timestamp: t.exit_time,
            receivedAt: Date.now(),
          };
        });
        setFeed(entries.slice(0, 20));
      } catch { /* seed failed, will populate from SSE */ }
    };
    seedFeed();
  }, []);

  // Add new trades from SSE
  useEffect(() => {
    if (!lastTrade || seenIds.current.has(lastTrade.trade_id)) return;
    seenIds.current.add(lastTrade.trade_id);
    setFeed((prev) => [{ ...lastTrade, receivedAt: Date.now() }, ...prev].slice(0, 20));
  }, [lastTrade]);

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Live Trade Feed</CardTitle>
      </CardHeader>
      <CardContent>
        {activePosition && (
          <div className={`flex items-center justify-between rounded-md border p-2 mb-2 ${
            activePosition.status === 'gtc_pending'
              ? 'border-yellow-700/50 bg-yellow-900/20'
              : 'border-amber-700/50 bg-amber-900/20 animate-pulse'
          }`}>
            <div className="flex items-center gap-2">
              <Badge variant={activePosition.side === 'YES' ? 'success' : 'destructive'} className="text-xs">
                {activePosition.side}
              </Badge>
              <span className={`text-xs font-medium ${
                activePosition.status === 'gtc_pending' ? 'text-yellow-400' : 'text-amber-400'
              }`}>
                {activePosition.status === 'gtc_pending' ? 'PENDING' : 'OPEN'}
              </span>
              {activePosition.status === 'gtc_pending' && (
                <span className="text-[10px] text-zinc-500" title="Good-Til-Cancelled limit order waiting to fill">GTC</span>
              )}
            </div>
            <div className="flex items-center gap-3 text-right">
              <span className="font-mono text-sm text-zinc-300">
                @{Number(activePosition.entry_price).toFixed(4)}
              </span>
              <span className="text-xs text-zinc-500 w-14 text-right">{timeAgo(activePosition.entry_time)}</span>
            </div>
          </div>
        )}
        {feed.length === 0 && !activePosition ? (
          <div className="flex items-center justify-center h-40 border-2 border-dashed border-zinc-800 rounded-lg">
            <p className="text-sm text-zinc-500">No trades yet</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
            {feed.map((t) => {
              const raw = Number(t.pnl);
              const pnl = Number.isFinite(raw) ? raw : 0;
              return (
                <div
                  key={t.trade_id}
                  className="flex items-center justify-between rounded-md border border-zinc-800 p-2"
                >
                  <div className="flex items-center gap-2">
                    <Badge variant={t.direction === 'YES' ? 'success' : 'destructive'} className="text-xs">
                      {t.direction}
                    </Badge>
                    <span className="text-xs text-zinc-500">{t.exit_reason}</span>
                  </div>
                  <div className="flex items-center gap-3 text-right">
                    <span className={`font-mono text-sm ${pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                    </span>
                    <span className="text-xs text-zinc-500 w-14 text-right">{timeAgo(t.timestamp)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
