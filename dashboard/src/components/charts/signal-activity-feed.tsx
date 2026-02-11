'use client';

import { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { formatUTCTime } from '@/lib/format';
import type { SignalActivityEvent } from '@/lib/types/bot-state';

interface SignalActivityFeedProps {
  activity: SignalActivityEvent[] | null;
}

const REASON_LABELS: Record<string, string> = {
  no_votes: 'No signal votes',
  insufficient_agreement: 'Insufficient agreement',
  contrarian_filter: 'Contrarian filter',
  low_confidence: 'Low confidence',
  entry_signal: 'Entry generated',
  clob_spread_too_wide: 'CLOB spread too wide',
  clob_entry_price_too_high: 'CLOB price too high',
  max_clob_entry_price: 'Price exceeds max',
  no_clob_price: 'No CLOB price available',
};

function VoteSummary({ votes }: { votes: Record<string, number> }) {
  const entries = Object.entries(votes);
  if (entries.length === 0) return null;
  return (
    <div className="flex gap-1 flex-wrap mt-1">
      {entries.map(([name, val]) => {
        const dir = val > 0 ? 'YES' : val < 0 ? 'NO' : 'neutral';
        const color = dir === 'YES' ? 'text-emerald-400' : dir === 'NO' ? 'text-red-400' : 'text-zinc-500';
        return (
          <span key={name} className={`text-[10px] font-mono ${color}`}>
            {name}={val > 0 ? '+' : ''}{typeof val === 'number' ? val.toFixed(2) : val}
          </span>
        );
      })}
    </div>
  );
}

export function SignalActivityFeed({ activity }: SignalActivityFeedProps) {
  const [feed, setFeed] = useState<SignalActivityEvent[]>([]);
  const seenIds = useRef(new Set<string>());

  // Sync from SSE data
  useEffect(() => {
    if (!activity || activity.length === 0) return;
    let changed = false;
    for (const evt of activity) {
      if (!seenIds.current.has(evt.id)) {
        seenIds.current.add(evt.id);
        changed = true;
      }
    }
    if (changed) {
      const merged = new Map<string, SignalActivityEvent>();
      for (const evt of activity) merged.set(evt.id, evt);
      for (const evt of feed) {
        if (!merged.has(evt.id)) merged.set(evt.id, evt);
      }
      setFeed(Array.from(merged.values()).slice(0, 20));
    }
  }, [activity]); // eslint-disable-line react-hooks/exhaustive-deps

  // Compute summary stats from feed
  const skipCount = feed.filter((e) => e.outcome === 'skip').length;
  const entryCount = feed.filter((e) => e.outcome === 'entry').length;
  const rejectedCount = feed.filter((e) => e.outcome === 'rejected').length;
  const feedTotal = skipCount + entryCount + rejectedCount;
  const feedSkipRate = feedTotal > 0 ? ((skipCount / feedTotal) * 100).toFixed(0) : '0';

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Signal Activity</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Compact summary bar */}
        {feed.length > 0 && (
          <div className="flex items-center gap-2 mb-3 text-xs flex-wrap">
            <span className="text-zinc-500">Recent:</span>
            <Badge variant="outline" className="border-amber-600 text-amber-400 text-xs">
              {skipCount} skips
            </Badge>
            {rejectedCount > 0 && (
              <Badge variant="destructive" className="text-xs">
                {rejectedCount} rejected
              </Badge>
            )}
            <span className="text-zinc-600">/</span>
            <Badge variant="success" className="text-xs">
              {entryCount} entries
            </Badge>
            <span className="text-zinc-500">
              ({feedSkipRate}% skip rate)
            </span>
          </div>
        )}

        {feed.length === 0 ? (
          <div className="flex items-center justify-center h-24 border-2 border-dashed border-zinc-800 rounded-lg">
            <p className="text-sm text-zinc-500 text-center px-4">
              No signal evaluations yet — one result per 15-minute window
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
            {feed.map((evt) => {
              const isEntry = evt.outcome === 'entry';
              const isRejected = evt.outcome === 'rejected';
              const badgeVariant = isEntry ? 'success' : isRejected ? 'destructive' : 'outline';
              const badgeClass = isEntry ? '' : isRejected ? '' : 'border-amber-600 text-amber-400';
              const badgeLabel = isEntry ? 'ENTRY' : isRejected ? 'REJECTED' : 'SKIP';
              const borderClass = isRejected ? 'border-red-800/50' : 'border-zinc-800';

              return (
                <div
                  key={evt.id}
                  className={`rounded-md border ${borderClass} p-2`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={badgeVariant}
                        className={`text-xs ${badgeClass}`}
                      >
                        {badgeLabel}
                      </Badge>
                      {evt.direction && (
                        <Badge
                          variant={evt.direction === 'YES' ? 'success' : 'destructive'}
                          className="text-xs"
                        >
                          {evt.direction}
                        </Badge>
                      )}
                      <span className="text-xs text-zinc-500">
                        {REASON_LABELS[evt.reason] || evt.reason}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-right">
                      {evt.confidence > 0 && (
                        <span className="font-mono text-sm text-zinc-300">
                          {(evt.confidence * 100).toFixed(1)}%
                        </span>
                      )}
                      <span className="text-xs text-zinc-500">
                        m{evt.minute}
                      </span>
                      <span className="text-xs text-zinc-500 w-20 text-right">
                        {formatUTCTime(evt.timestamp)}
                      </span>
                    </div>
                  </div>
                  {/* Detail row — votes and additional context */}
                  {(evt.detail || (evt.votes && Object.keys(evt.votes).length > 0)) && (
                    <div className="mt-1 pl-1">
                      {evt.votes && Object.keys(evt.votes).length > 0 && (
                        <VoteSummary votes={evt.votes} />
                      )}
                      {evt.detail && (
                        <p className="text-[10px] text-zinc-600 mt-0.5 font-mono truncate">
                          {evt.detail}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
