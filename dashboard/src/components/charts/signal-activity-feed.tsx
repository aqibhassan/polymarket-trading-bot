'use client';

import { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
};

function timeAgo(ts: string): string {
  if (!ts) return '--';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '--';
  const diff = Math.floor((Date.now() - d.getTime()) / 1000);
  if (diff < 0) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

/** Derive display label: actual outcome vs re-evaluation while in position */
function outcomeLabel(evt: SignalActivityEvent): { text: string; style: string } {
  if (evt.has_position) {
    // Re-evaluation while already holding a position
    return { text: 'HELD', style: 'border-blue-600 text-blue-400' };
  }
  if (evt.outcome === 'entry') {
    return { text: 'ENTRY', style: '' }; // uses variant="success"
  }
  return { text: 'SKIP', style: 'border-amber-600 text-amber-400' };
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

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-400">Signal Activity</CardTitle>
      </CardHeader>
      <CardContent>
        {feed.length === 0 ? (
          <div className="flex items-center justify-center h-24 border-2 border-dashed border-zinc-800 rounded-lg">
            <p className="text-sm text-zinc-500 text-center px-4">
              No signal evaluations yet — Activity appears during entry windows (min 6-10)
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
            {feed.map((evt) => {
              const label = outcomeLabel(evt);
              const dimmed = evt.has_position;
              return (
                <div
                  key={evt.id}
                  className={`flex items-center justify-between rounded-md border p-2 ${
                    dimmed ? 'border-zinc-800/50 opacity-60' : 'border-zinc-800'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={!evt.has_position && evt.outcome === 'entry' ? 'success' : 'outline'}
                      className={`text-xs ${label.style}`}
                    >
                      {label.text}
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
                      {evt.has_position
                        ? evt.outcome === 'entry'
                          ? 'Signal confirmed (in position)'
                          : `${REASON_LABELS[evt.reason] || evt.reason} (in position)`
                        : REASON_LABELS[evt.reason] || evt.reason}
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
                    <span className="text-xs text-zinc-500 w-14 text-right">
                      {timeAgo(evt.timestamp)}
                    </span>
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
