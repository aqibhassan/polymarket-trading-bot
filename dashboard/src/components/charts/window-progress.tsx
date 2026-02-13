'use client';

import { cn } from '@/lib/utils';

interface WindowProgressProps {
  minute: number;
  entryZoneStart?: number;
  entryZoneEnd?: number;
}

export function WindowProgress({ minute, entryZoneStart = 8, entryZoneEnd = 12 }: WindowProgressProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-zinc-400">Window Progress</span>
        <span className="text-zinc-300 font-mono">Minute {minute}/14</span>
      </div>
      <div className="flex gap-1">
        {Array.from({ length: 15 }, (_, i) => (
          <div
            key={i}
            className={cn(
              'h-8 flex-1 rounded-sm transition-colors relative',
              i < minute ? 'bg-zinc-600' : i === minute ? 'bg-emerald-500 animate-pulse' : 'bg-zinc-800',
              i >= entryZoneStart && i <= entryZoneEnd && i !== minute && 'ring-1 ring-blue-500/50',
            )}
          >
            {i === entryZoneStart && (
              <span className="absolute -top-5 left-0 text-[10px] text-blue-400">Entry Zone</span>
            )}
          </div>
        ))}
      </div>
      <div className="flex justify-between text-[10px] text-zinc-500">
        <span>0</span>
        <span>5</span>
        <span>10</span>
        <span>14</span>
      </div>
    </div>
  );
}
