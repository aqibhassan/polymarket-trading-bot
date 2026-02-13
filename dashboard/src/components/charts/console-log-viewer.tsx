'use client';

import { useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface ConsoleLogViewerProps {
  lines: string[];
  maxHeight?: number;
}

function getLineColor(line: string): string {
  if (line.includes('[BUY]')) return 'text-emerald-400';
  if (line.includes('[SKIP]')) return 'text-amber-400';
  if (line.includes('[SETTLE]')) return 'text-blue-400';
  if (line.includes('[ERROR]')) return 'text-red-400';
  if (line.includes('[WARN]')) return 'text-amber-400';
  return 'text-zinc-400';
}

export function ConsoleLogViewer({ lines, maxHeight = 500 }: ConsoleLogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium text-zinc-400">Console Output</CardTitle>
      </CardHeader>
      <CardContent>
        {lines.length === 0 ? (
          <div className="flex items-center justify-center h-[250px] text-zinc-500">
            No console output
          </div>
        ) : (
          <div
            ref={containerRef}
            className="bg-zinc-950 rounded-lg border border-zinc-800 p-3 overflow-y-auto font-mono text-xs leading-5"
            style={{ maxHeight }}
          >
            {lines.map((line, i) => (
              <div key={i} className={getLineColor(line)}>
                {line}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
