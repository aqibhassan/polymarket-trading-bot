'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ErrorBanner } from '@/components/ui/error-banner';
import { ConsoleLogViewer } from '@/components/charts/console-log-viewer';
import type { BWOHealthStatus } from '@/lib/types/bwo';

function ConnectionIndicator({ label, connected }: { label: string; connected: boolean }) {
  return (
    <div className="flex items-center gap-3 p-4 rounded-lg bg-zinc-800/50">
      <div className={`h-3 w-3 rounded-full ${connected ? 'bg-emerald-500' : 'bg-red-500'}`} />
      <div>
        <p className="text-sm font-medium text-zinc-200">{label}</p>
        <p className={`text-xs ${connected ? 'text-emerald-400' : 'text-red-400'}`}>
          {connected ? 'Connected' : 'Disconnected'}
        </p>
      </div>
    </div>
  );
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m ago`;
}

export default function HealthPage() {
  const [health, setHealth] = useState<BWOHealthStatus | null>(null);
  const [consoleLines, setConsoleLines] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [healthRes, logRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/console-log?lines=200'),
      ]);
      const healthData = await healthRes.json();
      const logData = await logRes.json();
      setHealth(healthData);
      setConsoleLines(logData.lines || []);
      setError(null);
    } catch {
      setError('Failed to load health data');
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const jsonlFresh = health ? health.jsonl_age_s < 600 : false;

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold text-zinc-50">System Health</h1>
      {error && <ErrorBanner message={error} onRetry={fetchData} />}

      {/* Connection Status */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <ConnectionIndicator
          label="Data Server"
          connected={health?.server_ok ?? false}
        />
        <ConnectionIndicator
          label="Paper Trader"
          connected={health?.paper_trader_active ?? false}
        />
        <ConnectionIndicator
          label="JSONL Data"
          connected={jsonlFresh}
        />
      </div>

      {/* Server Info */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Server Info</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <p className="text-xs text-zinc-500 uppercase">Total Records</p>
              <p className="text-xl font-bold text-zinc-50">{health?.total_records ?? '--'}</p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 uppercase">Last Modified</p>
              <p className="text-sm font-medium text-zinc-200">
                {health?.last_modified
                  ? new Date(health.last_modified).toLocaleString('en-GB', { timeZone: 'UTC' }) + ' UTC'
                  : '--'}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500 uppercase">JSONL Age</p>
              <p className={`text-sm font-medium ${jsonlFresh ? 'text-emerald-400' : 'text-red-400'}`}>
                {health ? formatAge(health.jsonl_age_s) : '--'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Console Log */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Console Log</CardTitle>
        </CardHeader>
        <CardContent>
          <ConsoleLogViewer lines={consoleLines} maxHeight={500} />
        </CardContent>
      </Card>
    </div>
  );
}
