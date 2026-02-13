'use client';

import { useState, useEffect } from 'react';
import { useBotState } from '@/lib/hooks/use-bot-state';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { ErrorBanner } from '@/components/ui/error-banner';
import { formatUTCTime, formatEventType, formatOrderId } from '@/lib/format';
import type { HealthStatus } from '@/lib/types/bot-state';
import type { AuditEvent } from '@/lib/types/trade';

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

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

export default function HealthPage() {
  const { state } = useBotState();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [healthRes, auditRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/audit-events?limit=30'),
      ]);
      const healthData = await healthRes.json();
      const auditData = await auditRes.json();
      setHealth(healthData);
      setAuditEvents(auditData.events || []);
      setError(null);
    } catch {
      setError('Failed to load health data');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold text-zinc-50">System Health</h1>
      {error && <ErrorBanner message={error} onRetry={fetchData} />}

      {/* Connection Status Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <ConnectionIndicator
          label="Binance WebSocket"
          connected={state.ws_status?.binance ?? false}
        />
        <ConnectionIndicator
          label="Polymarket"
          connected={state.ws_status?.polymarket ?? false}
        />
        <ConnectionIndicator
          label="TimescaleDB"
          connected={health?.timescaledb ?? false}
        />
        <ConnectionIndicator
          label="Redis"
          connected={health?.redis ?? false}
        />
        <ConnectionIndicator
          label="ClickHouse"
          connected={health?.clickhouse ?? false}
        />
      </div>

      {/* Bot Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Bot Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-6">
            <div className={`h-12 w-12 rounded-full flex items-center justify-center ${health?.bot_alive ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
              <div className={`h-5 w-5 rounded-full ${health?.bot_alive ? 'bg-emerald-500 animate-pulse' : 'bg-zinc-600'}`} />
            </div>
            <div className="space-y-1">
              <p className="text-lg font-semibold text-zinc-50">
                {health?.bot_alive ? 'Bot Online' : 'Bot Offline'}
              </p>
              <p className="text-sm text-zinc-400">
                {health != null && health.heartbeat_age_s != null
                  ? `Last heartbeat ${health.heartbeat_age_s}s ago`
                  : 'No heartbeat received'}
              </p>
              {health?.bot_alive && (
                <p className="text-xs text-zinc-600">Heartbeat checks every 30s</p>
              )}
            </div>
            <div className="ml-auto flex gap-2">
              {health?.mode && (
                <Badge variant="secondary">{health.mode}</Badge>
              )}
              {health?.strategy && (
                <Badge variant="outline">{health.strategy}</Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Uptime & Idle Time */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="pt-6">
            <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Uptime</p>
            <p className="mt-2 text-2xl font-bold text-zinc-50">
              {health?.uptime_s != null ? formatDuration(health.uptime_s) : '--'}
            </p>
            <p className="text-xs text-zinc-500 mt-1">Since bot started</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Idle Time</p>
            <p className={`mt-2 text-2xl font-bold ${
              health?.has_open_position
                ? 'text-emerald-400'
                : state.position?.status === 'gtc_pending'
                  ? 'text-amber-400'
                  : health?.idle_since_s != null && health.idle_since_s > 1800
                    ? 'text-amber-400'
                    : 'text-zinc-50'
            }`}>
              {health?.has_open_position
                ? 'In Trade'
                : state.position?.status === 'gtc_pending'
                  ? 'GTC Pending'
                  : health?.idle_since_s != null
                    ? formatDuration(health.idle_since_s)
                    : '--'}
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              {state.position?.status === 'gtc_pending'
                ? 'Limit order waiting to fill'
                : health?.last_trade_time
                  ? `Last trade: ${formatUTCTime(health.last_trade_time)}`
                  : 'No trades yet'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Position</p>
            <p className={`mt-2 text-2xl font-bold ${
              health?.has_open_position
                ? 'text-emerald-400'
                : state.position?.status === 'gtc_pending'
                  ? 'text-amber-400'
                  : 'text-zinc-500'
            }`}>
              {health?.has_open_position
                ? 'Open'
                : state.position?.status === 'gtc_pending'
                  ? 'GTC Pending'
                  : 'None'}
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              {state.position
                ? `${state.position.side} @ $${Number(state.position.entry_price).toFixed(4)}`
                : 'Waiting for signal'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Audit Events */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Audit Events</CardTitle>
        </CardHeader>
        <CardContent>
          {auditEvents.length > 0 ? (
            <div className="max-h-[400px] overflow-y-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Time</TableHead>
                    <TableHead>Event Type</TableHead>
                    <TableHead>Order ID</TableHead>
                    <TableHead>Market</TableHead>
                    <TableHead>Strategy</TableHead>
                    <TableHead>Details</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {auditEvents.map((event) => (
                    <TableRow key={event.event_id}>
                      <TableCell className="font-mono text-xs text-zinc-400">
                        {formatUTCTime(event.timestamp)}
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">{formatEventType(event.event_type)}</Badge>
                      </TableCell>
                      <TableCell className="font-mono text-xs text-zinc-400" title={formatOrderId(event.order_id || '')}>
                        {formatOrderId(event.order_id || '').slice(0, 8)}...
                      </TableCell>
                      <TableCell className="text-xs text-zinc-400">
                        {(event.market_id || '').slice(0, 16)}
                      </TableCell>
                      <TableCell className="text-xs text-zinc-400">
                        {event.strategy}
                      </TableCell>
                      <TableCell className="text-xs text-zinc-500 max-w-[200px] truncate">
                        {event.details}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-32 text-zinc-500">
              No audit events recorded
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
