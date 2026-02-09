'use client';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { Trade } from '@/lib/types/trade';

interface TradesTableProps {
  trades: Trade[];
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  const now = Date.now();
  const diff = Math.floor((now - d.getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function TradesTable({ trades }: TradesTableProps) {
  if (!trades.length) {
    return (
      <div className="flex items-center justify-center h-32 text-zinc-500">
        No trades yet
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Time</TableHead>
          <TableHead>Direction</TableHead>
          <TableHead className="text-right">Entry</TableHead>
          <TableHead className="text-right">Exit</TableHead>
          <TableHead className="text-right">Size</TableHead>
          <TableHead className="text-right">P&L</TableHead>
          <TableHead>Exit Reason</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {trades.map((trade) => (
          <TableRow key={trade.trade_id}>
            <TableCell className="font-mono text-xs text-zinc-400">
              {formatTime(trade.entry_time)}
            </TableCell>
            <TableCell>
              <Badge variant={trade.direction === 'YES' ? 'success' : 'destructive'}>
                {trade.direction}
              </Badge>
            </TableCell>
            <TableCell className="text-right font-mono">
              ${Number(trade.entry_price).toFixed(4)}
            </TableCell>
            <TableCell className="text-right font-mono">
              ${Number(trade.exit_price).toFixed(4)}
            </TableCell>
            <TableCell className="text-right font-mono">
              ${Number(trade.position_size).toFixed(2)}
            </TableCell>
            <TableCell
              className={cn(
                'text-right font-mono font-semibold',
                Number(trade.pnl) >= 0 ? 'text-emerald-400' : 'text-red-400',
              )}
            >
              {Number(trade.pnl) >= 0 ? '+' : ''}${Number(trade.pnl).toFixed(2)}
            </TableCell>
            <TableCell>
              <Badge variant="secondary">{trade.exit_reason}</Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
