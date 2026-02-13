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
import { timeAgo } from '@/lib/format';
import type { BWOTradeRecord } from '@/lib/types/bwo';

interface TradesTableProps {
  trades: BWOTradeRecord[];
  page?: number;
  totalPages?: number;
  onPageChange?: (page: number) => void;
}

export function TradesTable({ trades, page, totalPages, onPageChange }: TradesTableProps) {
  if (!trades.length && (!page || page === 1)) {
    return (
      <div className="flex items-center justify-center h-32 text-zinc-500">
        No trades yet
      </div>
    );
  }

  const hasPagination = totalPages !== undefined && totalPages > 1 && onPageChange;

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Time</TableHead>
            <TableHead>Side</TableHead>
            <TableHead>BTC Dir</TableHead>
            <TableHead className="text-right">Entry $</TableHead>
            <TableHead className="text-right">Settle</TableHead>
            <TableHead className="text-right">Shares</TableHead>
            <TableHead className="text-right">P&L</TableHead>
            <TableHead className="text-right">Confidence</TableHead>
            <TableHead>Result</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {trades.map((trade, i) => {
            const pnl = trade.pnl_net;
            return (
              <TableRow key={`${trade.window_ts}-${i}`}>
                <TableCell className="font-mono text-xs text-zinc-400">
                  {timeAgo(trade.window_ts)}
                </TableCell>
                <TableCell>
                  <Badge variant={trade.side === 'YES' ? 'success' : 'destructive'}>
                    {trade.side}
                  </Badge>
                </TableCell>
                <TableCell>
                  <span className={trade.btc_direction === 'UP' ? 'text-emerald-400' : 'text-red-400'}>
                    {trade.btc_direction}
                  </span>
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${trade.entry_price.toFixed(4)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${trade.settlement.toFixed(2)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {trade.shares.toFixed(2)}
                </TableCell>
                <TableCell
                  className={cn(
                    'text-right font-mono font-semibold',
                    pnl >= 0 ? 'text-emerald-400' : 'text-red-400',
                  )}
                >
                  {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                </TableCell>
                <TableCell className="text-right font-mono text-zinc-400">
                  {(trade.cont_prob * 100).toFixed(1)}%
                </TableCell>
                <TableCell>
                  <Badge variant={trade.correct ? 'success' : 'destructive'}>
                    {trade.correct ? 'WIN' : 'LOSS'}
                  </Badge>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

      {hasPagination && (
        <div className="flex items-center justify-between pt-4 px-1">
          <button
            onClick={() => onPageChange(Math.max(1, (page ?? 1) - 1))}
            disabled={!page || page <= 1}
            className="px-3 py-1.5 text-xs font-medium rounded-md border border-zinc-700 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-xs text-zinc-500">
            Page {page ?? 1} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(Math.min(totalPages ?? 1, (page ?? 1) + 1))}
            disabled={page !== undefined && page >= (totalPages ?? 1)}
            className="px-3 py-1.5 text-xs font-medium rounded-md border border-zinc-700 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
