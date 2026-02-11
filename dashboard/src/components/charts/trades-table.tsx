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
import { safeNum, timeAgo } from '@/lib/format';
import type { Trade } from '@/lib/types/trade';

interface TradesTableProps {
  trades: Trade[];
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
            <TableHead>Direction</TableHead>
            <TableHead className="text-right">Entry</TableHead>
            <TableHead className="text-right">CLOB</TableHead>
            <TableHead className="text-right">Exit</TableHead>
            <TableHead className="text-right">Size</TableHead>
            <TableHead className="text-right">P&L</TableHead>
            <TableHead className="text-right">R:R</TableHead>
            <TableHead>Exit Reason</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {trades.map((trade) => {
            const pnl = safeNum(trade.pnl);
            return (
              <TableRow key={trade.trade_id}>
                <TableCell className="font-mono text-xs text-zinc-400">
                  {timeAgo(trade.entry_time)}
                </TableCell>
                <TableCell>
                  <Badge variant={trade.direction === 'YES' ? 'success' : 'destructive'}>
                    {trade.direction}
                  </Badge>
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${safeNum(trade.entry_price).toFixed(4)}
                </TableCell>
                <TableCell className="text-right font-mono text-zinc-400" title={safeNum(trade.clob_entry_price) > 0 ? '' : 'No CLOB data available at trade time'}>
                  {safeNum(trade.clob_entry_price) > 0
                    ? `$${safeNum(trade.clob_entry_price).toFixed(4)}`
                    : 'N/A'}
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${safeNum(trade.exit_price).toFixed(4)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${safeNum(trade.position_size).toFixed(2)}
                </TableCell>
                <TableCell
                  className={cn(
                    'text-right font-mono font-semibold',
                    pnl >= 0 ? 'text-emerald-400' : 'text-red-400',
                  )}
                >
                  {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                </TableCell>
                <TableCell className="text-right font-mono text-zinc-400" title={safeNum(trade.bet_to_win_ratio) > 0 ? '' : 'No CLOB data'}>
                  {safeNum(trade.bet_to_win_ratio) > 0
                    ? `1:${(1 / safeNum(trade.bet_to_win_ratio)).toFixed(1)}`
                    : '—'}
                </TableCell>
                <TableCell>
                  <Badge variant="secondary">{trade.exit_reason || '--'}</Badge>
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
