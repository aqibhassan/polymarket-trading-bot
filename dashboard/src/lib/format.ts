/** Shared formatting utilities — single source of truth for the dashboard */

/** Safe Number conversion — returns 0 for null/undefined/NaN */
export function safeNum(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

/** Relative time string: "5s ago", "3m ago", "2h ago" */
export function timeAgo(ts: string): string {
  if (!ts) return '--';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '--';
  const diff = Math.floor((Date.now() - d.getTime()) / 1000);
  if (diff < 0) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** Absolute UTC time: "14:23:05 UTC" */
export function formatUTCTime(ts: string): string {
  if (!ts) return '--';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '--';
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'UTC' }) + ' UTC';
}

/** UTC date + time: "Feb 11, 14:23 UTC" or just time if today */
export function formatUTCDateTime(ts: string): string {
  if (!ts) return '--';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '--';
  const now = new Date();
  const isToday = d.toISOString().split('T')[0] === now.toISOString().split('T')[0];
  if (isToday) return formatUTCTime(ts);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' }) +
    ', ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC';
}

/** Token price: "$0.4800" (4 decimals) */
export function formatPrice(v: unknown, decimals = 4): string {
  const n = safeNum(v);
  return `$${n.toFixed(decimals)}`;
}

/** USDC amount: "$109.42" (2 decimals) */
export function formatUSD(v: unknown): string {
  return `$${safeNum(v).toFixed(2)}`;
}

/** P&L with sign and color class: { text: "+$3.08", className: "text-emerald-400" } */
export function formatPnl(v: unknown): { text: string; className: string } {
  const n = safeNum(v);
  return {
    text: `${n >= 0 ? '+' : ''}$${n.toFixed(2)}`,
    className: n >= 0 ? 'text-emerald-400' : 'text-red-400',
  };
}

/** Percentage: "83.3%" (1 decimal) */
export function formatPct(v: unknown, decimals = 1): string {
  return `${safeNum(v).toFixed(decimals)}%`;
}
