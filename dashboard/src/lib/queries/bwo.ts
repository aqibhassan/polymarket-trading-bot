import { bwoFetch } from '@/lib/api-client';
import type {
  BWOSummary,
  BWOTradeRecord,
  BWOEquityPoint,
  BWODailyPnl,
  BWOSkipMetrics,
  BWOAdvancedMetrics,
  BWOConfidenceBucket,
  BWOEntryPriceBucket,
  BWOHealthStatus,
  BWORollingWinRate,
} from '@/lib/types/bwo';

export async function getSummary(): Promise<BWOSummary> {
  return bwoFetch<BWOSummary>('/api/summary');
}

export async function getTrades(
  limit = 50,
  offset = 0
): Promise<{ trades: BWOTradeRecord[]; total: number }> {
  return bwoFetch(`/api/trades?limit=${limit}&offset=${offset}`);
}

export async function getEquityCurve(): Promise<BWOEquityPoint[]> {
  return bwoFetch<BWOEquityPoint[]>('/api/equity');
}

export async function getDailyPnl(): Promise<BWODailyPnl[]> {
  return bwoFetch<BWODailyPnl[]>('/api/daily-pnl');
}

export async function getSkipMetrics(): Promise<BWOSkipMetrics> {
  return bwoFetch<BWOSkipMetrics>('/api/skip-metrics');
}

export async function getAdvancedMetrics(): Promise<BWOAdvancedMetrics> {
  return bwoFetch<BWOAdvancedMetrics>('/api/advanced-metrics');
}

export async function getConfidenceDistribution(): Promise<BWOConfidenceBucket[]> {
  return bwoFetch<BWOConfidenceBucket[]>('/api/confidence-distribution');
}

export async function getEntryPriceAnalysis(): Promise<BWOEntryPriceBucket[]> {
  return bwoFetch<BWOEntryPriceBucket[]>('/api/entry-price-analysis');
}

export async function getHealth(): Promise<BWOHealthStatus> {
  return bwoFetch<BWOHealthStatus>('/api/health');
}

export async function getRollingWinRate(window = 10): Promise<BWORollingWinRate[]> {
  return bwoFetch<BWORollingWinRate[]>(`/api/rolling-win-rate?window=${window}`);
}

export async function getConsoleLog(lines = 100): Promise<string[]> {
  return bwoFetch<string[]>(`/api/console-log?lines=${lines}`);
}
