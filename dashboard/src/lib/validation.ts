/**
 * API input validation helpers.
 *
 * All query parameters arrive as strings from the URL.  These helpers
 * sanitise, bounds-check, and whitelist before values reach the DB layer.
 */

const ALLOWED_STRATEGIES = new Set([
  'singularity',
  'momentum_confirmation',
  'false_sentiment',
]);

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const ISO_DATETIME_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;
const SYMBOL_RE = /^[A-Z0-9]{3,20}$/;

/** Validate and return a strategy name, or the fallback if invalid. */
export function validStrategy(raw: string | null, fallback: string = 'singularity'): string {
  if (raw && ALLOWED_STRATEGIES.has(raw)) return raw;
  return fallback;
}

/** Validate strategy, returning undefined when absent (for optional params). */
export function validStrategyOrUndefined(raw: string | null): string | undefined {
  if (!raw) return undefined;
  return ALLOWED_STRATEGIES.has(raw) ? raw : undefined;
}

/** Parse an integer limit with bounds clamping and NaN protection. */
export function validLimit(raw: string | null, defaultVal: number, max: number): number {
  const parsed = parseInt(raw || String(defaultVal), 10);
  if (Number.isNaN(parsed) || parsed < 1) return defaultVal;
  return Math.min(parsed, max);
}

/** Validate an ISO date string (YYYY-MM-DD). */
export function validIsoDate(raw: string | null): string | null {
  if (!raw) return null;
  return ISO_DATE_RE.test(raw) && !Number.isNaN(Date.parse(raw)) ? raw : null;
}

/** Validate an ISO datetime string (YYYY-MM-DDTHH:MM:SS...). */
export function validIsoDatetime(raw: string | null): string | null {
  if (!raw) return null;
  return ISO_DATETIME_RE.test(raw) && !Number.isNaN(Date.parse(raw)) ? raw : null;
}

/** Validate a trading symbol (uppercase alphanumeric, 3-20 chars). */
export function validSymbol(raw: string | null, fallback: string = 'BTCUSDT'): string {
  if (raw && SYMBOL_RE.test(raw)) return raw;
  return fallback;
}
