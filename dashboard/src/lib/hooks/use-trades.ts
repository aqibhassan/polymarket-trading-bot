'use client';

import { useState, useEffect, useCallback } from 'react';
import type { Trade } from '@/lib/types/trade';

export function useTrades(limit: number = 10, refreshInterval: number = 10000) {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);

  const fetchTrades = useCallback(async () => {
    try {
      const offset = (page - 1) * limit;
      const res = await fetch(`/api/trades?limit=${limit}&offset=${offset}`);
      const data = await res.json();
      setTrades(data.trades || []);
      setTotal(data.total ?? 0);
    } catch { /* fetch failed */ } finally {
      setLoading(false);
    }
  }, [limit, page]);

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchTrades, refreshInterval]);

  const totalPages = Math.max(1, Math.ceil(total / limit));

  return { trades, loading, page, totalPages, total, setPage };
}
