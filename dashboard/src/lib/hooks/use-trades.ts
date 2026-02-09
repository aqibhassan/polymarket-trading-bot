'use client';

import { useState, useEffect } from 'react';
import type { Trade } from '@/lib/types/trade';

export function useTrades(limit: number = 10, refreshInterval: number = 30000) {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res = await fetch(`/api/trades?limit=${limit}`);
        const data = await res.json();
        setTrades(data.trades || []);
      } catch { /* fetch failed */ } finally {
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, refreshInterval);
    return () => clearInterval(interval);
  }, [limit, refreshInterval]);

  return { trades, loading };
}
