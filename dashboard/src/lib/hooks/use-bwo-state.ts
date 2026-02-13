'use client';

import { useState, useEffect, useCallback } from 'react';
import type { BWOSummary, BWOHealthStatus } from '@/lib/types/bwo';

export function useBwoState(refreshInterval = 10000) {
  const [summary, setSummary] = useState<BWOSummary | null>(null);
  const [health, setHealth] = useState<BWOHealthStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchState = useCallback(async () => {
    try {
      const [sumRes, healthRes] = await Promise.all([
        fetch('/api/summary'),
        fetch('/api/health'),
      ]);
      if (!sumRes.ok || !healthRes.ok) {
        setConnected(false);
        setError('Data server returned an error');
        return;
      }
      const sumData = await sumRes.json();
      const healthData = await healthRes.json();
      setSummary(sumData);
      setHealth(healthData);
      setConnected(true);
      setError(null);
    } catch {
      setConnected(false);
      setError('Failed to connect to data server');
    }
  }, []);

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchState, refreshInterval]);

  return { summary, health, connected, error, refresh: fetchState };
}
