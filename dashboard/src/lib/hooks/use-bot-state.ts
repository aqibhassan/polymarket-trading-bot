'use client';

import { useState, useEffect } from 'react';
import type { BotState } from '@/lib/types/bot-state';

const INITIAL_STATE: BotState = {
  heartbeat: null,
  balance: null,
  position: null,
  window: null,
  daily: null,
  ws_status: null,
  last_trade: null,
  signals: null,
  sizing: null,
  signal_activity: null,
};

export function useBotState() {
  const [state, setState] = useState<BotState>(INITIAL_STATE);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let es: EventSource | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let disposed = false;
    let retryDelay = 1000; // Start at 1s, exponential backoff up to 30s

    const connect = () => {
      if (disposed) return;
      // Clear any pending reconnect to prevent connection multiplication
      clearTimeout(reconnectTimeout);

      es = new EventSource('/api/sse');

      es.onopen = () => {
        if (!disposed) {
          setConnected(true);
          retryDelay = 1000; // Reset backoff on successful connect
        }
      };

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === 'bot-state') {
            setState(prev => ({ ...prev, ...parsed.data }));
          }
        } catch { /* ignore parse errors */ }
      };

      es.onerror = () => {
        if (disposed) return;
        setConnected(false);
        es?.close();
        es = null;
        // Exponential backoff: 1s → 2s → 4s → 8s → 16s → 30s max
        reconnectTimeout = setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 30000);
      };
    };

    connect();

    return () => {
      disposed = true;
      es?.close();
      es = null;
      clearTimeout(reconnectTimeout);
    };
  }, []);

  return { state, connected };
}
