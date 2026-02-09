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
};

export function useBotState() {
  const [state, setState] = useState<BotState>(INITIAL_STATE);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let es: EventSource | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      // Clear any pending reconnect to prevent connection multiplication
      clearTimeout(reconnectTimeout);

      es = new EventSource('/api/sse');

      es.onopen = () => {
        if (!disposed) setConnected(true);
      };

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === 'bot-state') {
            setState(parsed.data);
          }
        } catch { /* ignore parse errors */ }
      };

      es.onerror = () => {
        if (disposed) return;
        setConnected(false);
        es?.close();
        es = null;
        // Single reconnect — cleared at top of connect() if called again
        reconnectTimeout = setTimeout(connect, 3000);
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
