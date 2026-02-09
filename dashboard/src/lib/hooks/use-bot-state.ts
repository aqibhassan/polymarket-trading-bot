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

    const connect = () => {
      es = new EventSource('/api/sse');

      es.onopen = () => setConnected(true);

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === 'bot-state') {
            setState(parsed.data);
          }
        } catch { /* ignore parse errors */ }
      };

      es.onerror = () => {
        setConnected(false);
        es?.close();
        reconnectTimeout = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      es?.close();
      clearTimeout(reconnectTimeout);
    };
  }, []);

  return { state, connected };
}
