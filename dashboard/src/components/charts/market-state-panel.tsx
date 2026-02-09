'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { WindowProgress } from '@/components/charts/window-progress';
import type { BotWindow } from '@/lib/types/bot-state';

interface MarketStatePanelProps {
  window: BotWindow | null;
}

export function MarketStatePanel({ window }: MarketStatePanelProps) {
  const [countdown, setCountdown] = useState('--:--');

  useEffect(() => {
    if (!window) return;

    const tick = () => {
      const elapsed = Math.floor(Date.now() / 1000) - window.start_ts;
      const remaining = Math.max(0, 900 - elapsed);
      const mm = Math.floor(remaining / 60).toString().padStart(2, '0');
      const ss = (remaining % 60).toString().padStart(2, '0');
      setCountdown(`${mm}:${ss}`);
    };

    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [window?.start_ts, window]);

  if (!window) {
    return (
      <Card>
        <CardContent className="p-6 flex items-center justify-center h-32">
          <p className="text-sm text-zinc-500">Waiting for market window...</p>
        </CardContent>
      </Card>
    );
  }

  const btcPrice = Number(window.btc_close);
  const yesPrice = Number(window.yes_price);
  const cumReturn = window.cum_return_pct;

  return (
    <Card>
      <CardContent className="p-6">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-4">
          <div>
            <p className="text-xs text-zinc-500">BTC Price</p>
            <p className="text-xl font-mono font-bold text-zinc-50">
              ${btcPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          <div>
            <p className="text-xs text-zinc-500">YES Token</p>
            <p className="text-xl font-mono font-bold text-zinc-50">
              {yesPrice.toFixed(4)}
            </p>
          </div>
          <div>
            <p className="text-xs text-zinc-500">Window Countdown</p>
            <p className="text-xl font-mono font-bold text-amber-400">
              {countdown}
            </p>
          </div>
          <div>
            <p className="text-xs text-zinc-500">Cumulative Return</p>
            <p className={`text-xl font-mono font-bold ${cumReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {cumReturn >= 0 ? '+' : ''}{cumReturn.toFixed(4)}%
            </p>
          </div>
        </div>
        <WindowProgress minute={window.minute} />
      </CardContent>
    </Card>
  );
}
