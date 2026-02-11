'use client';

import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBannerProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-amber-700/50 bg-amber-900/20 px-4 py-2 text-sm">
      <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
      <span className="text-amber-300 flex-1">{message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300 transition-colors"
        >
          <RefreshCw className="h-3 w-3" />
          Retry
        </button>
      )}
    </div>
  );
}
