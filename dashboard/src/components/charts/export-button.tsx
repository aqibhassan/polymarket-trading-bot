'use client';

import { Download } from 'lucide-react';

interface ExportButtonProps {
  format?: 'csv' | 'json';
}

export function ExportButton({ format = 'csv' }: ExportButtonProps) {
  const handleExport = async () => {
    const res = await fetch(`/api/export/trades?format=${format}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bwo_trades.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={handleExport}
      className="inline-flex items-center gap-2 rounded-md bg-zinc-800 px-3 py-1.5 text-sm font-medium text-zinc-300 hover:bg-zinc-700 transition-colors border border-zinc-700"
    >
      <Download className="h-4 w-4" />
      Export {format.toUpperCase()}
    </button>
  );
}
