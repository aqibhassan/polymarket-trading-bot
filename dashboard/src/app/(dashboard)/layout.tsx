'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/layout/sidebar';
import { Menu } from 'lucide-react';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Mobile top bar */}
      <div className="fixed top-0 left-0 right-0 z-30 flex items-center h-14 px-4 bg-zinc-900 border-b border-zinc-800 md:hidden">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-2 -ml-2 text-zinc-400 hover:text-zinc-200"
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5" />
        </button>
        <span className="ml-3 text-emerald-500 font-bold text-lg tracking-tight">MVHE</span>
      </div>

      <main className="flex-1 md:ml-60 p-4 md:p-6 pt-18 md:pt-6">{children}</main>
    </div>
  );
}
