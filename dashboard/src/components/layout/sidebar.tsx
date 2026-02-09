'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  TrendingUp,
  Shield,
  Activity,
} from 'lucide-react';

const navItems = [
  { href: '/overview', label: 'Live Overview', icon: LayoutDashboard },
  { href: '/performance', label: 'Performance', icon: TrendingUp },
  { href: '/risk', label: 'Risk Monitor', icon: Shield },
  { href: '/health', label: 'System Health', icon: Activity },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 bg-zinc-900 border-r border-zinc-800 flex flex-col">
      <div className="p-5 border-b border-zinc-800">
        <h1 className="text-emerald-500 font-bold text-xl tracking-tight">
          MVHE
        </h1>
        <p className="text-zinc-500 text-xs mt-1">Dashboard</p>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-zinc-800 text-white'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-zinc-800">
        <p className="text-zinc-600 text-xs">v0.1.0 — Pre-alpha</p>
      </div>
    </aside>
  );
}
