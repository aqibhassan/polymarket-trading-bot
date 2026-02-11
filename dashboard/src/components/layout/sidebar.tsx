'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  LayoutDashboard,
  TrendingUp,
  Shield,
  Activity,
  LogOut,
  X,
} from 'lucide-react';

const navItems = [
  { href: '/overview', label: 'Overview', icon: LayoutDashboard },
  { href: '/performance', label: 'Performance', icon: TrendingUp },
  { href: '/risk', label: 'Risk Monitor', icon: Shield },
  { href: '/health', label: 'System Health', icon: Activity },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export function Sidebar({ open, onClose }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();

  async function handleLogout() {
    await fetch('/api/auth/logout', { method: 'POST' });
    router.push('/login');
    router.refresh();
  }

  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed left-0 top-0 h-screen w-60 bg-zinc-900 border-r border-zinc-800 flex flex-col z-50 transition-transform duration-200 ${
          open ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}
      >
        <div className="p-5 border-b border-zinc-800 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-emerald-500 font-bold text-xl tracking-tight">
                MVHE
              </h1>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 ring-1 ring-red-500/50">
                LIVE
              </span>
            </div>
            <p className="text-zinc-500 text-xs mt-1">Dashboard</p>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-zinc-400 hover:text-zinc-200 md:hidden"
            aria-label="Close menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(({ href, label, icon: Icon }) => {
            const isActive = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                onClick={onClose}
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

        <div className="p-4 border-t border-zinc-800 space-y-3">
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-md text-sm font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors"
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </button>
          <p className="text-zinc-600 text-xs">v1.0.0</p>
        </div>
      </aside>
    </>
  );
}
