'use client';

import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  icon?: React.ReactNode;
}

const trendIcons = {
  up: <TrendingUp className="h-4 w-4 text-emerald-400" />,
  down: <TrendingDown className="h-4 w-4 text-red-400" />,
  neutral: <Minus className="h-4 w-4 text-zinc-400" />,
};

const trendColors = {
  up: 'text-emerald-400',
  down: 'text-red-400',
  neutral: 'text-zinc-400',
};

export function StatCard({ title, value, subtitle, trend, trendValue, icon }: StatCardProps) {
  return (
    <Card>
      <CardContent className="p-4 md:p-6">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-zinc-400">{title}</p>
          {icon && <div className="text-zinc-500">{icon}</div>}
        </div>
        <div className="mt-2">
          <p className="text-xl md:text-2xl font-bold text-zinc-50">{value}</p>
        </div>
        {(trend || subtitle) && (
          <div className="mt-1 flex items-center gap-1">
            {trend && trendIcons[trend]}
            {trendValue && (
              <span className={cn('text-sm', trend ? trendColors[trend] : 'text-zinc-400')}>
                {trendValue}
              </span>
            )}
            {subtitle && <span className="text-sm text-zinc-500">{subtitle}</span>}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
