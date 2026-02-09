import { NextRequest, NextResponse } from 'next/server';
import { getAllBotState } from '@/lib/queries/bot-state';

export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();
  const { signal } = request;

  const stream = new ReadableStream({
    async start(controller) {
      let closed = false;

      const cleanup = () => {
        if (closed) return;
        closed = true;
        clearInterval(interval);
        clearTimeout(autoClose);
        try { controller.close(); } catch { /* already closed */ }
      };

      // Clean up on client disconnect
      signal.addEventListener('abort', cleanup);

      const send = (data: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
        } catch {
          cleanup();
        }
      };

      // Send initial state
      try {
        const state = await getAllBotState();
        send({ type: 'bot-state', data: state });
      } catch { /* initial state unavailable */ }

      // Poll every 2 seconds
      const interval = setInterval(async () => {
        if (closed) return;
        try {
          const state = await getAllBotState();
          send({ type: 'bot-state', data: state });
        } catch {
          send({ type: 'error', data: { message: 'Failed to read state' } });
        }
      }, 2000);

      // Auto-close after 5 minutes to prevent stale connections
      const autoClose = setTimeout(cleanup, 5 * 60 * 1000);
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}

export const dynamic = 'force-dynamic';
