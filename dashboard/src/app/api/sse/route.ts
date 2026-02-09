import { NextResponse } from 'next/server';
import { getAllBotState } from '@/lib/queries/bot-state';

export async function GET() {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: unknown) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
        } catch {
          // Controller may be closed
        }
      };

      // Send initial state
      try {
        const state = await getAllBotState();
        send({ type: 'bot-state', data: state });
      } catch { /* initial state unavailable */ }

      // Poll every 2 seconds
      const interval = setInterval(async () => {
        try {
          const state = await getAllBotState();
          send({ type: 'bot-state', data: state });
        } catch {
          send({ type: 'error', data: { message: 'Failed to read state' } });
        }
      }, 2000);

      // Auto-close after 5 minutes to prevent stale connections
      setTimeout(() => {
        clearInterval(interval);
        try {
          controller.close();
        } catch { /* already closed */ }
      }, 5 * 60 * 1000);
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
