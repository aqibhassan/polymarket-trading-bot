const BWO_API_URL = process.env.BWO_API_URL || 'http://localhost:8100';

export async function bwoFetch<T>(path: string): Promise<T> {
  const url = `${BWO_API_URL}${path}`;
  const res = await fetch(url, { next: { revalidate: 0 } });
  if (!res.ok) {
    throw new Error(`BWO API error: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}
