import { createClient } from '@clickhouse/client-web';

const clickhouse = createClient({
  url: `http://${process.env.CLICKHOUSE_HOST || 'localhost'}:${process.env.CLICKHOUSE_PORT || '8123'}`,
  database: 'mvhe',
});

export default clickhouse;
