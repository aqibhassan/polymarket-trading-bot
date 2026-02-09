import { Pool } from 'pg';

const pool = new Pool({
  connectionString: process.env.TIMESCALEDB_URL,
  max: 5,
  idleTimeoutMillis: 30000,
});

export default pool;
