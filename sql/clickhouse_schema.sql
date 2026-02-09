-- ClickHouse schema for MVHE analytics and audit trail.
-- Applied automatically by ClickHouseStore.connect() or manually via:
--   clickhouse-client --multiquery < sql/clickhouse_schema.sql

CREATE DATABASE IF NOT EXISTS mvhe;

CREATE TABLE IF NOT EXISTS mvhe.trades (
    trade_id String,
    market_id String,
    strategy String,
    direction String,  -- YES or NO
    entry_price Decimal64(6),
    exit_price Decimal64(6),
    position_size Decimal64(6),
    pnl Decimal64(6),
    fee_cost Decimal64(6),
    entry_time DateTime64(3),
    exit_time DateTime64(3),
    exit_reason String,
    window_minute UInt8,
    cum_return_pct Float64,
    confidence Float64
) ENGINE = MergeTree()
ORDER BY (strategy, entry_time)
PARTITION BY toYYYYMM(entry_time);

CREATE TABLE IF NOT EXISTS mvhe.audit_events (
    event_id String,
    order_id String,
    event_type String,
    market_id String,
    strategy String,
    details String,  -- JSON blob
    timestamp DateTime64(3)
) ENGINE = MergeTree()
ORDER BY (order_id, timestamp)
PARTITION BY toYYYYMM(timestamp);

CREATE TABLE IF NOT EXISTS mvhe.daily_summary (
    date Date,
    strategy String,
    trade_count UInt32,
    win_count UInt32,
    total_pnl Decimal64(6),
    total_fees Decimal64(6),
    max_drawdown Float64,
    avg_position_size Decimal64(6)
) ENGINE = SummingMergeTree()
ORDER BY (strategy, date);
