#!/usr/bin/env bash
# MVHE daily backup script — run via cron at 03:00 UTC:
#   0 3 * * * /opt/mvhe-live/scripts/backup.sh >> /var/log/mvhe-backup.log 2>&1
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/opt/mvhe-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
DATE=$(date -u +"%Y%m%d_%H%M%S")

mkdir -p "$BACKUP_DIR"

echo "[$(date -u)] Starting MVHE backup..."

# TimescaleDB
echo "[$(date -u)] Backing up TimescaleDB..."
docker compose -f docker-compose.prod.yml exec -T timescaledb \
    pg_dump -U "${POSTGRES_USER:-mvhe}" "${POSTGRES_DB:-mvhe}" \
    | gzip > "$BACKUP_DIR/timescaledb_${DATE}.sql.gz"

# ClickHouse — export key tables
echo "[$(date -u)] Backing up ClickHouse..."
for table in trades audit_events signal_activity; do
    docker compose -f docker-compose.prod.yml exec -T clickhouse \
        clickhouse-client --query "SELECT * FROM ${table} FORMAT CSVWithNames" \
        | gzip > "$BACKUP_DIR/clickhouse_${table}_${DATE}.csv.gz" 2>/dev/null || true
done

# Redis
echo "[$(date -u)] Backing up Redis..."
docker compose -f docker-compose.prod.yml exec -T redis redis-cli BGSAVE
sleep 2
docker compose -f docker-compose.prod.yml cp redis:/data/dump.rdb "$BACKUP_DIR/redis_${DATE}.rdb"

# Audit JSONL (if volume-mounted)
if [ -d "/opt/mvhe-live/audit" ]; then
    echo "[$(date -u)] Backing up audit logs..."
    tar czf "$BACKUP_DIR/audit_${DATE}.tar.gz" -C /opt/mvhe-live audit/
fi

# Cleanup old backups
echo "[$(date -u)] Cleaning up backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -type f -mtime "+${RETENTION_DAYS}" -delete

echo "[$(date -u)] Backup complete. Files in $BACKUP_DIR:"
ls -lh "$BACKUP_DIR"/*"${DATE}"* 2>/dev/null || echo "  (none found for today)"
