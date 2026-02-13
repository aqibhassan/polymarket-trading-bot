#!/usr/bin/env bash
# Deploy BWO Paper Traders (15m + 5m) to droplet 146.190.68.214
set -euo pipefail

DROPLET="root@146.190.68.214"
REMOTE_DIR="/opt/bwo-paper"

echo "=== BWO Paper Trader Deployment (15m + 5m) ==="

# Step 1: Create directory structure
echo "[1/5] Creating directories..."
ssh "$DROPLET" "mkdir -p $REMOTE_DIR/scripts $REMOTE_DIR/logs $REMOTE_DIR/models $REMOTE_DIR/data"

# Step 2: Copy scripts (maintain scripts/ directory for imports)
echo "[2/5] Copying scripts..."
scp scripts/bwo_paper_trader.py "$DROPLET:$REMOTE_DIR/scripts/"
scp scripts/bwo_5m_paper_trader.py "$DROPLET:$REMOTE_DIR/scripts/"
scp scripts/bwo_5m_backtest.py "$DROPLET:$REMOTE_DIR/scripts/"
scp scripts/bwo_continuation_backtest.py "$DROPLET:$REMOTE_DIR/scripts/"
scp scripts/backtest_before_window.py "$DROPLET:$REMOTE_DIR/scripts/"
scp scripts/fast_loader.py "$DROPLET:$REMOTE_DIR/scripts/"
# Ensure scripts/ is a Python package
ssh "$DROPLET" "touch $REMOTE_DIR/scripts/__init__.py"

# Step 3: Copy data files
echo "[3/5] Copying data files..."
scp data/btc_1m_2y.csv "$DROPLET:$REMOTE_DIR/data/"
scp data/btc_futures_1m_2y.csv "$DROPLET:$REMOTE_DIR/data/"
scp data/eth_futures_1m_2y.csv "$DROPLET:$REMOTE_DIR/data/"
if [ -f data/deribit_dvol_1m.csv ]; then
    scp data/deribit_dvol_1m.csv "$DROPLET:$REMOTE_DIR/data/"
fi
if [ -f data/deribit_btc_perp_1m.csv ]; then
    scp data/deribit_btc_perp_1m.csv "$DROPLET:$REMOTE_DIR/data/"
fi

# Step 4: Install dependencies
echo "[4/5] Installing dependencies..."
ssh "$DROPLET" "apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1 || true"
ssh "$DROPLET" "cd $REMOTE_DIR && python3 -m venv venv 2>/dev/null || true"
ssh "$DROPLET" "cd $REMOTE_DIR && source venv/bin/activate && pip install -q httpx numpy scikit-learn joblib"

# Step 5: Create and start systemd services
echo "[5/5] Setting up systemd services..."

# --- 15m Paper Trader ---
ssh "$DROPLET" "cat > /etc/systemd/system/bwo-paper-15m.service << 'UNIT'
[Unit]
Description=BWO Paper Trader - 15m (Hybrid B + BuyBoth D)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bwo-paper
Environment=PYTHONPATH=/opt/bwo-paper
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bwo-paper/venv/bin/python3 -u scripts/bwo_paper_trader.py
Restart=always
RestartSec=30
StandardOutput=append:/opt/bwo-paper/logs/15m_console.log
StandardError=append:/opt/bwo-paper/logs/15m_console.log

[Install]
WantedBy=multi-user.target
UNIT"

# --- 5m Paper Trader ---
ssh "$DROPLET" "cat > /etc/systemd/system/bwo-paper-5m.service << 'UNIT'
[Unit]
Description=BWO Paper Trader - 5m (Continuation at Min 2)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bwo-paper
Environment=PYTHONPATH=/opt/bwo-paper
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/bwo-paper/venv/bin/python3 -u scripts/bwo_5m_paper_trader.py
Restart=always
RestartSec=30
StandardOutput=append:/opt/bwo-paper/logs/5m_console.log
StandardError=append:/opt/bwo-paper/logs/5m_console.log

[Install]
WantedBy=multi-user.target
UNIT"

# Stop old single service if it exists
ssh "$DROPLET" "systemctl stop bwo-paper 2>/dev/null || true; systemctl disable bwo-paper 2>/dev/null || true"

# Enable and start both
ssh "$DROPLET" "systemctl daemon-reload && \
  systemctl enable bwo-paper-15m bwo-paper-5m && \
  systemctl restart bwo-paper-15m && \
  systemctl restart bwo-paper-5m"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "15m Monitor: ssh $DROPLET \"tail -f $REMOTE_DIR/logs/15m_console.log\""
echo "5m  Monitor: ssh $DROPLET \"tail -f $REMOTE_DIR/logs/5m_console.log\""
echo ""
echo "Status: ssh $DROPLET \"systemctl status bwo-paper-15m bwo-paper-5m\""
echo ""
echo "15m Trades: ssh $DROPLET \"tail -5 $REMOTE_DIR/logs/bwo_paper_trades.jsonl\""
echo "5m  Trades: ssh $DROPLET \"tail -5 $REMOTE_DIR/logs/bwo_5m_paper_trades.jsonl\""
echo ""
echo "Summaries:"
echo "  ssh $DROPLET \"cat $REMOTE_DIR/logs/bwo_paper_summary.json\""
echo "  ssh $DROPLET \"cat $REMOTE_DIR/logs/bwo_5m_paper_summary.json\""
