#!/usr/bin/env bash
# ============================================================================
# MVHE — Local Memory Stack Setup
# ============================================================================
# ALL components are:
#   ✓ 100% Free and Open Source (MIT License)
#   ✓ 100% Local — data stored as files on YOUR machine
#   ✓ Zero cloud, zero API keys, zero subscriptions, zero telemetry
#
# Installs two MCP servers for Claude Code:
#
#   1. @modelcontextprotocol/server-memory (by Anthropic)
#      → Knowledge graph as local JSONL file
#      → github.com/modelcontextprotocol/servers  |  MIT License
#
#   2. @allpepper/memory-bank-mcp (by alioshr)
#      → Project context as local Markdown files
#      → github.com/alioshr/memory-bank-mcp  |  MIT License
#
# Requirements: Node.js 18+, npx, Claude Code CLI
# npx caches packages locally — works offline after first download
# ============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${GREEN}[MVHE]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERR]${NC} $1"; exit 1; }
info()  { echo -e "${CYAN}[INFO]${NC} $1"; }

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  MVHE Memory Stack — 100% Free Local FOSS Setup   ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# --- Pre-flight ---
command -v node   >/dev/null 2>&1 || error "Node.js not found → https://nodejs.org"
command -v npx    >/dev/null 2>&1 || error "npx not found (ships with Node ≥5.2)"
command -v claude >/dev/null 2>&1 || error "Claude Code CLI not found → npm i -g @anthropic-ai/claude-code"

NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
[ "$NODE_VER" -ge 18 ] || error "Node.js 18+ required (current: $(node -v))"
log "Pre-flight passed ✓"

# --- Project root ---
PROJECT_ROOT="$(pwd)"
if [ ! -f "$PROJECT_ROOT/CLAUDE.md" ] && [ ! -d "$PROJECT_ROOT/.git" ]; then
    warn "No CLAUDE.md or .git found in: $PROJECT_ROOT"
    read -rp "Use this directory? [y/N] " REPLY
    [[ $REPLY =~ ^[Yy]$ ]] || exit 0
fi

# --- Storage directories ---
MEMORY_BANK_DIR="$PROJECT_ROOT/.memory-bank"
KG_FILE="$PROJECT_ROOT/.aim/memory.jsonl"
mkdir -p "$MEMORY_BANK_DIR" "$(dirname "$KG_FILE")"

log "Local storage:"
log "  📁 Memory Bank:     $MEMORY_BANK_DIR"
log "  📊 Knowledge Graph: $KG_FILE"

# --- .gitignore ---
GITIGNORE="$PROJECT_ROOT/.gitignore"
touch "$GITIGNORE"
for entry in ".memory-bank/" ".aim/" "*.jsonl"; do
    grep -qxF "$entry" "$GITIGNORE" 2>/dev/null || echo "$entry" >> "$GITIGNORE"
done
log "Updated .gitignore ✓"

# --- MCP 1: Knowledge Graph (Anthropic official) ---
log ""
info "Installing 1/2: Knowledge Graph (@modelcontextprotocol/server-memory)..."
claude mcp add \
    --scope project \
    knowledge-graph \
    -e MEMORY_FILE_PATH="$KG_FILE" \
    -- npx -y @modelcontextprotocol/server-memory@latest
log "Knowledge Graph ✓ → create_entities, create_relations, search_nodes, read_graph"

# --- MCP 2: Memory Bank ---
log ""
info "Installing 2/2: Memory Bank (@allpepper/memory-bank-mcp)..."
claude mcp add \
    --scope project \
    memory-bank \
    -e MEMORY_BANK_ROOT="$MEMORY_BANK_DIR" \
    -- npx -y @allpepper/memory-bank-mcp@latest
log "Memory Bank ✓ → initialize_memory_bank, read/update_bank_file, get_bank_status"

# --- Done ---
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Setup Complete — 100% Local, 100% FOSS        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
claude mcp list 2>/dev/null || warn "Restart Claude Code to verify"
echo ""
info "License: MIT (both)  |  Cloud: NONE  |  API keys: NONE  |  Telemetry: NONE"
echo ""
log "Restart Claude Code and start building. Memory auto-activates."
log "Uninstall: claude mcp remove knowledge-graph && claude mcp remove memory-bank"
