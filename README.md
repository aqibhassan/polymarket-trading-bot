# MVHE — Claude Code Brain + Memory Setup

## What This Is

A complete **modular brain architecture** for Claude Code, purpose-built for the Micro-Volatility Harvesting Engine. Every component is **100% free, open source, and local**.

```
Zero cloud services    Zero API keys    Zero subscriptions
Zero telemetry         MIT licensed     Your data stays yours
```

---

## Architecture Overview

```
CLAUDE.md (root)                    ← Lean routing index (~60 lines)
│
├── .claude/
│   ├── rules/                      ← Path-scoped rules (auto-load by file path)
│   │   ├── strategies.md           ← Loads for src/strategies/**
│   │   ├── execution.md            ← Loads for src/execution/**
│   │   ├── risk.md                 ← Loads for src/risk/**
│   │   ├── data-pipeline.md        ← Loads for src/data/**
│   │   ├── testing.md              ← Loads for tests/**
│   │   └── security.md             ← Always active (no path scope)
│   │
│   ├── skills/                     ← Deep domain knowledge (on-demand)
│   │   ├── volatility-engine/SKILL.md
│   │   ├── risk-management/SKILL.md
│   │   ├── data-pipeline/SKILL.md
│   │   └── backtesting/SKILL.md
│   │
│   └── commands/                   ← Slash commands
│       ├── new-strategy.md         ← /new-strategy
│       └── backtest-review.md      ← /backtest-review
│
├── .memory-bank/                   ← Memory Bank MCP (project markdown files)
│   ├── activeContext.md            ← What Claude is currently working on
│   ├── progress.md                 ← Project progress tracking
│   └── decisions.md                ← Recent decisions and rationale
│
├── .aim/                           ← Knowledge Graph MCP (JSONL)
│   └── memory.jsonl                ← Entities, relations, observations
│
└── docs/brain/                     ← Reference docs Claude can read
    ├── ARCHITECTURE.md             ← System design, data flow, schemas
    └── DECISIONS.md                ← Architecture Decision Records
```

## How the Two Memory Systems Work Together

| Feature | Memory Bank | Knowledge Graph |
|---------|-------------|-----------------|
| **Format** | Markdown files | JSONL (entities + relations) |
| **Best for** | Current context, progress, notes | Relationships, patterns, facts |
| **Scope** | Project-level | Project-level (can be global) |
| **Example** | "Currently building Binance WS connector" | "BinanceConnector --uses--> WebSocket --has--> reconnect logic" |
| **Persistence** | Survives sessions | Survives sessions |
| **License** | MIT | MIT |
| **Cloud** | None | None |

**Memory Bank** = "What am I working on right now?"
**Knowledge Graph** = "What do I know about this codebase?"

## How Rules Auto-Load

Claude Code's path-scoped rules activate **automatically** based on which files you're working on:

```yaml
# .claude/rules/strategies.md has this frontmatter:
---
paths:
  - "src/strategies/**/*.py"
  - "tests/strategies/**/*.py"
---
```

When you ask Claude to edit `src/strategies/mean_reversion.py`, the strategies rule loads automatically. You never need to reference it manually.

**Rules without path scope** (like `security.md`) load for **every** task.

## Setup (2 Minutes)

### Prerequisites
- Node.js 18+ (`node -v`)
- Claude Code CLI (`claude --version`)

### Install

```bash
cd /path/to/your/mvhe/project
chmod +x scripts/setup-memory.sh
./scripts/setup-memory.sh
```

### Verify

```bash
# Restart Claude Code
claude

# Check servers are running
claude mcp list

# You should see:
#   knowledge-graph  (project)  npx @modelcontextprotocol/server-memory
#   memory-bank      (project)  npx @allpepper/memory-bank-mcp
```

### Manual Install (if script doesn't work)

```bash
# Knowledge Graph — Anthropic's official MCP server
claude mcp add --scope project knowledge-graph \
  -e MEMORY_FILE_PATH="$(pwd)/.aim/memory.jsonl" \
  -- npx -y @modelcontextprotocol/server-memory@latest

# Memory Bank — project context files
claude mcp add --scope project memory-bank \
  -e MEMORY_BANK_ROOT="$(pwd)/.memory-bank" \
  -- npx -y @allpepper/memory-bank-mcp@latest
```

## FOSS Verification

| Component | Source | License | Data Storage |
|-----------|--------|---------|-------------|
| `@modelcontextprotocol/server-memory` | [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | MIT | Local `.aim/memory.jsonl` |
| `@allpepper/memory-bank-mcp` | [github.com/alioshr/memory-bank-mcp](https://github.com/alioshr/memory-bank-mcp) | MIT | Local `.memory-bank/*.md` |
| Claude Code rules/skills | Built-in Claude Code feature | N/A | Local `.claude/` directory |
| All brain docs | This repo | Your IP | Local `docs/brain/` |

**No cloud services. No API keys. No accounts. No telemetry. No cost.**

## Uninstall

```bash
claude mcp remove knowledge-graph
claude mcp remove memory-bank
rm -rf .memory-bank .aim
```

## File Inventory

```
scripts/setup-memory.sh                    Setup script
CLAUDE.md                                  Root brain (routing index)
.claude/rules/strategies.md                Strategy code rules
.claude/rules/execution.md                 Execution/Rust rules
.claude/rules/risk.md                      Risk management rules
.claude/rules/data-pipeline.md             Data pipeline rules
.claude/rules/testing.md                   Testing rules
.claude/rules/security.md                  Security rules (always on)
.claude/skills/volatility-engine/SKILL.md  Volatility estimation knowledge
.claude/skills/risk-management/SKILL.md    Risk/sizing knowledge
.claude/skills/data-pipeline/SKILL.md      Exchange connectivity knowledge
.claude/skills/backtesting/SKILL.md        Backtesting framework knowledge
.claude/commands/new-strategy.md           /new-strategy slash command
.claude/commands/backtest-review.md        /backtest-review slash command
docs/brain/ARCHITECTURE.md                 System architecture reference
docs/brain/DECISIONS.md                    Architecture decision records
```
