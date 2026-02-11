# Session End Rule — Always Update Memory

At the END of every chat session (before the conversation closes), you MUST update all memory systems:

## Required Updates
1. **MEMORY.md** (`~/.claude/projects/.../memory/MEMORY.md`) — concise summary, keep under 200 lines
2. **Memory-bank MCP** — update relevant files via `mcp__memory-bank__memory_bank_update`:
   - `progress.md` — current status, latest changes, known issues
   - `architecture.md` — system design, infrastructure, schemas
   - `conventions.md` — coding patterns, rules, key files
   - `rules.md` — operational and deployment rules
   - `projectbrief.md` — project overview, status, performance
   - `performance-analysis.md` — trading stats (if changed)
3. **Knowledge graph MCP** — update entities and relations via `mcp__knowledge-graph__*`:
   - Add new entities for new components/features
   - Update observations on existing entities
   - Add relations between new and existing entities

## When to Trigger
- When user says "done", "that's all", "end session", or similar
- When a major task is completed
- When user explicitly asks to update memory
- Before context window fills up (proactively save state)

## What to Include
- Changes made in this session (files modified, features added, bugs fixed)
- New architectural decisions or patterns established
- New pitfalls/gotchas discovered
- Updated project state (balance, trades, deployments)
- Any new rules or conventions agreed upon
