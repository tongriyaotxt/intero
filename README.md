# intero

**A memory & heartbeat organ for frozen LLMs — it remembers you, and speaks first.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen)](tests/)
[![MCP](https://img.shields.io/badge/MCP-server-purple)](.kimi/mcp.json.example)

[中文文档](README.zh-CN.md)

Your LLM forgets you the moment the session ends, and never speaks unless spoken to.
**intero** bolts two organs onto any frozen LLM — zero fine-tuning of the base model:

- 🧠 **Memory organ** — normalized atomic facts, salience-gated writes, nightly dream curation, contradiction arbitration. Survives restarts; auditable as a human-readable wiki.
- 💓 **Heartbeat organ** — an autonomy loop: intentions bid against a *silence floor* in an auction, internal drives (loneliness / curiosity / memory hygiene) sprout impulses even when nothing happened, and a daemon proactively pings you via Windows balloon / voice.

```
17:47  (you're away)  🎈 balloon: "今晚还去攀岩不？"
17:49  (you open your LLM CLI, new session)
       【它曾主动说】09-17 17:47 它说：今晚还去攀岩不？
       LLM picks up the thread and keeps chatting — with full memory context.
```

## Why not just RAG?

Head-to-head benchmark, 14-day simulated life (630 ingested items), reproduced by `bench/longitudinal.py`:

| | hit@5 | inventory | notes |
|---|---|---|---|
| **intero (light)** | **0.977** | 626 | curation: 553 consolidated / 17 dedup / 16 conflicts flagged |
| naive RAG (write-all) | 0.977 | 630 | no curation, contradictions pile up silently |
| intero (Titans full) | 0.98 | 482 | surprise-gated; λ stayed 0.0 — see honest notes |

Plus things RAG structurally can't do: expiry-aware forgetting, contradiction surfacing, and *speaking first*.

## Quickstart (Windows)

```bat
git clone https://github.com/tongriyaotxt/intero.git
cd intero
scripts\setup.bat       :: venv + deps + encoder prefetch + self-test
scripts\chat.bat        :: new window, Kimi CLI with intero MCP attached
```

Linux/macOS: `bash scripts/setup.sh`, then run the daemon:

```bash
INTERO_STORE=.intero/content.db PYTHONPATH=. python -m intero.daemon --serve
```

## Architecture

```
Host LLM (frozen) ── MCP stdio ──► mcp_server (thin client)
                                      │ localhost HTTP (~50ms)
                        ┌─────────────▼─────────────┐
                        │  daemon (always-on heart) │
                        │   Intero facade           │
                        │   ├─ ContentStore (sqlite: raw text + vectors, one tx)
                        │   ├─ gate: salient ∨ novel
                        │   ├─ Heartbeat: auction / refractory / circadian floor
                        │   ├─ drives: social · curiosity · memory_health
                        │   └─ dream cycle: curate → sprout intentions → export wiki
                        └──────┬───────────┬────────┘
                          balloon/voice   in-context thread
                          (you're away)   【它曾主动说】(you're back)
```

**MCP tools**: `memory_write` · `memory_recall` · `memory_status` · `add_intention` · `heartbeat_tick` · `dream_now`

Works with any MCP host. For Kimi CLI: `kimi --mcp-config-file .kimi/mcp.json` (see `.kimi/mcp.json.example`).

## Key ideas (all battle-tested, honestly reported)

- **LLM normalization beats fine-tuning encoders**: every open-source Chinese encoder failed whole-sentence negation ("我从来不喝咖啡" ≈ "我喝咖啡不加糖", cos 0.78). Rewriting user utterances into atomic facts before embedding removes the trap entirely — and stays human-auditable.
- **Write rate is the recall ceiling**: toy-scale benchmarks hide it; at life scale a pure-surprise gate collapsed hit@5 to 0.12. Fix: salience (∨ novelty) gate → 0.98. Full story in [bench/LONGITUDINAL.md](bench/LONGITUDINAL.md).
- **Proactivity needs brakes**: silence has a floor price, acting incurs a refractory period, and a feedback ledger (did the user engage?) down-weights topics you keep ignoring.
- **Everything survives process death**: MCP hosts spawn-per-call — all state (memories, heartbeat, intentions, feedback) lives in one sqlite file.

## Honest limitations

- Titans online weights (`INTERO_MODE=full`) are kept as an experimental layer; in our setting (frozen encoder + sparse events + personal scale) the parametric readout never beat chance (λ≡0), so **light mode is the default**. Not a refutation of Titans — a negative result under our constraints.
- Salience v1 is a regex over user self-reference; mixed real corpora will leak (LLM-scored importance is the v2 path).
- Single-user; Windows-first notifications (balloon/SAPI; other OS = easy PR).
- LLM features (normalization / intention sprouting / phrasing) need an OpenAI-compatible API key (`.env`, see `.env.example`); offline falls back to pass-through.

## Roadmap

- [ ] conversation digest → sharper followup/care intentions
- [ ] wiki as source of truth (currently read-only export) — real right-to-delete
- [ ] delivery channels: Telegram/IM bots
- [ ] multi-user isolation

## License

Apache-2.0. Model weights you plug in (e.g. bge) carry their own licenses.
