# STORY-003: Isolate Session History Per Conversation Context

## Story

**As** a user chatting with ZeroClaw across multiple groups and DMs,
**I want** my conversation history to be isolated per chat context (group/channel/DM),
**so that** the agent doesn't mix unrelated conversations and has the right context for each.

## Priority

Must Have

## Story Points

5

## Epic

EPIC-003: ZeroClaw Enhancements

## Context

Session history is keyed by `{channel}_{sender}` (e.g. `slack_U038GEE07`, `whatsapp_+34622922443`), so all messages from the same user across different groups, channels, and DMs are mixed into a single conversation context. The `reply_target` field on `ChannelMessage` contains the chat context (group JID for WhatsApp, channel ID for Slack) but is not included in the session key.

**Original bug (v0.1.8):** Session history was not persisted after LLM response. Fixed in v0.5.0 — `append_sender_turn()` now persists each turn immediately.

**Current bug (v0.5.0):** Session key lacks chat context, causing cross-conversation bleed.

## Expert Review Summary (2026-03-19)

Four experts reviewed this story: architect, data-integrity-guardian, performance-oracle, security-sentinel. Key findings:

### Blockers Identified

1. **Hydration key mismatch (data-guardian):** The JSONL `session_path()` sanitizer is lossy — replaces `+`, `@`, `.` with `_`. Session keys can't round-trip through disk. WhatsApp sessions (`whatsapp_+34622922443`) sanitize to filename `whatsapp__34622922443` but runtime lookups use the unsanitized key. **Sessions already don't survive restarts.**

2. **No TTL cleanup for daemon sessions (perf-oracle, data-guardian):** `cleanup_stale()` is only called for gateway sessions. Channel daemon JSONL files accumulate forever. The proposed key change multiplies file count (senders × groups).

### Recommended Path: Switch to SqliteSessionBackend

Both blockers are solved by switching the channel daemon from `SessionStore` (JSONL) to `SqliteSessionBackend` (already exists in `session_sqlite.rs`):
- SQLite stores keys as-is (no sanitizer, no lossy encoding)
- SQLite has working `cleanup_stale()` with TTL
- `migrate_from_jsonl()` already exists and is tested
- FTS5 search across sessions is a bonus

### Design Decisions Confirmed

- `reply_target` is a non-optional `String` — always populated across all channels (architect)
- Pattern already validated by `interruption_scope_key()` at line 398 (architect)
- Do NOT change `conversation_memory_key()` — memory uses per-message UUIDs, no isolation issue (architect)
- The change is a net security improvement — eliminates cross-context bleed (sec-sentinel)
- No performance concern at current scale (~3x file/entry multiplier is negligible) (perf-oracle)

## Acceptance Criteria

- [ ] Channel daemon uses `SqliteSessionBackend` instead of `SessionStore` (JSONL)
- [ ] Existing JSONL sessions migrated via `migrate_from_jsonl()`
- [ ] `conversation_history_key()` includes `reply_target` for per-context isolation
- [ ] WhatsApp group messages have separate session from DMs
- [ ] Slack messages in different channels have separate sessions
- [ ] Session history survives daemon restarts within TTL window
- [ ] `cleanup_stale()` wired up for channel daemon sessions
- [ ] No regression in multi-turn within a single conversation context

## Technical Notes

### Implementation Plan

#### Step 1: Switch channel daemon to SqliteSessionBackend

In `src/channels/mod.rs` (~line 4205), change session store initialization from `SessionStore::new()` to `SqliteSessionBackend::new()`. Run `migrate_from_jsonl()` at startup to import existing JSONL data.

#### Step 2: Update conversation_history_key()

In `src/channels/mod.rs` (~line 386):

```rust
fn conversation_history_key(msg: &traits::ChannelMessage) -> String {
    match &msg.thread_ts {
        Some(tid) => format!("{}_{}_{}_{}", msg.channel, msg.reply_target, tid, msg.sender),
        None => format!("{}_{}_{}", msg.channel, msg.reply_target, msg.sender),
    }
}
```

This matches the existing `interruption_scope_key()` format at line 398.

#### Step 3: Wire up cleanup_stale() for daemon sessions

Add periodic `cleanup_stale()` invocation in the daemon heartbeat or scheduler loop, consuming the existing `session_ttl_hours` config.

### Key Files

- `src/channels/mod.rs` — `conversation_history_key()` (~line 386), session store init (~line 4205), startup hydration (~line 4223)
- `src/channels/session_sqlite.rs` — `SqliteSessionBackend`, `migrate_from_jsonl()`
- `src/channels/session_store.rs` — JSONL backend (being replaced)
- `src/channels/session_backend.rs` — `SessionBackend` trait, `cleanup_stale()`

### Verification

1. Start daemon → verify "Migrated N JSONL sessions" log (if JSONL files exist)
2. Send message in WhatsApp group → verify session created with group JID in key
3. Send DM → verify separate session entry in sessions.db
4. Restart daemon → verify session history restored correctly
5. Repeat for Slack across two different channels
6. Wait past TTL → verify `cleanup_stale()` removes expired sessions

### Follow-up Items (separate PRs)

- Set explicit file permissions (0600/0700) on sessions directory and DB (sec-sentinel)
- Switch `Vec<ChatMessage>` to `VecDeque` for O(1) history trim (perf-oracle)
- Fix FTS5 quote escaping in SQLite backend search (sec-sentinel)

## Dependencies

None

## Risk

- **Medium** — touches session initialization and key format, but:
  - SQLite backend already exists with full test coverage
  - JSONL migration is handled by existing `migrate_from_jsonl()`
  - Old JSONL files renamed to `.jsonl.migrated` (reversible)
  - `reply_target` pattern already validated by `interruption_scope_key()`
