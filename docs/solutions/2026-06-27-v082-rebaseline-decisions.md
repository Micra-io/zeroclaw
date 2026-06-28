# v0.8.2 rebaseline — overlay decisions (2026-06-27)

Rebaseline of `Micra-io/zeroclaw` `main` from upstream **v0.8.1** (`768f8a28a`)
onto **v0.8.2** (`56b5a1f75`), 154 upstream commits. The 10-commit overlay
shrank to **7** (6 feature commits + a regenerated fmt/clippy cleanup). Two
features changed disposition versus a straight replay:

## 1. WhatsApp `allowed_groups` — dropped in favour of upstream #7720

Upstream v0.8.2 (`ba46f82d3`, PR #7720) shipped a **native**
`WhatsAppConfig.allowed_groups` plus a 4th `allowed_groups_resolver` argument to
`WhatsAppWebChannel::new` and a free `is_group_chat_allowed` gate. Its semantics
are narrower than the fork's old `is_chat_allowed`:

| | fork (old) | upstream #7720 |
|---|---|---|
| Match | full JID or user-part | full JID or user-part |
| Scope | groups **and** DMs | groups only (DMs bypass) |
| Sentinels | `"*"` (all), `"dm"` (DMs) | none |

**Decision: drop the fork's implementation, adopt upstream's; re-port only
`mention_name`.** Verified against the live mbp13 config — every other gating
field (`mode`, `dm_policy`, `group_policy`, `mention_only`, `self_chat_mode`,
mention-patterns) is **upstream-native** in v0.8.2; the fork only ever added
`allowed_groups` + `mention_name`. The production config
`allowed_groups = [JID_A, JID_B, "dm"]` + `dm_policy = "allowlist"` +
`mode = "personal"` behaves identically under upstream: the two group JIDs match,
the `"dm"` token becomes an inert no-op, and DMs (which bypass the group gate)
are filtered by the upstream-native `dm_policy = "allowlist"` against
`peer_groups.whatsapp_default.external_peers`. **`mention_name` ("claw") has no
upstream equivalent and is carried forward** (schema + struct field,
`contains_mention_name`/`strip_mention_name` Unicode-safe helpers, and the
mention_only text-fallback branch).

**Redeploy follow-up:** remove the now-dead `"dm"` entry from `allowed_groups`
in mbp13 `config.toml` (harmless but misleading under upstream's matcher).

## 2. Anthropic chunked history-trim — dropped (superseded by upstream #8196)

Upstream #8196 (`a8da39703`) ripped out history pruning/compression and replaced
in-loop per-message trimming with a **whole-turn reactive** model
(`agent/history_trim.rs::trim_to_recent_turns` at the turn budget boundary). The
fork's `/2` emergency-fallback hook site no longer exists, and whole-turn
dropping preserves a byte-stable prompt prefix at least as well as the fork's
per-message chunked trim — i.e. upstream now achieves the cache-stability goal
the fork commits (`ea0935868` + `a094b1e34`) were built for.

**Decision: drop both commits entirely.** No re-implementation.

## Replay notes (the non-trivial merges)

- **memory-metadata:** 3-way merge in `sqlite.rs store_with_agent` (kept
  upstream's graceful-embedding `match` + fork's `metadata` param/INSERT) and the
  orchestrator autosave (fork's `store_with_metadata` carrying the group_jid,
  storing upstream's image-stripped `autosave_content`). One upstream-added
  `MemoryEntry` literal (`conflict.rs`) needed `metadata: None`.
- **passive group storage:** layered `observe_store` additively on upstream's
  `allowed_groups_resolver`; observe write sits **after** the allowed_groups gate
  and **before** the mention_only gate.
- **group-context:** placed the injection **before** upstream's new
  `thinking.params.system_prompt_prefix` wrap; added `thinking_overrides` +
  `scope_overrides` (new upstream `ChannelRuntimeContext` fields) to the two
  fork-added test literals.

See `docs/superpowers/plans/2026-06-27-rebaseline-v082-plan.md` (gitignored) for
the full audit.
