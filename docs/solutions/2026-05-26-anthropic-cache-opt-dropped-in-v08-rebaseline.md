---
title: "Anthropic prompt-cache fork optimizations dropped in the v0.8 rebaseline (STORY-018 split + STORY-020)"
date: 2026-05-26
category: rebaseline-disposition
tags: [anthropic, prompt-caching, cache-optimization, system-prompt, rebaseline, dropped]
module: ["crates/zeroclaw-providers/src/anthropic.rs", "crates/zeroclaw-runtime/src/agent"]
stories: [STORY-018, STORY-020, STORY-011]
issues: ["Micra-io/zeroclaw#47", "Micra-io/zeroclaw#51"]
symptoms:
  - "Anthropic cache hit rate plateaus (~45-75% instead of ~95%)"
  - "the cached system block is invalidated every second by a per-call timestamp"
  - "`stable_prefix` / `system_with_stable_prefix` exists but is never wired into production"
---

# Why the fork's Anthropic prompt-cache optimizations were dropped in the 2026-05-26 v0.8 rebaseline

During the fork↔upstream rebaseline cutover (origin/main → upstream `v0.8.0-beta-1` + clean overlay;
spec `docs/superpowers/specs/2026-05-26-rebaseline-cutover-refresh-design.md`), two EPIC-005 (#51)
prompt-caching stories were **dropped** after an audit. This records why, so the work isn't
mistakenly re-attempted as a straight port.

## What was dropped

- **STORY-018 — Anthropic "stable system-block split"** (fork commits `4b5f108e` / PR #29 part). The
  re-applied commit added a `stable_prefix` field to `ChatMessage`, a `system_with_stable_prefix()`
  constructor, and two-block emission in `convert_messages` — **but never called it from production.**
  The ~600-line `SystemPromptParts` wiring from the *original* PR #29 that would set `stable_prefix`
  was not carried into the rebaseline branch. `git grep system_with_stable_prefix` finds **only test
  call sites**; every production literal sets `stable_prefix: None`. So the feature was **dead
  plumbing even on the fork branch** — porting it to v0.8 would add an always-`None` field across ~19
  files for zero behavioral effect.
- **STORY-020 — "move per-call content out of system message"** (fork commit `1b955996`, = old
  STORY-011, GitHub #47). Its tests call the dropped `system_with_stable_prefix`, so it **does not
  compile** against the v0.8 tree. A correct port is a **~500-line / 8-file rewrite** (move the
  per-second `## Current Date & Time` block and the mutable `Model:` line out of upstream's single
  cached system block, reorder truncation, and rewrite the cache-breakpoint guards against upstream's
  `convert_messages`). That is a cache **optimization** (cost/latency), not a correctness feature, and
  was out of scope for a clean rebaseline overlay.

## Why this is the right call (not a regression)

Current upstream **already has its own ephemeral `cache_control` scheme** that the fork commits
predate and would have collided with: `anthropic.rs::apply_oauth_system_prompt` emits a cached
identity prefix block; the system prompt is always emitted as `SystemPrompt::Blocks` with
`cache_control`; `should_cache_conversation` + `apply_cache_to_last_message` cache the conversation
tail, wired into both the streaming and non-streaming request builders. So the fork's *concern*
(cache-stable prompts) is largely served by upstream — just differently.

## The genuine residual gap (the real future work)

Upstream's cache scheme has one real weakness STORY-018/020 targeted: `system_prompt.rs` bakes a
**per-second timestamp** (`## Current Date & Time`) and the **mutable model name** into the *cached*
system block, so the system-prompt cache is effectively invalidated each second / on every `/model`
switch. A future **focused cache-opt story** (not an overlay port) should, against current upstream:

1. move the `## Current Date & Time` block out of the cached system prompt; emit the timestamp only on
   the user turn (upstream already puts `[{now}]` there at `loop_.rs`);
2. split `## Runtime` into a stable `## Host Environment` and relocate `Model:` to the user-turn
   preamble (it mutates via `/model`);
3. reorder `system_prompt.rs` so the stable suffix survives truncation;
4. write fresh cache guards against the single-block `convert_messages` (NOT the dropped
   `system_with_stable_prefix`), asserting the serialized system block carries no timestamp/`Model:`
   and exactly one `ephemeral` breakpoint;
5. add a test proving the timestamp still reaches the model via the user turn.

## What WAS kept

The other half of the fork's caching work — **chunked history trimming** (PR #30, `06672edb`) — was a
real, isolated delta upstream lacks and **was ported** (overlay commit `feat(fork/anthropic): chunked
history trim…`). It keeps the message-level cache prefix byte-stable for `max` turns instead of
trimming every turn. STORY-019 (history soft-target contract tests) was also ported.

**Bottom line:** the fork's Anthropic prompt-cache *optimizations* (STORY-018 split, STORY-020) are
deferred to a dedicated cache-opt story, not lost — production runs fine without them, and upstream's
own caching applies. GitHub #47 is closed; EPIC-005 (#51) tracks the deferral.
