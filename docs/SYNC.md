# Keeping the Micra-io/zeroclaw fork in sync with upstream

This fork's `main` = **current `upstream/master` + a small, clean overlay of fork-only feature commits.**
That structure is the whole point — it makes each upstream sync a mechanical rebase, not a rescue
operation. Keep it that way.

> **Remotes:** `upstream` = `zeroclaw-labs/zeroclaw` (read-only), `origin` = `Micra-io/zeroclaw` (the fork).
> Last clean re-baseline: 2026-06-19 (`docs/superpowers/specs/2026-06-19-rebaseline-refresh-design.md`).

## Prime directive: sync small and often

The painful 560-commits-behind re-baseline happened because the fork was left to drift. A ~30-commit
catch-up is minutes; a 500-commit one is a multi-day project. **Sync weekly, or whenever upstream is
~50–100 commits ahead — never more.** (See the drift check at the bottom.)

## Two lanes — don't mix them

| What | Mechanism | Why |
|---|---|---|
| **Fork `main`** | **rebase** the overlay onto fresh `upstream/master` | keeps `main` = upstream + clean droppable per-feature commits |
| **Your upstream PR branches** (e.g. `feat/otel-gen-ai-attrs`) | **merge** `upstream/master` in (don't rebase) | upstream squash-merges your PR, so merge commits are harmless and rebasing churns reviewers |

## The sync loop (fork `main`)

```bash
cd /Users/alexanderalyushin/micra-io/zeroclaw
git fetch upstream origin

# How far behind are we? (sanity check before starting)
git rev-list --count main..upstream/master       # upstream commits we don't have yet

# Work on a throwaway sync branch, never on main directly
git checkout -B sync/$(date +%Y-%m-%d) origin/main

# Replay the overlay onto the new upstream tip. merge-base is found automatically;
# the only commits replayed are our fork overlay (everything on main not in upstream/master).
git rebase upstream/master
#   - resolve conflicts per-feature (they're localized to the files each commit touches)
#   - if a fork feature is now UPSTREAM (its PR merged, or upstream added an equivalent):
#       DROP that commit — `git rebase --skip` (or delete its line in an interactive rebase).
#       Verify with: git merge-base --is-ancestor <equiv-upstream-sha> upstream/master
#   - the overlay should SHRINK over time, never grow.
```

### Validate (same gates as the re-baseline)

```bash
cargo build  --workspace --features whatsapp-web,observability-otel
cargo clippy --workspace --all-targets --exclude zeroclaw-desktop \
             --features whatsapp-web,channel-telegram,observability-otel -- -D warnings
cargo fmt --check
cargo nextest run --workspace --features whatsapp-web,channel-telegram,observability-otel --retries 2
```

Notes learned the hard way:
- **Use the full feature set** above. `--features whatsapp-web` alone does **not** compile the Telegram
  tests (`channel-telegram` is separate), and the fork features live behind these flags.
- `tools::delegate::*` background-task tests are **load-flaky** under the full parallel suite (they spawn
  real work and poll with a timeout). `--retries 2` absorbs them; they pass in isolation. Not a fork bug.
- `cargo deny check` runs in CI; install locally with `cargo install cargo-deny` if you want it pre-push.

### Cut over `main` (force-push — `main` is protected)

`main` has `allow_force_pushes: false`, so a rebased `main` needs the protection toggled. Capture →
enable → push → **restore**, all in one go so protection is never left open:

```bash
# 1. capture current protection + build enable/restore payloads
python3 - <<'PY'
import json, subprocess
o = json.loads(subprocess.check_output(["gh","api","repos/Micra-io/zeroclaw/branches/main/protection"]))
def p(force):
    d={"required_status_checks":(None if not o.get("required_status_checks") else
        {"strict":o["required_status_checks"]["strict"],"contexts":o["required_status_checks"].get("contexts",[])}),
       "enforce_admins":o.get("enforce_admins",{}).get("enabled",False),"restrictions":None,
       "required_pull_request_reviews":None,"allow_force_pushes":force}
    r=o.get("required_pull_request_reviews")
    if r: d["required_pull_request_reviews"]={k:r[k] for k in
        ["dismiss_stale_reviews","require_code_owner_reviews","required_approving_review_count","require_last_push_approval"] if k in r}
    for k in ["required_linear_history","allow_deletions","block_creations","required_conversation_resolution","lock_branch","allow_fork_syncing"]:
        if isinstance(o.get(k),dict): d[k]=o[k]["enabled"]
    return d
json.dump(p(True), open("/tmp/prot_enable.json","w")); json.dump(p(False), open("/tmp/prot_restore.json","w"))
print("captured; allow_force_pushes was", o.get("allow_force_pushes",{}).get("enabled"))
PY

OLD=$(git rev-parse origin/main)            # lease baseline (fail-closed if main moved)
gh api -X PUT repos/Micra-io/zeroclaw/branches/main/protection --input /tmp/prot_enable.json >/dev/null
git push --force-with-lease=main:$OLD origin sync/$(date +%Y-%m-%d):main
gh api -X PUT repos/Micra-io/zeroclaw/branches/main/protection --input /tmp/prot_restore.json >/dev/null
rm -f /tmp/prot_enable.json /tmp/prot_restore.json

# verify
git fetch origin && git rev-parse origin/main      # == your sync branch HEAD
gh api repos/Micra-io/zeroclaw/branches/main/protection | python3 -c \
  "import json,sys; print('allow_force_pushes restored to', json.load(sys.stdin)['allow_force_pushes']['enabled'])"
```

> Account must be `alexandme` (admin): `gh auth switch --user alexandme`.
> Prefer not toggling each time? Either accept this script, or relax `main` protection to allow admin
> force-push (weaker, but reasonable for a solo fork that rewrites `main` by design).

### Rollback

Every cutover tags the prior `main`: `archive/pre-cutover-YYYY-MM-DD`. To revert:
```bash
git push --force-with-lease origin archive/pre-cutover-<date>:main   # (toggle protection as above)
```

## Keep the overlay shrinking

The overlay should trend toward **only the features upstream will never take** (WhatsApp
`mention_only`/`allowed_groups`/group-context/vCard/passive-observation, Telegram `allowed_chats`/
`allowed_dm_users`, memory `metadata`+group-JID, cron simplified-delivery gating — upstream has
**declined** the WhatsApp/group PRs #2564/#3923/#4705).

For anything general-purpose (observability, caching, cost): upstream it as a PR (lane 2) and **drop it
from the overlay the moment it merges**. Each sync, before re-applying a "maybe upstreamed" commit,
audit it — if upstream now provides the behavior, drop the fork commit.

## Production redeploy is separate

Syncing `main` does **not** touch the running daemon (it runs a pinned binary). Before redeploying a
freshly-synced `main`, mind the **v0.7→v0.8 on-disk layout migration** (`data_dir` moved to
`~/.zeroclaw/data`; the maresme `sessions.db` stays at `~/.zeroclaw/workspace/sessions/`) — see the
deploy runbook and `docs/superpowers/plans/2026-05-26-rebaseline-cutover.md` Phase 5.

## Drift check (run weekly / schedule it)

```bash
git fetch upstream --quiet
echo "upstream is $(git rev-list --count main..upstream/master) commits ahead of fork main"
```

Wire this into the daemon cron (or a scheduled GitHub Action) to ping when the count crosses ~50, so you
sync before it drifts.
