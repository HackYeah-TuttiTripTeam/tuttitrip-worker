#!/usr/bin/env bash
# Build the worker image of one branch and (re)start its container if the
# backend env exists, then clean up images/containers of deleted branches.
#   deploy/deploy.sh <branch> [git-sha]
# Runs on the self-hosted runner `tuttitrip-worker-deploy` (on the host).
# Env (optional): OPENROUTER_API_KEY (GitHub secret), TT_SKIP_SMOKE=1.
# Host files: ~/tuttitrip/envs/<env>.worker.env (backend deploy),
#             ~/tuttitrip/worker.env (this script + hand edits).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)
# shellcheck source=deploy/lib.sh
. "$here/lib.sh"

branch=${1:?usage: deploy.sh <branch> [sha]}
sha=${2:-$(git -C "$repo" rev-parse HEAD)}
env=$(tt_env "$branch")
[ -n "$env" ] || { tt_log "branch '$branch' has an empty slug"; exit 1; }
if [ "$branch" != main ] && [ "$branch" != develop ] && tt_is_persistent "$env"; then
  tt_log "branch '$branch' collides with a persistent environment"; exit 1
fi
image=$(tt_image "$env")
container=$(tt_container "$env")

# --- build (outside the host lock; tagging is atomic) ---------------------------
docker build -q -t "$image" \
  --label tuttitrip.managed=true --label tuttitrip.role=worker \
  --label tuttitrip.env="$env" --label tuttitrip.branch="$branch" --label tuttitrip.sha="$sha" \
  "$repo" >/dev/null
tt_log "built $image ($sha)"

mkdir -p "$TT_STATE_DIR"
exec 9>"$TT_STATE_DIR/deploy.lock"
flock 9  # shared with the backend deploy: one change on the host at a time

# Worker settings shared by every env (mode 600, never committed). Created on
# first deploy with host defaults; keys already present are never changed, so
# hand edits survive. OPENROUTER_API_KEY comes from the GitHub secret.
umask 077
touch "$TT_STATE_DIR/worker.env"
ensure_key() {
  grep -q "^$1=" "$TT_STATE_DIR/worker.env" || printf '%s=%s\n' "$1" "$2" >>"$TT_STATE_DIR/worker.env"
}
# Ollama (nomic-embed-text) is reachable through the ollama_net network.
ensure_key TUTTITRIP_LLM__EMBEDDING_BASE_URL http://ollama:11434/v1
if [ -n "${OPENROUTER_API_KEY:-}" ]; then ensure_key OPENROUTER_API_KEY "$OPENROUTER_API_KEY"; fi

started=0
if [ -f "$(tt_env_file "$env")" ]; then
  # Replaces whatever runs there, including a develop/main fallback worker.
  tt_start_worker "$env" "$image"
  started=1
else
  tt_log "backend env '$env' does not exist (no $(tt_env_file "$env")):"
  tt_log "image $image is built; the backend deploy starts it when the env appears"
fi

# Dangling previous builds of our images (only ours: label filter).
docker image prune -f --filter label=tuttitrip.role=worker >/dev/null 2>&1 || true

# The read-only DBOS dashboard (https://tuttitrip-dbos.gburek.app) runs from
# tuttitrip-worker:main. Its container belongs to the backend's admin stack
# (tuttitrip-backend deploy/admin/); we only restart it on the new image.
admin_compose="$TT_STATE_DIR/admin/compose.yaml"
if [ "$env" = main ] && [ -f "$admin_compose" ]; then
  TT_ADMIN_STATE="$TT_STATE_DIR/admin" docker compose -f "$admin_compose" \
    --project-directory "$TT_STATE_DIR/admin" --profile dbos up -d --no-deps dbos-dashboard \
    && tt_log "tuttitrip-dbos-dashboard restarted on $image" \
    || tt_log "WARNING: could not restart tuttitrip-dbos-dashboard"
fi
flock -u 9

if [ "$started" = 1 ]; then
  tt_wait_healthy "$container"
  tt_log "$container is healthy"
  if [ "${TT_SKIP_SMOKE:-0}" != 1 ]; then
    "$here/smoke.sh" "$env"
  fi
fi

# A failed cleanup never fails the deploy (the cleanup workflow reports it).
"$here/cleanup.sh" || tt_log "WARNING: cleanup did not complete; deploy itself succeeded"
