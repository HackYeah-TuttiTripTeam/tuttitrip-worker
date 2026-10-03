#!/usr/bin/env bash
# End-to-end check of env $1: the backend enqueues `ping`, this worker runs it.
#   deploy/smoke.sh <env>
# POST <api>/jobs/ping returns a workflow id; GET <api>/jobs/ping/<id> must
# reach SUCCESS within ~3 minutes, otherwise the deploy fails.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=deploy/lib.sh
. "$here/lib.sh"

env=${1:?usage: smoke.sh <env>}
api=$(tt_api_url "$env")

code=$(curl -sS -o /tmp/tt-smoke.$$ -w '%{http_code}' -X POST "$api/jobs/ping" || true)
body=$(cat /tmp/tt-smoke.$$ 2>/dev/null || true); rm -f /tmp/tt-smoke.$$
if [ "$code" = 404 ]; then
  tt_log "WARNING: $api has no /jobs/ping (older backend); smoke test skipped"
  exit 0
fi
if [ "$code" != 202 ]; then
  tt_log "smoke test FAILED: POST $api/jobs/ping answered $code: $body"; exit 1
fi
id=$(jq -r .workflow_id <<<"$body")
state=""
for _ in $(seq 60); do
  state=$(curl -fsS "$api/jobs/ping/$id" | jq -r .status) || state=""
  case "$state" in SUCCESS|ERROR|CANCELLED|MAX_RECOVERY_ATTEMPTS_EXCEEDED) break ;; esac
  sleep 3
done
if [ "$state" != SUCCESS ]; then
  tt_log "smoke test FAILED: ping $id via $api ended as '${state:-unknown}'"
  curl -fsS "$api/jobs/ping/$id" >&2 || true
  docker logs --tail 50 "$(tt_container "$env")" >&2 || true
  exit 1
fi
tt_log "smoke test passed: $api enqueued ping $id, worker returned SUCCESS"
