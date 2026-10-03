#!/usr/bin/env bash
# Remove worker images of branches that no longer exist in the worker repo.
# If the backend env of such a branch still exists, its worker is switched to
# the fallback image (develop, then main) instead of being left without one.
# main/develop and anything without our prefix + labels are never touched.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)
# shellcheck source=deploy/lib.sh
. "$here/lib.sh"

mkdir -p "$TT_STATE_DIR"
exec 9>"$TT_STATE_DIR/deploy.lock"
flock 9

branches=$(git -C "$repo" ls-remote --heads origin | sed 's#.*refs/heads/##')
if ! grep -qx main <<<"$branches" || ! grep -qx develop <<<"$branches"; then
  tt_log "branch list looks wrong (no main/develop), refusing to clean up"; exit 1
fi
declare -A live=()
while read -r b; do [ -n "$b" ] && live[$(tt_env "$b")]=1; done <<<"$branches"
is_stale() { ! tt_is_persistent "$1" && [ -z "${live[$1]:-}" ]; }

# 1. Containers running a stale image: switch to the fallback (or remove them
#    when the backend env is gone too; the backend cleanup would anyway).
switched=0
while read -r name image; do
  [ -n "$name" ] || continue
  [[ $name == "$TT_CONTAINER_PREFIX"-* ]] || continue
  [[ $image == "$TT_IMAGE_REPO":* ]] || continue
  image_env=${image#"$TT_IMAGE_REPO":}
  is_stale "$image_env" || continue
  env=${name#"$TT_CONTAINER_PREFIX"-}
  if [ -f "$(tt_env_file "$env")" ] && fallback=$(tt_pick_image "${TT_FALLBACK_ENVS[0]}"); then
    tt_log "$name runs stale $image; switching to $fallback"
    tt_start_worker "$env" "$fallback"
  else
    tt_log "$name runs stale $image and has no backend env; removing it"
    docker rm -f "$name" >/dev/null
  fi
  switched=$((switched + 1))
done < <(docker ps -a --filter label=tuttitrip.managed=true --filter label=tuttitrip.role=worker \
           --format '{{.Names}} {{.Image}}')

# 2. Stale images (nothing uses them any more).
removed=0
while read -r ref; do
  [ -n "$ref" ] || continue
  [[ $ref == "$TT_IMAGE_REPO":* ]] || continue
  image_env=${ref#"$TT_IMAGE_REPO":}
  is_stale "$image_env" || continue
  tt_log "removing stale image $ref"
  docker rmi "$ref" >/dev/null 2>&1 || tt_log "could not remove $ref (still in use?)"
  removed=$((removed + 1))
done < <(docker image ls --filter label=tuttitrip.managed=true --filter label=tuttitrip.role=worker \
           --format '{{.Repository}}:{{.Tag}}')

tt_log "cleanup done ($switched container(s) switched/removed, $removed image(s) removed)"
