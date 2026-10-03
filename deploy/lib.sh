#!/usr/bin/env bash
# Shared naming rules for the worker's deploy scripts. Source it, don't run it.
# Must stay identical to tt_env in tuttitrip-backend/deploy/lib.sh
# (deploy/CONVENTIONS.md there is the source of truth).
#
#   branch            env              image                         container
#   main              main             tuttitrip-worker:main         tuttitrip-worker-main
#   develop           develop          tuttitrip-worker:develop      tuttitrip-worker-develop
#   feature/cos tam   feature-cos-tam  tuttitrip-worker:feature-cos-tam  tuttitrip-worker-feature-cos-tam
#
# Everything this tooling touches carries the `tuttitrip-worker` prefix and the
# labels tuttitrip.managed=true + tuttitrip.role=worker; nothing else is touched.

# shellcheck disable=SC2034  # variables are used by the scripts sourcing this file
TT_IMAGE_REPO="tuttitrip-worker"
TT_CONTAINER_PREFIX="tuttitrip-worker"
TT_API_PREFIX="tuttitrip-api"
TT_DOMAIN="${TT_DOMAIN:-gburek.app}"
TT_NETWORK="tuttitrip"
TT_STATE_DIR="${TT_STATE_DIR:-$HOME/tuttitrip}"
# Same cap as the backend: "tuttitrip-api-" + 49 chars = one 63-char DNS label.
TT_MAX_SLUG=49
# Fallback order when an env has no image of its own (CONVENTIONS.md).
TT_FALLBACK_ENVS=(develop main)
# Extra Docker networks the worker joins if they exist (local models on the
# host: the `ollama` container is reachable as http://ollama:11434 there).
TT_EXTRA_NETWORKS="${TT_EXTRA_NETWORKS:-ollama_net}"

# Lowercase, every run of non [a-z0-9] chars -> "-", trimmed, max 49 chars.
tt_slugify() {
  local s
  s=$(printf '%s' "$1" | LC_ALL=C tr '[:upper:]' '[:lower:]' \
      | LC_ALL=C sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
  s=${s:0:$TT_MAX_SLUG}
  printf '%s' "${s%-}"
}

# Environment label for a branch: main, develop or the branch slug.
tt_env() {
  case "$1" in
    main) printf 'main' ;;
    develop) printf 'develop' ;;
    *) tt_slugify "$1" ;;
  esac
}

tt_is_persistent() { [ "$1" = main ] || [ "$1" = develop ]; }
tt_image() { printf '%s:%s' "$TT_IMAGE_REPO" "$1"; }
tt_container() { printf '%s-%s' "$TT_CONTAINER_PREFIX" "$1"; }
# Worker env file, written by the backend deploy (restricted DB role, DBOS).
tt_env_file() { printf '%s/envs/%s.worker.env' "$TT_STATE_DIR" "$1"; }
tt_api_url() {
  if [ "$1" = main ]; then printf 'https://%s.%s' "$TT_API_PREFIX" "$TT_DOMAIN"
  else printf 'https://%s-%s.%s' "$TT_API_PREFIX" "$1" "$TT_DOMAIN"; fi
}

tt_log() { printf '[tuttitrip-worker-deploy] %s\n' "$*" >&2; }

tt_image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Is this container ours? (name prefix alone is not enough)
tt_is_our_container() {
  [ "$(docker inspect -f '{{index .Config.Labels "tuttitrip.role"}}' "$1" 2>/dev/null || true)" = worker ]
}

# First existing image for an env: its own, then develop, then main.
tt_pick_image() {
  local candidate
  for candidate in "$1" "${TT_FALLBACK_ENVS[@]}"; do
    if tt_image_exists "$(tt_image "$candidate")"; then
      tt_image "$candidate"; return 0
    fi
  done
  return 1
}

# (Re)start the worker of env $1 from image $2. Needs envs/<env>.worker.env.
# Env files (a later file overrides an earlier one), same as the backend's
# fallback start so both produce the same container:
#   envs/<env>.worker.env  written by the backend deploy (DB role, DBOS version)
#   worker.env             worker secrets/host settings (OPENROUTER_API_KEY, ...)
tt_start_worker() {
  local env=$1 image=$2 container envfile net
  container=$(tt_container "$env")
  envfile=$(tt_env_file "$env")
  [ -f "$envfile" ] || { tt_log "no $envfile: backend env '$env' does not exist"; return 1; }
  local args=(--env-file "$envfile")
  [ -f "$TT_STATE_DIR/worker.env" ] && args+=(--env-file "$TT_STATE_DIR/worker.env")
  if docker container inspect "$container" >/dev/null 2>&1; then
    tt_is_our_container "$container" || { tt_log "$container exists but is not ours; refusing"; return 1; }
    docker rm -f "$container" >/dev/null
  fi
  docker run -d --name "$container" --network "$TT_NETWORK" --restart unless-stopped \
    --stop-timeout 40 \
    --label tuttitrip.managed=true --label tuttitrip.role=worker --label tuttitrip.env="$env" \
    "${args[@]}" "$image" >/dev/null
  for net in $TT_EXTRA_NETWORKS; do
    if docker network inspect "$net" >/dev/null 2>&1; then
      docker network connect "$net" "$container" >/dev/null
    fi
  done
  tt_log "started $container from $image"
}

# Wait until the container's healthcheck says healthy.
tt_wait_healthy() {
  local container=$1 status
  for _ in $(seq 60); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo missing)
    [ "$status" = healthy ] && return 0
    [ "$status" = missing ] && break
    sleep 3
  done
  tt_log "$container is not healthy (status: ${status:-unknown})"
  docker logs --tail 60 "$container" >&2 || true
  return 1
}
