---
name: deploy-debug
description: Diagnose a failed or misbehaving tuttitrip-worker deployment on dellpromaxgb10 (deploy job red, worker unhealthy, jobs stuck ENQUEUED, smoke test failing, wrong image after fallback). Use when the deploy workflow fails or the backend reports the worker missing.
---

# Debug a worker deployment

Host: `ssh cyprian@dellpromaxgb10` (Tailscale). Touch only `tuttitrip-worker-*`
containers and `tuttitrip-worker:*` images; never print env file contents.

1. CI: `gh run list --workflow CI -L 5`, `gh run view <id> --log-failed`.
   No runner picking up `deploy`? `systemctl status 'actions.runner.*tuttitrip-worker*'`.
2. Containers and images:
   `docker ps -a --filter label=tuttitrip.role=worker --format '{{.Names}} {{.Image}} {{.Status}}'`
   `docker image ls tuttitrip-worker`
3. Logs: `docker logs --tail 100 tuttitrip-worker-<env>`. Expect
   `DBOS launched!`, `Listening to 3 queues` and `worker ready: env=<env>`.
   - `permission denied for schema dbos` / missing `dbos` tables: the backend
     deploy did not run `dbos migrate -r tuttitrip_worker`; redeploy the backend env.
   - `permission denied for table ...`: grant missing in the backend's
     `deploy/worker-grants.sql`.
   - connection errors: the backend env (Postgres, `envs/<env>.worker.env`) is missing.
4. Jobs stuck `ENQUEUED`: compare versions. Worker log "Application version: X"
   must equal the backend's `TUTTITRIP_DBOS__APPLICATION_VERSION` (both `<env>`),
   application name `tuttitrip-worker` on both sides, and the queue name must
   be one the worker registered (`default`, `local_llm`, `openrouter`).
5. Health: `docker inspect -f '{{.State.Health.Status}}' tuttitrip-worker-<env>`;
   unhealthy = the main loop cannot query the DBOS system database.
6. Smoke test by hand: `curl -X POST https://tuttitrip-api-<env>.gburek.app/jobs/ping`
   then `curl https://tuttitrip-api-<env>.gburek.app/jobs/ping/<id>`
   (main: `https://tuttitrip-api.gburek.app`). Backend `/health` shows the heartbeat.
7. Embeddings failing: the container must be on `ollama_net`
   (`docker inspect -f '{{json .NetworkSettings.Networks}}' ...`); the backend's
   fallback start does not attach it, a worker deploy does.
8. Redeploy: re-run the `deploy` job, or on the host from a checkout:
   `deploy/deploy.sh <branch> <sha>`. Clean up: `deploy/cleanup.sh`.
