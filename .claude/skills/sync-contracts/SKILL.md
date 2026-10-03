---
name: sync-contracts
description: Change the worker <-> backend job contract (workflow/queue names, payloads, events, CONTRACT_VERSION), regenerate contracts/jobs.schema.json and get the backend mirror updated. Use when changing a job payload, adding a job, or when contracts-check reports drift.
---

# Change or sync the job contract

The worker is canonical: `src/tuttitrip_worker/contracts.py` rendered to
`contracts/jobs.schema.json` by `scripts/export_contracts.py`. The backend
mirror is `src/tuttitrip/shared/jobs/contracts.py` in tuttitrip-backend,
rendered with the same algorithm (no titles/descriptions, sorted keys).

## Compatible change (new workflow, new optional field)

1. Change `contracts.py` (and the workflow, see `new-workflow`).
2. `uv run python scripts/export_contracts.py` and run all checks.
3. Merge here first (to `develop`), deploy, then mirror in the backend:
   same field names, types, constraints and defaults; regenerate its file.
4. `contracts-check` here warns until the mirror matches (diff in the job
   summary); the backend's `contracts-check` fails until it matches.

## Incompatible change (needs a new CONTRACT_VERSION)

1. Worker: add the new version to `SUPPORTED_CONTRACT_VERSIONS`, accept both
   shapes in the workflow (branch on `contract_version`), set
   `CONTRACT_VERSION` to the new one, deploy. The heartbeat now advertises
   `min_contract_version`=old, `contract_version`=new.
2. Backend: bump its `CONTRACT_VERSION`, change the models, regenerate.
3. Worker: once no old jobs are queued, drop the old version.

## Drift reported by CI

Read the diff (backend file vs ours). If the backend is behind, open the
mirror PR there. If we changed something by accident, revert it here.
