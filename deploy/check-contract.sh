#!/usr/bin/env bash
# Compare our canonical contract (contracts/jobs.schema.json) with the backend
# mirror in tuttitrip-backend: same branch, else develop, else main.
# Reports drift (warning + diff in the job summary) without failing: the
# worker is canonical and the backend's contracts-check is the one that fails.
# Skips with a warning when BACKEND_REPO_TOKEN (GH_TOKEN) is not configured.
set -euo pipefail
repo="HackYeah-TuttiTripTeam/tuttitrip-backend"
path="contracts/jobs.schema.json"
branch="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
summary="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "::warning::BACKEND_REPO_TOKEN secret is not set; contract drift check skipped."
  exit 0
fi

fetch() {
  curl -fsS -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github.raw+json" \
    "https://api.github.com/repos/$repo/contents/$path?ref=$1"
}

theirs=$(mktemp)
trap 'rm -f "$theirs"' EXIT
used=""
for ref in "$branch" develop main; do
  if fetch "$ref" >"$theirs" 2>/dev/null; then used=$ref; break; fi
done
if [ -z "$used" ]; then
  echo "::warning::$repo has no $path on $branch, develop or main (or the token cannot read it); skipped."
  exit 0
fi

if diff_out=$(diff -u --label "tuttitrip-backend@$used:$path" --label "tuttitrip-worker:$path" \
  <(jq -S . "$theirs") <(jq -S . "$path")); then
  echo "Contract matches tuttitrip-backend@$used."
else
  echo "$diff_out"
  echo "::warning::Backend mirror (tuttitrip-backend@$used) differs from this contract; update the mirror (skill sync-contracts)."
  {
    echo "### Contract drift: tuttitrip-backend@$used vs this branch"
    echo '```diff'
    echo "$diff_out"
    echo '```'
  } >>"$summary"
fi
