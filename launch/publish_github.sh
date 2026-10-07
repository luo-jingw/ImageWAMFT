#!/usr/bin/env bash
# Publishes the local `main` to the public GitHub repository luo-jingw/ImageFT (PROJECT.md -> Credentials).
#
# `papers/` is removed from every commit of the published history: the arXiv PDF is under arXiv's non-exclusive
# distribution license and may not be redistributed. The rewrite runs in a scratch clone, so the local repository keeps
# its full history and its pre-push hook. Published commit hashes therefore differ from the local ones. The rewrite is
# deterministic (filter-branch keeps authors, committers, dates and messages), so a later run fast-forwards the remote.
# The token is read from TOKEN_FILE by a per-command credential helper and is never written to git config.
#   bash launch/publish_github.sh
set -euo pipefail
PROJECT_ROOT=/home/user1/workspace/jingwu/ImageWAMFT
REMOTE=https://github.com/luo-jingw/ImageFT.git
TOKEN_FILE=/home/user1/workspace/jingwu/github_token

WORK="$(mktemp -d "${TMPDIR:-/tmp}/imagewamft_publish.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
git clone --quiet --no-local --single-branch --branch main "${PROJECT_ROOT}" "${WORK}/repo"
cd "${WORK}/repo"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --prune-empty \
  --index-filter 'git rm -r --cached --quiet --ignore-unmatch papers' -- main >/dev/null
if git rev-list main -- papers | grep -q .; then
  echo "publish_github: papers/ is still present in the rewritten history" >&2
  exit 1
fi
git -c credential.helper= \
  -c "credential.helper=!f() { echo username=luo-jingw; printf 'password=%s\n' \"\$(tr -d '[:space:]' < ${TOKEN_FILE})\"; }; f" \
  push "${REMOTE}" main:main
echo "published $(git rev-parse --short main) (local $(git -C "${PROJECT_ROOT}" rev-parse --short main)) to ${REMOTE}"
