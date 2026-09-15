#!/usr/bin/env bash
# Replace the GitHub wiki pages with docs/wiki/*.md.
# The wiki must be enabled and have one page saved in the web UI so its git repository exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${WIKI_REMOTE:-$(git -C "$ROOT" remote get-url origin | sed 's/\.git$//').wiki.git}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --quiet "$REMOTE" "$WORKDIR/wiki"
find "$WORKDIR/wiki" -maxdepth 1 -name '*.md' -delete
cp "$ROOT"/docs/wiki/*.md "$WORKDIR/wiki/"

cd "$WORKDIR/wiki"
git add -A
if git diff --cached --quiet; then
  echo "Wiki already up to date."
  exit 0
fi
git commit --quiet -m "docs(wiki): sync from $(git -C "$ROOT" rev-parse --short HEAD)"
git push --quiet origin HEAD
echo "Wiki published to $REMOTE"
