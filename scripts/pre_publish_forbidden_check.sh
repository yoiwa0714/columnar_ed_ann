#!/usr/bin/env bash
set -euo pipefail

# 公開前に「公開してはいけないパス」が追跡対象に含まれていないかを検査する。
# 使い方:
#   ./scripts/pre_publish_forbidden_check.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v jj >/dev/null 2>&1; then
  echo "ERROR: jj command not found"
  exit 2
fi

REF="main"
if ! jj --no-pager bookmark list | grep -q '^main'; then
  REF='@-'
fi

TMP_TRACKED="tmp/pre_publish_tracked_paths.txt"
mkdir -p tmp
jj file list -r "$REF" > "$TMP_TRACKED"

declare -a FORBIDDEN_PATHS=(
  "chat_logs"
  "modules_backup/modules_v049_backup"
  ".gitignore"
  "columnar_ed_ann.code-workspace"
  "workspace.code-workspace"
  "scripts"
)

echo "[check] reference: $REF"
echo "[check] tracked paths file: $TMP_TRACKED"

violation=0
for p in "${FORBIDDEN_PATHS[@]}"; do
  # ディレクトリ指定は配下パスも違反扱いにする。
  if grep -Eq "^${p}(/.*)?$" "$TMP_TRACKED"; then
    echo "NG: forbidden tracked path detected -> $p"
    grep -E "^${p}(/.*)?$" "$TMP_TRACKED" | sed 's/^/  - /'
    violation=1
  fi
done

if [[ "$violation" -ne 0 ]]; then
  echo "FAILED: forbidden paths exist in tracked files"
  exit 1
fi

echo "OK: no forbidden tracked paths detected"
exit 0
