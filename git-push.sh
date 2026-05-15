#!/usr/bin/env bash
# git-push.sh — KoreanStocks 안전 push 스크립트
#
# GitHub Actions가 매일 data/storage/stock_analysis.db 를 커밋·푸시하므로,
# 로컬 DB 파일은 skip-worktree 플래그로 추적을 억제한다.
# 이 스크립트는 push 전 skip-worktree 를 임시 해제 → rebase → 복원을 자동 처리한다.
#
# 사용법:
#   ./git-push.sh              # 현재 브랜치 → origin/main
#   ./git-push.sh --no-pull    # pull(rebase) 없이 push만

set -euo pipefail

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-main}"
DB_PATH="data/storage/stock_analysis.db"
TEMP_DB="/tmp/koreanstocks_db_$(date +%s).bak"
NO_PULL=false

for arg in "$@"; do
  [[ "$arg" == "--no-pull" ]] && NO_PULL=true
done

# ── 정리 함수 (오류 시에도 반드시 실행) ────────────────────────────
cleanup() {
  local exit_code=$?
  local tmp="${TEMP_DB:-}"
  if [[ -n "$tmp" && -f "$tmp" ]]; then
    echo "  🔄 skip-worktree 복원 + 로컬 DB 복원..."
    git update-index --skip-worktree "$DB_PATH" 2>/dev/null || true
    cp "$tmp" "$DB_PATH"
    rm -f "$tmp"
  fi
  if [[ $exit_code -ne 0 ]]; then
    echo "❌ push 실패 (exit $exit_code). 로컬 DB는 복원됐습니다."
  fi
}
trap cleanup EXIT

# ── DB skip-worktree 상태 확인 ──────────────────────────────────────
DB_HAS_SKIP=false
if git ls-files -v "$DB_PATH" 2>/dev/null | grep -q "^S"; then
  DB_HAS_SKIP=true
fi

echo "🚀 KoreanStocks git push"
echo "   브랜치: $BRANCH → $REMOTE"

# ── 1. skip-worktree 임시 해제 + 로컬 DB 백업 ──────────────────────
if [[ "$DB_HAS_SKIP" == true ]]; then
  echo "  1/4 DB skip-worktree 임시 해제 + 로컬 DB 백업..."
  git update-index --no-skip-worktree "$DB_PATH"
  cp "$DB_PATH" "$TEMP_DB"
  # 커밋된 DB 버전으로 복원 (rebase/merge 충돌 방지)
  git checkout HEAD -- "$DB_PATH"
else
  echo "  1/4 DB skip-worktree 없음 — 그대로 진행"
fi

# ── 2. 원격 변경사항 rebase ─────────────────────────────────────────
if [[ "$NO_PULL" == false ]]; then
  echo "  2/4 원격 변경사항 pull --rebase..."
  git pull --rebase "$REMOTE" "$BRANCH"
else
  echo "  2/4 --no-pull 옵션: pull 생략"
fi

# ── 3. push ────────────────────────────────────────────────────────
echo "  3/4 push..."
git push "$REMOTE" "$BRANCH"

# ── 4. 정리 (cleanup 함수에서 처리) ────────────────────────────────
echo "  4/4 skip-worktree 복원 + 로컬 DB 복원..."
# cleanup() 이 EXIT trap 으로 실행됨 — 여기서 TEMP_DB 삭제 후 trap 무력화
git update-index --skip-worktree "$DB_PATH" 2>/dev/null || true
cp "$TEMP_DB" "$DB_PATH"
rm -f "$TEMP_DB"
# TEMP_DB 삭제했으므로 trap에서 중복 실행 방지
TEMP_DB=""

echo "✅ 완료!"
git log --oneline -3
