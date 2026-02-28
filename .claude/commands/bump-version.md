프로젝트 전체의 버전 표기를 통일하는 작업을 수행해줘.

## 1단계: 목표 버전 결정

- `$ARGUMENTS`가 있으면 그 값을 목표 버전으로 사용한다. (예: `0.3.0`)
- `$ARGUMENTS`가 없으면 `pyproject.toml`의 `version` 필드 값을 정본으로 읽어서 목표 버전으로 사용한다.

## 2단계: 전체 버전 표기 스캔

아래 명령으로 프로젝트 내 모든 버전 표기를 찾는다 (dist/, .egg-info/, __pycache__ 제외):

```bash
grep -rn "0\.[0-9]\+\.[0-9]\+" . \
  --include="*.py" --include="*.toml" --include="*.md" \
  --include="*.html" --include="*.json" --include="*.yml" \
  | grep -v "dist/" | grep -v ".egg-info/" | grep -v "__pycache__"
```

## 3단계: 불일치 파일 수정

목표 버전과 다른 표기를 아래 파일에서 찾아 모두 수정한다.

| 파일 | 패턴 |
|---|---|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/koreanstocks/__init__.py` | `VERSION = "X.Y.Z"` |
| `src/koreanstocks/core/config.py` | `VERSION = "X.Y.Z"` |
| `src/koreanstocks/api/app.py` | `version="X.Y.Z"` (FastAPI) |
| `src/koreanstocks/static/dashboard.html` | `v{X.Y.Z}` (UI 표시) |
| `README.md` | 배지 `version-X.Y.Z-blue`, 구조도 주석 |
| `CLAUDE.md` | 헤더 `vX.Y.Z`, 구조도 주석 |
| `docs/ML_ANALYSIS.md` | 헤더 `vX.Y.Z` |
| `docs/NEWS_ANALYSIS.md` | 헤더 `vX.Y.Z` |
| `docs/TECHNICAL_ANALYSIS.md` | 헤더 `vX.Y.Z` |

**주의:** `docs/` 내 표·본문의 역사적 기록 (예: "v0.2.3 기준 실험", "v0.2.2 대비") 은 수정하지 않는다.

## 4단계: 검증

수정 후 동일한 스캔 명령을 다시 실행해서 불일치 항목이 없는지 확인한다.

## 5단계: commit & push

모든 수정이 완료되면 변경된 파일을 스테이징하고 아래 형식으로 커밋한다:

```
chore: 전체 버전 표기 vX.Y.Z으로 통일
```

그리고 push까지 완료한다.
