# /bump-version

프로젝트 전체의 버전 표기를 단일 버전으로 통일하는 작업을 수행합니다.

## 작업 절차

### 1단계: 현재 버전 기준 확인

`pyproject.toml`의 `version` 필드를 **정본(source of truth)**으로 사용합니다.

```bash
grep '^version' pyproject.toml
```

### 2단계: 전체 버전 표기 스캔

아래 명령으로 프로젝트 내 모든 버전 표기를 찾습니다 (dist/, .egg-info/, __pycache__ 제외):

```bash
grep -rn "0\.[0-9]\+\.[0-9]\+" . \
  --include="*.py" --include="*.toml" --include="*.md" \
  --include="*.html" --include="*.json" --include="*.yml" --include="*.yaml" \
  | grep -v "dist/" | grep -v ".egg-info/" | grep -v "__pycache__"
```

### 3단계: 버전 불일치 파악 및 수정

정본 버전과 다른 표기를 찾아 모두 수정합니다.

**수정 대상 파일 목록 (이 프로젝트 기준):**

| 파일 | 위치 | 형식 |
|---|---|---|
| `pyproject.toml` | `version = "X.Y.Z"` | 정본 |
| `src/koreanstocks/__init__.py` | `VERSION = "X.Y.Z"` | 동기화 |
| `src/koreanstocks/core/config.py` | `VERSION = "X.Y.Z"` | 동기화 |
| `src/koreanstocks/api/app.py` | `version="X.Y.Z"` | FastAPI/Swagger UI |
| `src/koreanstocks/static/dashboard.html` | `v{X.Y.Z}` | 대시보드 UI |
| `README.md` | 배지 `version-X.Y.Z-blue`, 구조도 주석 | 문서 |
| `CLAUDE.md` | 헤더 `vX.Y.Z`, 구조도 주석 | 문서 |
| `docs/ML_ANALYSIS.md` | 헤더 `vX.Y.Z` | 기술 문서 |
| `docs/NEWS_ANALYSIS.md` | 헤더 `vX.Y.Z` | 기술 문서 |
| `docs/TECHNICAL_ANALYSIS.md` | 헤더 `vX.Y.Z` | 기술 문서 |

**주의:** `docs/` 내 표/본문에서 이전 버전을 언급하는 **역사적 기록** (예: "v0.2.2 대비", "v0.2.3 기준 실험 결과")은 수정하지 않습니다.

### 4단계: 수정 완료 후 검증

수정 후 다시 스캔하여 불일치 항목이 없는지 확인합니다.

### 5단계: commit

```
chore: 전체 버전 표기 vX.Y.Z으로 통일
```

## $ARGUMENTS 처리

- 인자가 있으면 (`/bump-version 0.3.0`) 해당 버전으로 모든 파일을 업데이트합니다.
- 인자가 없으면 (`/bump-version`) pyproject.toml 기준으로 불일치 항목만 찾아 동기화합니다.
