"""프로젝트 공통 상수 모음"""
from typing import Dict, List, Tuple

# ── 버킷 (후보군 분류) ──────────────────────────────────────────────────────

# 버킷 기본값 (bucket 필드가 없는 종목에 배정)
BUCKET_DEFAULT: str = 'volume'

# 버킷 한국어 레이블 (대시보드·슬라이드 배지, 추천 결과 저장용)
BUCKET_LABELS: Dict[str, str] = {
    'volume':   '거래량 상위',
    'momentum': '상승 모멘텀',
    'rebound':  '반등 후보',
}

# 버킷별 후보 풀 할당 비율 (합계 = 1.0)
BUCKET_RATIOS: List[Tuple[str, float]] = [
    ('volume',   0.40),
    ('momentum', 0.35),
    ('rebound',  0.25),
]
