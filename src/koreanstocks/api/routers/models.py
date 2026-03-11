"""모델 신뢰도 라우터 — GET /api/model_health"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from koreanstocks.core.config import config
from koreanstocks.core.constants import MIN_MODEL_AUC

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])

# params.json 파일이 저장된 디렉토리 — BASE_DIR 기준 (pipx/PyPI 설치 환경 호환)
PARAMS_DIR = Path(config.BASE_DIR) / "models" / "saved" / "model_params"

_MODEL_CONFIGS = [
    ("random_forest",     "랜덤 포레스트",      "random_forest_params.json"),
    ("gradient_boosting", "그래디언트 부스팅",   "gradient_boosting_params.json"),
    ("lightgbm",          "LightGBM",           "lightgbm_params.json"),
    ("catboost",          "CatBoost",           "catboost_params.json"),
    ("xgboost_ranker",    "XGBoost Ranker",     "xgboost_ranker_params.json"),
    ("tcn",               "TCN (딥러닝)",        "tcn_params.json"),
]

_MIN_AUC_THRESHOLD = MIN_MODEL_AUC  # core/constants.py 단일 소스


def _days_since(saved_at: str) -> int:
    """저장 시각(ISO 문자열) → 현재까지 경과 일수."""
    try:
        dt = datetime.fromisoformat(saved_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return max(0, (now - dt).days)
    except Exception:
        return -1


def _load_model_info(name: str, label: str, filename: str) -> dict | None:
    """단일 모델 params.json 로드 및 집계."""
    path = PARAMS_DIR / filename
    if not path.exists():
        logger.warning(f"model params not found: {path}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
    except Exception as e:
        logger.error(f"params 로드 실패 [{filename}]: {e}")
        return None

    # null 값 방어 — p.get(key, default)는 키가 존재하고 값이 null이면 None 반환
    _raw_auc = p.get("test_auc")
    test_auc  = float(_raw_auc) if _raw_auc is not None else 0.0
    _raw_cv  = p.get("cv_auc_mean")
    cv_mean   = float(_raw_cv)  if _raw_cv  is not None else 0.0
    saved_at  = p.get("saved_at") or ""
    model_type = p.get("model_type", "binary_classifier")
    # ranker는 test_logloss=None으로 저장됨 (log_loss 미정의)
    raw_logloss = p.get("test_logloss")
    test_logloss = float(raw_logloss) if raw_logloss is not None else None

    _raw_overfit = p.get("overfit_gap")
    overfit_gap  = float(_raw_overfit) if _raw_overfit is not None else 0.0

    architecture = p.get("architecture", "")  # TCN 등 딥러닝 모델 구분용
    # Log Loss 라벨: ranker → "N/A (ranker)", TCN → "N/A (딥러닝 BCE)", 그 외 → 값
    if test_logloss is None:
        logloss_label = "N/A (딥러닝 BCE)" if architecture else "N/A (ranker)"
    else:
        logloss_label = None  # JS에서 값으로 표시

    return {
        "name":               name,
        "label":              label,
        "model_type":         model_type,
        "architecture":       architecture,
        "test_auc":           test_auc,
        "train_auc":          p.get("train_auc", 0.0),
        "cv_auc_mean":        cv_mean,
        "cv_auc_std":         p.get("cv_auc_std", 0.0),
        "overfit_gap":        overfit_gap,
        "regime_gap":         round(test_auc - cv_mean, 4),
        "test_logloss":       test_logloss,
        "logloss_label":      logloss_label,
        "quality_pass":       p.get("quality_pass", False),
        "training_samples":   p.get("training_samples", 0),
        "purging_days":       p.get("purging_days", 0),
        "saved_at":           saved_at,
        "days_since_training": _days_since(saved_at),
        "training_duration":  p.get("training_duration", 0.0),
        "feature_importances": p.get("feature_importances", []),
    }


def _compute_ensemble(models: list[dict]) -> dict:
    """앙상블 집계 및 드리프트 등급 산출."""
    active    = [m for m in models if m is not None]
    n         = len(active)
    total_cfg = len(_MODEL_CONFIGS)   # 설정된 전체 모델 수 (파일 유무 무관)

    if n == 0:
        return {
            "active_count":       0,
            "total_model_count":  total_cfg,
            "tcn_active":         False,
            "mean_test_auc":      0.0,
            "mean_overfit_gap":   0.0,
            "mean_regime_gap":    0.0,
            "all_quality_pass":   False,
            "min_auc_threshold":  _MIN_AUC_THRESHOLD,
            "days_since_training": -1,
            "drift_level":        "HIGH",
            "drift_factors":      ["모델 파일 없음"],
            "retrain_recommended": True,
        }

    tcn_model        = next((m for m in active if m["name"] == "tcn"), None)
    tree_models      = [m for m in active if m["name"] != "tcn"]
    mean_test_auc    = round(sum(m["test_auc"]    for m in active) / n, 4)
    mean_overfit_gap = round(sum(m["overfit_gap"] for m in active) / n, 4)
    mean_regime_gap  = round(sum(m["regime_gap"]  for m in active) / n, 4)
    all_quality_pass  = all(m["quality_pass"] for m in active)
    any_date_unknown  = any(m["days_since_training"] == -1 for m in active)
    valid_days        = [m["days_since_training"] for m in active if m["days_since_training"] != -1]
    days_since        = max(valid_days) if valid_days else -1

    # 드리프트 등급 결정
    factors: list[str] = []
    retrain = False

    if any_date_unknown:
        factors.append("일부 모델의 학습 날짜 정보 없음 (saved_at 파싱 실패) — 재학습 권장")
        retrain = True
    if days_since != -1 and days_since > 30:
        factors.append(f"마지막 학습 {days_since}일 경과 (권장: 30일 이내)")
        retrain = True
    if mean_overfit_gap > 0.10:
        factors.append(f"평균 과적합 갭 {mean_overfit_gap:.4f} (임계: 0.10)")
        retrain = True
    if not all_quality_pass:
        factors.append(f"품질 기준 미달 모델 존재 (AUC < {_MIN_AUC_THRESHOLD})")
        retrain = True

    if retrain:
        drift_level = "HIGH"
    elif (days_since != -1 and days_since > 14) or mean_regime_gap > 0.07:
        drift_level = "MEDIUM"
        if days_since != -1 and days_since > 14:
            factors.append(f"마지막 학습 {days_since}일 경과 (권장: 14일 이내)")
        if mean_regime_gap > 0.07:
            factors.append(f"레짐 갭 {mean_regime_gap:.4f} > 0.07 (시장 의존도 주의)")
    else:
        drift_level = "LOW"

    return {
        "active_count":       n,
        "total_model_count":  total_cfg,
        "tcn_active":         tcn_model is not None,
        "tcn_test_auc":       tcn_model["test_auc"]    if tcn_model else None,
        "tcn_cv_auc":         tcn_model["cv_auc_mean"] if tcn_model else None,
        "tcn_overfit_gap":    tcn_model["overfit_gap"] if tcn_model else None,
        "tcn_quality_pass":   tcn_model["quality_pass"] if tcn_model else None,
        "tree_mean_test_auc": round(sum(m["test_auc"] for m in tree_models) / len(tree_models), 4) if tree_models else None,
        "mean_test_auc":      mean_test_auc,
        "mean_overfit_gap":   mean_overfit_gap,
        "mean_regime_gap":    mean_regime_gap,
        "all_quality_pass":   all_quality_pass,
        "min_auc_threshold":  _MIN_AUC_THRESHOLD,
        "days_since_training": days_since,
        "drift_level":        drift_level,
        "drift_factors":      factors,
        "retrain_recommended": retrain,
    }


@router.get("/model_health")
def get_model_health():
    """ML 모델 신뢰도 및 앙상블 상태 반환."""
    models = []
    for name, label, filename in _MODEL_CONFIGS:
        info = _load_model_info(name, label, filename)
        if info:
            models.append(info)

    ensemble = _compute_ensemble(models)

    return {
        "models": models,
        "ensemble": ensemble,
        "scoring_formula": {
            "with_ml":    "tech×0.40 + ml×0.35 + sentiment_norm×0.25",
            "without_ml": "tech×0.65 + sentiment_norm×0.35",
        },
    }
