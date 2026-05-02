from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RowFeatures:
    baseline_len: int
    current_best_len: int
    preopt_saved: int
    row_rank: int
    # Optional adaptive-routing signals. Defaults preserve compatibility with
    # previous call sites that only passed the original four fields.
    total_rows: int = 100
    historical_saved_moves: int = 0
    historical_runtime_s: float = 0.0
    regression_count: int = 0


@dataclass(frozen=True)
class PolicyConfig:
    # Legacy absolute thresholds kept as fallback / compatibility.
    light_min_path_len: int = 560
    aggressive_min_path_len: int = 700
    force_aggressive_top_n: int = 24
    min_improvement_to_skip: int = 4

    # Adaptive routing knobs. These make compute follow hard-tail rows and
    # historical saved-moves-per-second rather than fixed magic constants only.
    adaptive_enabled: bool = True
    light_percentile: float = 0.45
    aggressive_percentile: float = 0.76
    force_aggressive_percentile: float = 0.24
    min_roi_for_aggressive: float = 0.03
    max_regressions_for_aggressive: int = 1


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    try:
        if den <= 0:
            return default
        return float(num) / float(den)
    except Exception:
        return default


def difficulty_percentile(features: RowFeatures) -> float:
    """Return a stable row difficulty percentile.

    Lower row_rank means harder row in the existing search policy.  The return
    value is normalized to [0, 1], where values near 0 are the hardest rows.
    """
    total = max(int(features.total_rows or 0), 1)
    rank = max(int(features.row_rank), 0)
    return min(max(rank / total, 0.0), 1.0)


def historical_roi(features: RowFeatures) -> float:
    """Saved moves per second from previous exact-search profile data."""
    return _safe_ratio(
        float(features.historical_saved_moves),
        max(float(features.historical_runtime_s), 1e-9),
    )


def classify_row_legacy(features: RowFeatures, config: PolicyConfig) -> str:
    if features.current_best_len < config.light_min_path_len:
        return 'skip'
    if features.preopt_saved >= config.min_improvement_to_skip and features.current_best_len < config.aggressive_min_path_len:
        return 'skip'
    if features.row_rank < config.force_aggressive_top_n:
        return 'aggressive'
    if features.current_best_len >= config.aggressive_min_path_len:
        return 'aggressive'
    if features.current_best_len >= config.light_min_path_len:
        return 'light'
    return 'skip'


def classify_row(features: RowFeatures, config: PolicyConfig) -> str:
    """Classify a row into skip/light/aggressive search.

    The old implementation used only absolute thresholds: 560 / 700 / top-24.
    This version keeps those thresholds as a fallback but can also route by
    percentile, historical ROI, and regression risk from prior search profiles.
    """
    if not config.adaptive_enabled:
        return classify_row_legacy(features, config)

    difficulty = difficulty_percentile(features)
    roi = historical_roi(features)

    # Very short / already-good rows are skipped unless prior profile data
    # proves they are cheap to improve.
    if features.current_best_len < config.light_min_path_len and difficulty > config.light_percentile and roi <= 0.0:
        return 'skip'

    # If pre-optimization already saved enough and the row is not a hard tail,
    # avoid spending more compute unless historical ROI is positive.
    if (
        features.preopt_saved >= config.min_improvement_to_skip
        and features.current_best_len < config.aggressive_min_path_len
        and roi <= 0.0
    ):
        return 'skip'

    force_aggressive = difficulty <= config.force_aggressive_percentile
    hard_by_length = features.current_best_len >= config.aggressive_min_path_len
    hard_by_percentile = difficulty <= config.aggressive_percentile
    positive_roi = roi >= config.min_roi_for_aggressive
    low_regression_risk = features.regression_count <= config.max_regressions_for_aggressive

    if (force_aggressive or hard_by_length or (hard_by_percentile and positive_roi)) and low_regression_risk:
        return 'aggressive'
    if features.current_best_len >= config.light_min_path_len or positive_roi:
        return 'light'
    return 'skip'


def tier_params(tier: str, args: Any) -> Dict[str, Any]:
    if tier == 'aggressive':
        return {
            'beam_width': int(args.aggressive_beam_width),
            'max_steps': int(args.aggressive_max_steps),
            'history_depth': int(args.aggressive_history_depth),
            'mitm_depth': int(args.aggressive_mitm_depth),
            'time_budget_s': float(args.aggressive_time_budget_per_row),
            'window_lengths': [int(x) for x in str(args.aggressive_window_lengths).split(',') if str(x).strip()],
            'window_samples': int(args.aggressive_window_samples),
            'beam_mode': str(args.aggressive_beam_mode),
        }
    if tier == 'light':
        return {
            'beam_width': int(args.light_beam_width),
            'max_steps': int(args.light_max_steps),
            'history_depth': int(args.light_history_depth),
            'mitm_depth': int(args.light_mitm_depth),
            'time_budget_s': float(args.light_time_budget_per_row),
            'window_lengths': [int(x) for x in str(args.light_window_lengths).split(',') if str(x).strip()],
            'window_samples': int(args.light_window_samples),
            'beam_mode': str(args.light_beam_mode),
        }
    return {
        'beam_width': 0,
        'max_steps': 0,
        'history_depth': 0,
        'mitm_depth': 0,
        'time_budget_s': 0.0,
        'window_lengths': [],
        'window_samples': 0,
        'beam_mode': 'simple',
    }
