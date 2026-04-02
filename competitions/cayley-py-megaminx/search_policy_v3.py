from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RowFeatures:
    baseline_len: int
    current_best_len: int
    preopt_saved: int
    row_rank: int


@dataclass(frozen=True)
class PolicyConfig:
    light_min_path_len: int = 560
    aggressive_min_path_len: int = 700
    force_aggressive_top_n: int = 24
    min_improvement_to_skip: int = 4


def classify_row(features: RowFeatures, config: PolicyConfig) -> str:
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
