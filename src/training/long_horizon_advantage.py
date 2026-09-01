"""Project-local veRL advantage estimators for long-horizon ablations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence


def normalized_exponential_weights(length: int, alpha: float = 1.05) -> list[float]:
    """Return early-heavy weights with mean one, computed stably in log space."""

    if length <= 0:
        raise ValueError("length must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    log_alpha = math.log(alpha)
    logs = [(length - 1 - position) * log_alpha for position in range(length)]
    peak = max(logs)
    stable = [math.exp(value - peak) for value in logs]
    scale = length / sum(stable)
    return [value * scale for value in stable]


def lata_scale(length: int) -> float:
    if length <= 0:
        raise ValueError("length must be positive")
    return 1.0 / math.sqrt(length)


def _config_value(config: Any, section: str, key: str, default: float) -> float:
    if config is None:
        return default
    subsection = config.get(section) if hasattr(config, "get") else getattr(config, section, None)
    if subsection is None:
        return default
    value = subsection.get(key, default) if hasattr(subsection, "get") else getattr(subsection, key, default)
    return float(value)


def _group_normalized_scores(
    token_level_rewards: Any,
    index: Sequence[Any],
    epsilon: float,
    normalize_by_std: bool,
) -> Any:
    import torch

    scores = token_level_rewards.sum(dim=-1)
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for row, group_id in enumerate(index):
        grouped[group_id].append(scores[row])

    with torch.no_grad():
        for row, group_id in enumerate(index):
            group_scores = torch.stack(grouped[group_id])
            if group_scores.numel() == 1:
                mean = torch.zeros((), device=scores.device, dtype=scores.dtype)
                std = torch.ones((), device=scores.device, dtype=scores.dtype)
            else:
                mean = group_scores.mean()
                std = group_scores.std()
            scores[row] = scores[row] - mean
            if normalize_by_std:
                scores[row] = scores[row] / (std + epsilon)
    return scores


def _discount_tensor(response_mask: Any, alpha: float, epsilon: float) -> Any:
    import torch

    lengths = response_mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float64)
    positions = torch.arange(response_mask.shape[1], device=response_mask.device, dtype=torch.float64)
    log_weights = (lengths - 1 - positions.unsqueeze(0)) * math.log(alpha)
    masked_logs = log_weights.masked_fill(response_mask == 0, -float("inf"))
    peak = masked_logs.max(dim=1, keepdim=True).values
    stable = torch.exp(log_weights - peak) * response_mask.to(torch.float64)
    weight_sum = stable.sum(dim=1, keepdim=True).clamp(min=epsilon)
    return (stable * lengths / weight_sum).to(torch.float32)


def compute_grpo_turn_discounted_outcome_advantage(
    token_level_rewards: Any,
    response_mask: Any,
    index: Sequence[Any],
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    **_: Any,
) -> tuple[Any, Any]:
    """Apply standard group normalization and early-heavy token weights."""

    import torch

    alpha = _config_value(config, "turn_discount", "alpha", 1.05)
    with torch.no_grad():
        scores = _group_normalized_scores(token_level_rewards, index, epsilon, norm_adv_by_std_in_grpo)
        advantages = scores.unsqueeze(-1) * _discount_tensor(response_mask, alpha, epsilon)
        advantages = advantages * response_mask
    return advantages, advantages


def compute_grpo_lata_outcome_advantage(
    token_level_rewards: Any,
    response_mask: Any,
    index: Sequence[Any],
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    **_: Any,
) -> tuple[Any, Any]:
    """Turn-Discount followed by explicit inverse-square-root length scaling."""

    import torch

    alpha = _config_value(config, "turn_discount", "alpha", 1.05)
    with torch.no_grad():
        scores = _group_normalized_scores(token_level_rewards, index, epsilon, norm_adv_by_std_in_grpo)
        weights = _discount_tensor(response_mask, alpha, epsilon)
        lengths = response_mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32).sqrt()
        advantages = scores.unsqueeze(-1) * weights * response_mask / lengths
    return advantages, advantages


def register_with_verl() -> None:
    """Register both estimators without modifying the installed veRL package."""

    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("grpo_turn_discounted")(compute_grpo_turn_discounted_outcome_advantage)
    register_adv_est("grpo_lata")(compute_grpo_lata_outcome_advantage)
