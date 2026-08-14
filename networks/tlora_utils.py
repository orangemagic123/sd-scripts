from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor


def validate_tlora_schedule(max_timestep: float, min_rank: int, alpha: float) -> None:
    """Validate the scalar parameters used by the timestep/rank schedule."""

    if not math.isfinite(float(max_timestep)) or float(max_timestep) <= 0:
        raise ValueError(f"tlora_max_timestep must be finite and greater than zero, got {max_timestep}")
    if int(min_rank) < 1:
        raise ValueError(f"tlora_min_rank must be at least one, got {min_rank}")
    if not math.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError(f"tlora_alpha_rank_scale must be finite and non-negative, got {alpha}")


def timestep_progress(
    timestep: Union[Tensor, float, int],
    max_timestep: float,
    alpha: float = 1.0,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return the normalized active-rank progress for every batch item.

    A value of zero selects ``min_rank`` and a value of one selects the full
    module rank. Out-of-range scheduler values are clamped before applying the
    exponent so fractional exponents cannot produce NaNs.
    """

    validate_tlora_schedule(max_timestep, 1, alpha)
    values = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
    progress = ((float(max_timestep) - values) / float(max_timestep)).clamp_(0.0, 1.0)
    return progress.pow(float(alpha))


def build_rank_mask(
    progress: Tensor,
    rank: int,
    min_rank: int = 1,
    *,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Build a per-sample prefix mask for a module's actual rank.

    ``progress`` may contain one scheduler value for a whole batch or one value
    per sample. A repeated batch, such as classifier-free guidance, is handled
    by repeating each source rank the required number of times.
    """

    rank = int(rank)
    if rank < 1:
        raise ValueError(f"rank must be at least one, got {rank}")

    effective_min_rank = min(max(int(min_rank), 1), rank)
    values = progress.to(device=device, dtype=torch.float32).reshape(-1)

    if batch_size is not None:
        batch_size = int(batch_size)
        if values.numel() == 1 and batch_size != 1:
            values = values.expand(batch_size)
        elif values.numel() != batch_size:
            if batch_size % values.numel() != 0:
                raise ValueError(
                    f"Cannot align {values.numel()} timestep values with activation batch size {batch_size}"
                )
            values = values.repeat(batch_size // values.numel())

    active_ranks = torch.floor(values * (rank - effective_min_rank)).to(torch.long)
    active_ranks = active_ranks.add_(effective_min_rank).clamp_(effective_min_rank, rank)
    rank_ids = torch.arange(rank, device=values.device)
    mask = rank_ids.unsqueeze(0) < active_ranks.unsqueeze(1)
    return mask.to(dtype=dtype if dtype is not None else torch.float32)
