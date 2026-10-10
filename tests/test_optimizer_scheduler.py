from types import SimpleNamespace

import pytest
import torch
from diffusers.optimization import get_piecewise_constant_schedule

from library.optimizer import get_scheduler_fix


def _args(warmup_steps=0, step_rules="1:10,0.5"):
    return SimpleNamespace(
        optimizer_type="AdamW",
        lr_scheduler="piecewise_constant",
        lr_scheduler_type="",
        lr_scheduler_args=[f"step_rules={step_rules!r}"],
        max_train_steps=20,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
    )


def _optimizer():
    return torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 2e-5},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 4e-5},
        ]
    )


def _advance(optimizer, scheduler, steps=1):
    for _ in range(steps):
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize("warmup_steps,num_processes", [(4, 1), (0.2, 1), (0.1, 2)])
def test_piecewise_warmup_keeps_absolute_boundaries_and_group_lrs(warmup_steps, num_processes):
    optimizer = _optimizer()
    scheduler = get_scheduler_fix(_args(warmup_steps, "1:10,0.5:15,0.25"), optimizer, num_processes)

    assert scheduler.get_last_lr() == pytest.approx([0, 0])
    _advance(optimizer, scheduler, 2)
    assert scheduler.get_last_lr() == pytest.approx([1e-5, 2e-5])
    _advance(optimizer, scheduler, 2)
    assert scheduler.get_last_lr() == pytest.approx([2e-5, 4e-5])
    _advance(optimizer, scheduler, 5)
    assert scheduler.get_last_lr() == pytest.approx([2e-5, 4e-5])
    _advance(optimizer, scheduler)
    assert scheduler.get_last_lr() == pytest.approx([1e-5, 2e-5])
    _advance(optimizer, scheduler, 4)
    assert scheduler.get_last_lr() == pytest.approx([1e-5, 2e-5])
    _advance(optimizer, scheduler)
    assert scheduler.get_last_lr() == pytest.approx([5e-6, 1e-5])


def test_warmup_respects_non_unit_initial_multiplier():
    optimizer = _optimizer()
    scheduler = get_scheduler_fix(_args(4, "0.5:10,0.25"), optimizer, 1)
    _advance(optimizer, scheduler, 2)
    assert scheduler.get_last_lr() == pytest.approx([5e-6, 1e-5])
    _advance(optimizer, scheduler, 2)
    assert scheduler.get_last_lr() == pytest.approx([1e-5, 2e-5])


@pytest.mark.parametrize("step_rules", ["1:10,0.5", "0.5:5,0.25:10,0.1", "0.75"])
def test_zero_warmup_matches_original_diffusers_schedule(step_rules):
    optimizer = _optimizer()
    scheduler = get_scheduler_fix(_args(0, step_rules), optimizer, 1)
    original_optimizer = _optimizer()
    original_scheduler = get_piecewise_constant_schedule(original_optimizer, step_rules)

    for _ in range(21):
        assert scheduler.get_last_lr() == pytest.approx(original_scheduler.get_last_lr())
        _advance(optimizer, scheduler)
        _advance(original_optimizer, original_scheduler)


@pytest.mark.parametrize("resume_step", [2, 8, 12])
def test_piecewise_warmup_state_resume_matches_uninterrupted_training(resume_step):
    args = _args(4)
    optimizer = _optimizer()
    scheduler = get_scheduler_fix(args, optimizer, 1)
    _advance(optimizer, scheduler, resume_step)

    restored_optimizer = _optimizer()
    restored_scheduler = get_scheduler_fix(args, restored_optimizer, 1)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())

    for _ in range(20 - resume_step):
        assert restored_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
        assert [group["lr"] for group in restored_optimizer.param_groups] == pytest.approx(scheduler.get_last_lr())
        _advance(optimizer, scheduler)
        _advance(restored_optimizer, restored_scheduler)
