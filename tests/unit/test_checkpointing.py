from __future__ import annotations

from pathlib import Path

import torch

from organ_seg_clip.training.checkpointing import load_checkpoint, load_pretrained_submodule, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = None
    target = tmp_path / "checkpoint.pt"
    save_checkpoint(target, model=model, optimizer=optimizer, scaler=scaler, epoch=3, config={"x": 1}, metrics={"loss": 0.5})
    restored = torch.nn.Linear(4, 3)
    payload = load_checkpoint(target, model=restored, optimizer=None, scaler=None)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert payload["epoch"] == 3


def test_checkpoint_round_trip_restores_scheduler_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / float(step + 1))
    optimizer.step()
    scheduler.step()
    target = tmp_path / "checkpoint_with_scheduler.pt"

    save_checkpoint(
        target,
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        epoch=2,
        config={"x": 1},
        metrics={"loss": 0.5},
    )

    restored = torch.nn.Linear(4, 3)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lr_lambda=lambda step: 1.0 / float(step + 1),
    )
    payload = load_checkpoint(
        target,
        model=restored,
        optimizer=restored_optimizer,
        scaler=None,
        scheduler=restored_scheduler,
    )

    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert payload["scheduler_state"] is not None
    assert payload["rng_state"] is not None


def test_pretrained_submodule_load_strips_prefix(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    payload = {
        "state_dict": {
            "patch_encoder.weight": model.weight.detach().clone(),
            "patch_encoder.bias": model.bias.detach().clone(),
        }
    }
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(payload, checkpoint)
    target = torch.nn.Linear(4, 3)
    result = load_pretrained_submodule(checkpoint, model=target, candidate_prefixes=("patch_encoder",))
    assert result["matched_keys"] == 2
    assert torch.equal(model.weight, target.weight)
    assert torch.equal(model.bias, target.bias)
