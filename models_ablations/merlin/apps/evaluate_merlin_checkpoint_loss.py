#!/usr/bin/env python
"""Evaluate a completed Merlin ablation checkpoint on the configured val split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from merlin_ablation.config import load_config
from merlin_ablation.data import build_datasets
from merlin_ablation.losses import AuxiliaryDiagnosticLosses
from merlin_ablation.modeling import MerlinReportTrainingWrapper
from merlin_ablation.train import _evaluate, _make_loader, _prepare_imports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Merlin ablation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt from a completed run.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <config output_dir>/real_val_evaluation.json.",
    )
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        help="Do not preflight cached image embeddings before loading the checkpoint.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else config.output_dir / "real_val_evaluation.json"

    _prepare_imports(config)
    start = time.time()
    datasets = build_datasets(config)
    if config.model.image_embedding_mode == "cached" and not args.skip_cache_check:
        _assert_cached_embeddings_exist(datasets.val_records)

    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    model = MerlinReportTrainingWrapper(config).to(device)
    family_count = 0 if datasets.semantic_lookup is None else len(datasets.semantic_lookup.spec.family_vocab)
    subtype_count = 0 if datasets.semantic_lookup is None else len(datasets.semantic_lookup.spec.subtype_vocab)
    aux_losses = AuxiliaryDiagnosticLosses(
        config.losses,
        hidden_size=model.hidden_size,
        family_count=family_count,
        subtype_count=subtype_count,
    ).to(device)

    print(f"[merlin-eval] loading checkpoint={checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    aux_losses.load_state_dict(checkpoint["aux_losses"], strict=True)
    del checkpoint

    val_loader = _make_loader(config, datasets.val_records, split_name="val", shuffle=False)
    print(
        "[merlin-eval] "
        f"run_id={config.train.run_id} val_split={config.data.val_split} "
        f"records={len(datasets.val_records)} batch_size={config.train.batch_size}",
        flush=True,
    )
    metrics = _evaluate(model, aux_losses, val_loader, config, device)
    elapsed = time.time() - start
    payload = {
        "format": "merlin_ablation_checkpoint_eval_v1",
        "config_path": str(Path(args.config).expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "elapsed_seconds": elapsed,
        "dataset_summary": datasets.summary,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[merlin-eval] metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
    print(f"[merlin-eval] wrote {output_path}", flush=True)


def _assert_cached_embeddings_exist(records: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    seen: set[str] = set()
    for record in records:
        path = str(record["image_embedding"])
        if path in seen:
            continue
        seen.add(path)
        if not Path(path).is_file():
            missing.append(path)
            if len(missing) >= 10:
                break
    if missing:
        examples = "\n".join(missing)
        raise FileNotFoundError(
            "Missing cached Merlin image embeddings for validation. "
            "Build the cache first, then rerun evaluation. Examples:\n"
            f"{examples}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
