#!/usr/bin/env python3
"""Analyze OrganSegCLIP representations from a checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_encoder_config
from organ_seg_clip.data.dataset import MerlinWholeStudyDataset, collate_whole_study_batch, load_samples_from_config
from organ_seg_clip.models import build_model, load_distilled_visual_encoder
from organ_seg_clip.models.interfaces.types import EncoderBatch
from organ_seg_clip.models.losses import OrganSegLossComposer
from organ_seg_clip.training.checkpointing import load_checkpoint
from organ_seg_clip.training.engine import _move_batch_to_device
from organ_seg_clip.utils.io import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--train-limit", type=int, default=512, help="Train embeddings for frozen probes.")
    parser.add_argument("--val-limit", type=int, default=256, help="Validation embeddings for retrieval/probes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=16, help="Print progress every N batches; 0 disables progress logs.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint_kind = _checkpoint_kind(args.checkpoint)
    if checkpoint_kind == "distilled_visual_encoder":
        model, payload = load_distilled_visual_encoder(args.checkpoint, map_location=device)
        model = model.to(device)
        model.eval()
        config = model.config
        loss_composer = None
    else:
        config = load_encoder_config(args.config)
        model = build_model(config).to(device)
        payload = load_checkpoint(args.checkpoint, model=model, optimizer=None, scaler=None, map_location=device, strict=False)
        model.eval()
        loss_composer = OrganSegLossComposer(config.loss)

    train_pack = _collect_split(
        config,
        model=model,
        loss_composer=loss_composer,
        split=config.data.train_split,
        limit=args.train_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        sample_seed=config.training.seed,
        log_every=max(0, int(args.log_every)),
    )
    val_pack = _collect_split(
        config,
        model=model,
        loss_composer=loss_composer,
        split=config.data.val_split,
        limit=args.val_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        sample_seed=config.training.seed + 1,
        log_every=max(0, int(args.log_every)),
    )

    analysis_start = time.time()
    result: dict[str, Any] = {
        "analysis_mode": checkpoint_kind,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": payload.get("epoch", payload.get("source_checkpoint_epoch")),
        "checkpoint_step": payload.get("step", payload.get("source_checkpoint_step")),
        "train_count": int(train_pack["study_count"]),
        "val_count": int(val_pack["study_count"]),
        "train_split_seconds": float(train_pack.get("split_seconds", 0.0)),
        "val_split_seconds": float(val_pack.get("split_seconds", 0.0)),
        "val_losses": val_pack["loss_metrics"],
        "organ_representation_analysis": _organ_representation_analysis(
            val_pack,
            organ_names=tuple(config.data.organ_names),
        ),
        "frozen_probes": _probe_metrics(
            train_pack,
            val_pack,
            device=device,
            steps=max(0, int(args.probe_steps)),
            lr=float(args.probe_lr),
        ),
    }
    if checkpoint_kind == "full_model":
        result["retrieval"] = _retrieval_metrics(val_pack, organ_names=tuple(config.data.organ_names))
        result["same_organ_analysis"] = _same_organ_analysis(val_pack, organ_names=tuple(config.data.organ_names))
    else:
        result["retrieval"] = {"available": False, "reason": "distilled visual encoder does not expose text embeddings"}
        result["same_organ_analysis"] = {"available": False, "reason": "distilled visual encoder does not expose text embeddings"}
    result["analysis_seconds"] = float(time.time() - analysis_start)
    result["zz_summary"] = _build_summary(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        dump_json(Path(args.output), result)


def _collect_split(
    config,
    *,
    model,
    loss_composer,
    split: str,
    limit: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sample_seed: int,
    log_every: int,
) -> dict[str, Any]:
    samples, _ = load_samples_from_config(config, split=split, sample_seed=sample_seed)
    samples = samples[: max(0, int(limit))]
    dataset = MerlinWholeStudyDataset(samples, config=config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_whole_study_batch)
    loss_sums: dict[str, float] = defaultdict(float)
    loss_counts: dict[str, float] = defaultdict(float)
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    raw_labels: list[str] = []
    organ_ids: list[int] = []
    study_ids: list[str] = []
    organ_names: list[str] = []
    study_count = 0
    total_batches = len(loader)
    total_samples = len(dataset)
    split_start = time.time()
    print(
        f"[analyze_encoder] split={split} samples={total_samples} batches={total_batches} "
        f"batch_size={batch_size} device={device}"
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch)
            mask = batch.organ_text_mask.detach().cpu().bool()
            batch_weight = float(batch.images.shape[0])
            if hasattr(outputs, "organ_image_embeddings"):
                loss_output, metric_output = loss_composer(outputs, batch)
                scalar_metrics = loss_output.to_dict() | metric_output
                for key, value in scalar_metrics.items():
                    loss_sums[key] += float(value) * batch_weight
                    loss_counts[key] += batch_weight
                image_emb = F.normalize(outputs.organ_image_embeddings.detach().cpu(), dim=-1)
                text_emb = F.normalize(outputs.organ_text_embeddings.detach().cpu(), dim=-1)
                chunks["organ_image"].append(image_emb[mask])
                chunks["organ_text"].append(text_emb[mask])
                report_emb = F.normalize(outputs.report_image_embeddings.detach().cpu(), dim=-1)
                chunks["report_image"].append(report_emb)
            else:
                organ_emb = F.normalize(outputs.organ_embeddings.detach().cpu(), dim=-1)
                chunks["organ_image"].append(organ_emb[mask])
                report_emb = F.normalize(outputs.report_embedding.detach().cpu(), dim=-1)
                chunks["report_image"].append(report_emb)
            chunks["organ_labels"].append(batch.organ_labels.detach().cpu()[mask])
            chunks["organ_label_mask"].append(batch.organ_label_mask.detach().cpu()[mask])
            chunks["lesion_organ_labels"].append(batch.lesion_organ_labels.detach().cpu()[mask])
            chunks["lesion_organ_mask"].append(batch.lesion_organ_mask.detach().cpu()[mask])
            chunks["lesion_global_labels"].append(batch.lesion_global_labels.detach().cpu())
            chunks["lesion_global_mask"].append(batch.lesion_global_mask.detach().cpu())

            for batch_index, sid in enumerate(batch.study_ids):
                study_count += 1
                for organ_index, text_ok in enumerate(mask[batch_index].tolist()):
                    if not text_ok:
                        continue
                    raw_labels.append(_normalize_label(batch.organ_raw_texts[batch_index][organ_index]))
                    organ_ids.append(int(organ_index))
                    study_ids.append(str(sid))
                    organ_names.append(str(config.data.organ_names[organ_index]))
            if log_every > 0 and (batch_index % log_every == 0 or batch_index == total_batches):
                elapsed = time.time() - split_start
                studies_per_second = study_count / max(elapsed, 1.0e-6)
                print(
                    f"[analyze_encoder] split={split} batch={batch_index}/{total_batches} "
                    f"studies={study_count}/{total_samples} elapsed_s={elapsed:.1f} "
                    f"studies_per_s={studies_per_second:.2f}"
                )
    packed = {key: torch.cat(value, dim=0) if value else torch.empty(0) for key, value in chunks.items()}
    packed.update(
        {
            "raw_labels": raw_labels,
            "organ_ids": torch.tensor(organ_ids, dtype=torch.long),
            "study_ids": study_ids,
            "organ_names": organ_names,
            "study_count": study_count,
            "split_seconds": float(time.time() - split_start),
            "loss_metrics": {key: loss_sums[key] / max(loss_counts[key], 1.0) for key in sorted(loss_sums)},
        }
    )
    return packed


def _checkpoint_kind(checkpoint_path: str | os.PathLike[str]) -> str:
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == "organsegclip_visual_encoder_v1":
        return "distilled_visual_encoder"
    return "full_model"


def _retrieval_metrics(pack: dict[str, Any], *, organ_names: tuple[str, ...]) -> dict[str, Any]:
    image = pack["organ_image"]
    text = pack["organ_text"]
    labels = pack["raw_labels"]
    organ_ids = pack["organ_ids"]
    if image.numel() == 0 or text.numel() == 0:
        return {"valid_organs": 0}
    sims = image @ text.T
    label_equal = torch.tensor([[labels[i] == labels[j] for j in range(len(labels))] for i in range(len(labels))], dtype=torch.bool)
    organ_equal = organ_ids[:, None] == organ_ids[None, :]
    positives = label_equal & organ_equal
    global_top = sims.argmax(dim=1)
    identity_top1 = (organ_ids[global_top] == organ_ids).float().mean().item()
    global_top1 = positives.gather(1, global_top[:, None]).squeeze(1).float().mean().item()

    same_organ_top1_hits = []
    same_organ_margins = []
    per_organ: dict[str, dict[str, float]] = {}
    for index in range(sims.shape[0]):
        candidates = organ_equal[index].clone()
        if candidates.sum() <= 1:
            continue
        local_scores = sims[index].masked_fill(~candidates, float("-inf"))
        winner = int(local_scores.argmax().item())
        same_organ_top1_hits.append(float(positives[index, winner].item()))
        pos_scores = sims[index][positives[index]]
        neg_scores = sims[index][candidates & ~positives[index]]
        if pos_scores.numel() > 0 and neg_scores.numel() > 0:
            same_organ_margins.append(float(pos_scores.mean().item() - neg_scores.mean().item()))
    for organ_index, organ_name in enumerate(organ_names):
        rows = (organ_ids == organ_index).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        hits = []
        margins = []
        for row in rows.tolist():
            candidates = organ_equal[row]
            if candidates.sum() <= 1:
                continue
            winner = int(sims[row].masked_fill(~candidates, float("-inf")).argmax().item())
            hits.append(float(positives[row, winner].item()))
            pos_scores = sims[row][positives[row]]
            neg_scores = sims[row][candidates & ~positives[row]]
            if pos_scores.numel() > 0 and neg_scores.numel() > 0:
                margins.append(float(pos_scores.mean().item() - neg_scores.mean().item()))
        per_organ[organ_name] = {
            "count": int(rows.numel()),
            "same_organ_finding_top1": _mean(hits),
            "same_organ_finding_margin": _mean(margins),
        }
    return {
        "valid_organs": int(image.shape[0]),
        "global_image_to_text_top1_same_organ_same_finding": float(global_top1),
        "global_image_to_text_top1_organ_identity": float(identity_top1),
        "same_organ_finding_top1": _mean(same_organ_top1_hits),
        "same_organ_finding_margin": _mean(same_organ_margins),
        "per_organ": per_organ,
    }


def _same_organ_analysis(pack: dict[str, Any], *, organ_names: tuple[str, ...]) -> dict[str, Any]:
    """Analyze within-organ image/text separation over collected embeddings.

    This is intentionally independent of the dataloader batch size used for the
    analyzer. The training loss only sees same-organ negatives when they appear
    in a comparison batch, but this diagnostic builds all same-organ pairs from
    the collected validation subset.
    """
    image = pack["organ_image"]
    text = pack["organ_text"]
    labels = pack["raw_labels"]
    organ_ids = pack["organ_ids"]
    if image.numel() == 0 or text.numel() == 0:
        return {"valid_organs": 0}

    sims = image @ text.T
    count = int(sims.shape[0])
    eye = torch.eye(count, dtype=torch.bool)
    label_equal = torch.tensor([[labels[i] == labels[j] for j in range(count)] for i in range(count)], dtype=torch.bool)
    organ_equal = organ_ids[:, None] == organ_ids[None, :]

    same_organ_candidates = organ_equal & ~eye
    same_organ_positive = same_organ_candidates & label_equal
    same_organ_negative = same_organ_candidates & ~label_equal
    cross_organ_negative = ~organ_equal

    top1_hits: list[float] = []
    top5_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    baseline_rates: list[float] = []
    row_margins: list[float] = []
    for row in range(count):
        candidates = same_organ_candidates[row]
        positives = same_organ_positive[row]
        if candidates.sum() <= 0 or positives.sum() <= 0:
            continue
        candidate_scores = sims[row].masked_fill(~candidates, float("-inf"))
        order = torch.argsort(candidate_scores, descending=True)
        positive_order = positives[order]
        top1_hits.append(float(positive_order[0].item()))
        top5_hits.append(float(positive_order[: min(5, int(candidates.sum().item()))].any().item()))
        first_positive = (positive_order.nonzero(as_tuple=False).flatten()[0].item() + 1) if positive_order.any() else 0
        reciprocal_ranks.append(float(1.0 / first_positive) if first_positive else 0.0)
        baseline_rates.append(float(positives.sum().item() / max(candidates.sum().item(), 1)))
        negatives = same_organ_negative[row]
        if negatives.any():
            row_margins.append(float(sims[row][positives].mean().item() - sims[row][negatives].mean().item()))

    per_organ: dict[str, dict[str, float]] = {}
    for organ_index, organ_name in enumerate(organ_names):
        rows = (organ_ids == organ_index).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        organ_hits: list[float] = []
        organ_top5: list[float] = []
        organ_mrr: list[float] = []
        organ_baselines: list[float] = []
        organ_margins: list[float] = []
        label_counts: dict[str, int] = defaultdict(int)
        for row in rows.tolist():
            label_counts[labels[row]] += 1
            candidates = same_organ_candidates[row]
            positives = same_organ_positive[row]
            if candidates.sum() <= 0 or positives.sum() <= 0:
                continue
            candidate_scores = sims[row].masked_fill(~candidates, float("-inf"))
            order = torch.argsort(candidate_scores, descending=True)
            positive_order = positives[order]
            organ_hits.append(float(positive_order[0].item()))
            organ_top5.append(float(positive_order[: min(5, int(candidates.sum().item()))].any().item()))
            first_positive = (positive_order.nonzero(as_tuple=False).flatten()[0].item() + 1) if positive_order.any() else 0
            organ_mrr.append(float(1.0 / first_positive) if first_positive else 0.0)
            organ_baselines.append(float(positives.sum().item() / max(candidates.sum().item(), 1)))
            negatives = same_organ_negative[row]
            if negatives.any():
                organ_margins.append(float(sims[row][positives].mean().item() - sims[row][negatives].mean().item()))
        largest_label = max(label_counts.values()) if label_counts else 0
        per_organ[organ_name] = {
            "count": int(rows.numel()),
            "finding_class_count": int(len(label_counts)),
            "largest_finding_fraction": float(largest_label / max(int(rows.numel()), 1)),
            "top1_excluding_self": _mean(organ_hits),
            "top5_excluding_self": _mean(organ_top5),
            "mrr_excluding_self": _mean(organ_mrr),
            "baseline_positive_rate": _mean(organ_baselines),
            "margin_vs_same_organ_negative": _mean(organ_margins),
        }

    pos_mean = _masked_mean(sims, same_organ_positive)
    same_neg_mean = _masked_mean(sims, same_organ_negative)
    cross_neg_mean = _masked_mean(sims, cross_organ_negative)
    return {
        "valid_organs": count,
        "same_organ_positive_pairs_excluding_self": int(same_organ_positive.sum().item()),
        "same_organ_negative_pairs": int(same_organ_negative.sum().item()),
        "cross_organ_negative_pairs": int(cross_organ_negative.sum().item()),
        "positive_cosine_mean_excluding_self": pos_mean,
        "same_organ_negative_cosine_mean": same_neg_mean,
        "cross_organ_negative_cosine_mean": cross_neg_mean,
        "margin_vs_same_organ_negative": pos_mean - same_neg_mean,
        "margin_vs_cross_organ_negative": pos_mean - cross_neg_mean,
        "top1_excluding_self": _mean(top1_hits),
        "top5_excluding_self": _mean(top5_hits),
        "mrr_excluding_self": _mean(reciprocal_ranks),
        "baseline_positive_rate": _mean(baseline_rates),
        "top1_lift_over_baseline": _mean(top1_hits) - _mean(baseline_rates),
        "per_organ": per_organ,
    }


def _organ_representation_analysis(pack: dict[str, Any], *, organ_names: tuple[str, ...]) -> dict[str, Any]:
    """Analyze image-only organ embeddings as geometric objects.

    This section is intentionally decoder-oriented: it only uses the image ->
    embedding pathway (`organ_image`) and measures whether organ embeddings vary
    meaningfully across studies and labels.
    """
    organ_embeddings = pack["organ_image"]
    organ_ids = pack["organ_ids"]
    diagnostic_labels = pack["organ_labels"].reshape(-1)
    diagnostic_mask = pack["organ_label_mask"].reshape(-1).bool()
    lesion_labels = pack["lesion_organ_labels"].reshape(-1)
    lesion_mask = pack["lesion_organ_mask"].reshape(-1).bool()
    if organ_embeddings.numel() == 0 or organ_ids.numel() == 0:
        return {"valid_organs": 0}

    embedding_count = int(organ_embeddings.shape[0])
    cosine_similarity = organ_embeddings @ organ_embeddings.T
    euclidean_distance = torch.cdist(organ_embeddings, organ_embeddings, p=2)
    same_organ = organ_ids[:, None] == organ_ids[None, :]
    different_organ = ~same_organ
    non_self = ~torch.eye(embedding_count, dtype=torch.bool)

    centroid_vectors: dict[str, torch.Tensor] = {}
    per_organ: dict[str, dict[str, Any]] = {}
    for organ_index, organ_name in enumerate(organ_names):
        rows = (organ_ids == organ_index).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        organ_vectors = organ_embeddings[rows]
        centroid_vectors[organ_name] = F.normalize(organ_vectors.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        same_rows = torch.zeros((embedding_count,), dtype=torch.bool)
        same_rows[rows] = True
        pair_mask = same_rows[:, None] & same_rows[None, :] & non_self
        organ_result: dict[str, Any] = {
            "count": int(rows.numel()),
            "cosine_mean_all_pairs": _masked_mean(cosine_similarity, pair_mask),
            "cosine_std_all_pairs": _masked_std(cosine_similarity, pair_mask),
            "euclidean_mean_all_pairs": _masked_mean(euclidean_distance, pair_mask),
            "euclidean_std_all_pairs": _masked_std(euclidean_distance, pair_mask),
            "diagnostic_label_analysis": _binary_group_geometry(
                organ_vectors,
                diagnostic_labels[rows],
                diagnostic_mask[rows],
            ),
            "lesion_label_analysis": _binary_group_geometry(
                organ_vectors,
                lesion_labels[rows],
                lesion_mask[rows],
            ),
        }
        per_organ[organ_name] = organ_result

    centroid_names = [name for name in organ_names if name in centroid_vectors]
    centroid_stack = torch.stack([centroid_vectors[name] for name in centroid_names], dim=0) if centroid_names else torch.empty(0)
    centroid_cosine = centroid_stack @ centroid_stack.T if centroid_stack.numel() else torch.empty(0)
    centroid_euclidean = torch.cdist(centroid_stack, centroid_stack, p=2) if centroid_stack.numel() else torch.empty(0)

    return {
        "valid_organs": embedding_count,
        "same_organ_cosine_mean": _masked_mean(cosine_similarity, same_organ & non_self),
        "same_organ_cosine_std": _masked_std(cosine_similarity, same_organ & non_self),
        "different_organ_cosine_mean": _masked_mean(cosine_similarity, different_organ),
        "different_organ_cosine_std": _masked_std(cosine_similarity, different_organ),
        "same_organ_euclidean_mean": _masked_mean(euclidean_distance, same_organ & non_self),
        "same_organ_euclidean_std": _masked_std(euclidean_distance, same_organ & non_self),
        "different_organ_euclidean_mean": _masked_mean(euclidean_distance, different_organ),
        "different_organ_euclidean_std": _masked_std(euclidean_distance, different_organ),
        "same_minus_different_cosine_margin": _masked_mean(cosine_similarity, same_organ & non_self)
        - _masked_mean(cosine_similarity, different_organ),
        "different_minus_same_euclidean_margin": _masked_mean(euclidean_distance, different_organ)
        - _masked_mean(euclidean_distance, same_organ & non_self),
        "organ_centroid_cosine_similarity": _named_matrix(centroid_names, centroid_cosine),
        "organ_centroid_cosine_distance": _named_matrix(centroid_names, 1.0 - centroid_cosine),
        "organ_centroid_euclidean_distance": _named_matrix(centroid_names, centroid_euclidean),
        "per_organ": per_organ,
    }


def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
    frozen_probes = result.get("frozen_probes", {})
    organ_analysis = result.get("organ_representation_analysis", {})
    same_organ = result.get("same_organ_analysis", {})
    val_losses = result.get("val_losses", {})

    summary: dict[str, Any] = {
        "analysis_mode": result.get("analysis_mode"),
        "checkpoint_epoch": result.get("checkpoint_epoch"),
        "checkpoint_step": result.get("checkpoint_step"),
        "train_count": result.get("train_count"),
        "val_count": result.get("val_count"),
        "analysis_seconds": result.get("analysis_seconds", 0.0),
        "split_seconds": {
            "train": result.get("train_split_seconds"),
            "val": result.get("val_split_seconds"),
        },
        "key_probes": {
            "organ_diagnostic_balanced_accuracy": _nested_float(frozen_probes, "organ_diagnostic", "balanced_accuracy"),
            "organ_lesion_balanced_accuracy": _nested_float(frozen_probes, "organ_lesion", "balanced_accuracy"),
            "global_lesion_from_report_image_balanced_accuracy": _nested_float(
                frozen_probes,
                "global_lesion_from_report_image",
                "balanced_accuracy",
            ),
        },
        "organ_geometry": {
            "same_organ_cosine_mean": _float_or_none(organ_analysis.get("same_organ_cosine_mean")),
            "different_organ_cosine_mean": _float_or_none(organ_analysis.get("different_organ_cosine_mean")),
            "same_minus_different_cosine_margin": _float_or_none(organ_analysis.get("same_minus_different_cosine_margin")),
            "same_organ_euclidean_mean": _float_or_none(organ_analysis.get("same_organ_euclidean_mean")),
            "different_organ_euclidean_mean": _float_or_none(organ_analysis.get("different_organ_euclidean_mean")),
            "different_minus_same_euclidean_margin": _float_or_none(organ_analysis.get("different_minus_same_euclidean_margin")),
        },
        "same_organ_retrieval": {
            "available": bool(same_organ.get("available", True)),
            "top1_excluding_self": _float_or_none(same_organ.get("top1_excluding_self")),
            "top5_excluding_self": _float_or_none(same_organ.get("top5_excluding_self")),
            "mrr_excluding_self": _float_or_none(same_organ.get("mrr_excluding_self")),
            "baseline_positive_rate": _float_or_none(same_organ.get("baseline_positive_rate")),
            "top1_lift_over_baseline": _float_or_none(same_organ.get("top1_lift_over_baseline")),
            "margin_vs_same_organ_negative": _float_or_none(same_organ.get("margin_vs_same_organ_negative")),
            "margin_vs_cross_organ_negative": _float_or_none(same_organ.get("margin_vs_cross_organ_negative")),
        },
        "val_loss_highlights": {
            "total_loss": _float_or_none(val_losses.get("total_loss")),
            "organ_alignment_loss": _float_or_none(val_losses.get("organ_alignment_loss")),
            "report_alignment_loss": _float_or_none(val_losses.get("report_alignment_loss")),
            "segmentation_dice": _float_or_none(val_losses.get("segmentation_dice")),
            "organ_attention_accuracy": _float_or_none(val_losses.get("organ_attention_accuracy")),
            "patch_organ_presence_accuracy": _float_or_none(val_losses.get("patch_organ_presence_accuracy")),
        },
    }

    if isinstance(same_organ.get("per_organ"), dict):
        summary["same_organ_retrieval"]["best_organs_by_top1"] = _top_per_organs(
            same_organ["per_organ"],
            metric="top1_excluding_self",
            largest=True,
            limit=5,
        )
        summary["same_organ_retrieval"]["worst_organs_by_top1"] = _top_per_organs(
            same_organ["per_organ"],
            metric="top1_excluding_self",
            largest=False,
            limit=5,
        )
        summary["same_organ_retrieval"]["best_organs_by_margin"] = _top_per_organs(
            same_organ["per_organ"],
            metric="margin_vs_same_organ_negative",
            largest=True,
            limit=5,
        )
        summary["same_organ_retrieval"]["worst_organs_by_margin"] = _top_per_organs(
            same_organ["per_organ"],
            metric="margin_vs_same_organ_negative",
            largest=False,
            limit=5,
        )

    if isinstance(organ_analysis.get("per_organ"), dict):
        summary["organ_geometry"]["best_organs_by_diagnostic_centroid_distance"] = _top_per_organs(
            organ_analysis["per_organ"],
            metric="diagnostic_label_analysis.positive_negative_centroid_euclidean_distance",
            largest=True,
            limit=5,
        )
        summary["organ_geometry"]["best_organs_by_lesion_centroid_distance"] = _top_per_organs(
            organ_analysis["per_organ"],
            metric="lesion_label_analysis.positive_negative_centroid_euclidean_distance",
            largest=True,
            limit=5,
        )

    return summary


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or not mask.any():
        return 0.0
    return float(values[mask].mean().item())


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0 or not mask.any():
        return 0.0
    masked = values[mask]
    if masked.numel() <= 1:
        return 0.0
    return float(masked.std(unbiased=False).item())


def _named_matrix(names: list[str], values: torch.Tensor) -> dict[str, dict[str, float]]:
    if values.numel() == 0:
        return {}
    return {
        row_name: {
            col_name: float(values[row_index, col_index].item())
            for col_index, col_name in enumerate(names)
        }
        for row_index, row_name in enumerate(names)
    }


def _binary_group_geometry(vectors: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    mask = mask.bool().reshape(-1)
    if vectors.numel() == 0 or mask.sum() == 0:
        return {"valid_count": int(mask.sum().item())}
    valid_vectors = F.normalize(vectors[mask].float(), dim=-1)
    valid_labels = labels[mask].reshape(-1).float()
    positives = valid_labels >= 0.5
    negatives = ~positives
    result: dict[str, Any] = {
        "valid_count": int(mask.sum().item()),
        "positive_count": int(positives.sum().item()),
        "negative_count": int(negatives.sum().item()),
    }
    if positives.sum() == 0 or negatives.sum() == 0:
        return result

    cosine_similarity = valid_vectors @ valid_vectors.T
    euclidean_distance = torch.cdist(valid_vectors, valid_vectors, p=2)
    non_self = ~torch.eye(int(valid_vectors.shape[0]), dtype=torch.bool)
    same_label = ((positives[:, None] & positives[None, :]) | (negatives[:, None] & negatives[None, :])) & non_self
    different_label = ((positives[:, None] & negatives[None, :]) | (negatives[:, None] & positives[None, :]))

    pos_centroid = F.normalize(valid_vectors[positives].mean(dim=0, keepdim=True), dim=-1)
    neg_centroid = F.normalize(valid_vectors[negatives].mean(dim=0, keepdim=True), dim=-1)
    centroid_cosine = float((pos_centroid @ neg_centroid.T).item())
    result.update(
        {
            "same_label_cosine_mean": _masked_mean(cosine_similarity, same_label),
            "different_label_cosine_mean": _masked_mean(cosine_similarity, different_label),
            "same_minus_different_cosine_margin": _masked_mean(cosine_similarity, same_label)
            - _masked_mean(cosine_similarity, different_label),
            "same_label_euclidean_mean": _masked_mean(euclidean_distance, same_label),
            "different_label_euclidean_mean": _masked_mean(euclidean_distance, different_label),
            "different_minus_same_euclidean_margin": _masked_mean(euclidean_distance, different_label)
            - _masked_mean(euclidean_distance, same_label),
            "positive_negative_centroid_cosine_similarity": centroid_cosine,
            "positive_negative_centroid_cosine_distance": 1.0 - centroid_cosine,
            "positive_negative_centroid_euclidean_distance": float(torch.cdist(pos_centroid, neg_centroid, p=2).item()),
        }
    )
    return result


def _probe_metrics(train: dict[str, Any], val: dict[str, Any], *, device: torch.device, steps: int, lr: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["organ_diagnostic"] = _binary_probe(
        train["organ_image"], train["organ_labels"], train["organ_label_mask"],
        val["organ_image"], val["organ_labels"], val["organ_label_mask"],
        device=device, steps=steps, lr=lr,
    )
    metrics["organ_lesion"] = _binary_probe(
        train["organ_image"], train["lesion_organ_labels"], train["lesion_organ_mask"],
        val["organ_image"], val["lesion_organ_labels"], val["lesion_organ_mask"],
        device=device, steps=steps, lr=lr,
    )
    metrics["global_lesion_from_report_image"] = _binary_probe(
        train["report_image"], train["lesion_global_labels"], train["lesion_global_mask"],
        val["report_image"], val["lesion_global_labels"], val["lesion_global_mask"],
        device=device, steps=steps, lr=lr,
    )
    return metrics


def _binary_probe(train_x, train_y, train_mask, val_x, val_y, val_mask, *, device: torch.device, steps: int, lr: float) -> dict[str, float]:
    train_mask = train_mask.bool().reshape(-1)
    val_mask = val_mask.bool().reshape(-1)
    if train_mask.sum() < 8 or val_mask.sum() < 1 or train_x.numel() == 0 or val_x.numel() == 0:
        return {"valid_train": int(train_mask.sum()), "valid_val": int(val_mask.sum()), "accuracy": 0.0, "balanced_accuracy": 0.0}
    x = train_x[train_mask].float().to(device)
    y = train_y.reshape(-1)[train_mask].float().to(device)
    vx = val_x[val_mask].float().to(device)
    vy = val_y.reshape(-1)[val_mask].float().to(device)
    head = torch.nn.Linear(x.shape[-1], 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = head(x).squeeze(-1)
        pos = y >= 0.5
        neg = ~pos
        losses = []
        if pos.any():
            losses.append(F.binary_cross_entropy_with_logits(logits[pos], y[pos]))
        if neg.any():
            losses.append(F.binary_cross_entropy_with_logits(logits[neg], y[neg]))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (head(vx).squeeze(-1).sigmoid() >= 0.5).float()
        correct = pred == vy
        pos = vy >= 0.5
        neg = ~pos
        pos_acc = float(correct[pos].float().mean().item()) if pos.any() else 0.0
        neg_acc = float(correct[neg].float().mean().item()) if neg.any() else 0.0
        return {
            "valid_train": int(train_mask.sum().item()),
            "valid_val": int(val_mask.sum().item()),
            "positive_rate_val": float(vy.mean().item()),
            "accuracy": float(correct.float().mean().item()),
            "positive_accuracy": pos_acc,
            "negative_accuracy": neg_acc,
            "balanced_accuracy": 0.5 * (pos_acc + neg_acc) if pos.any() and neg.any() else float(correct.float().mean().item()),
        }


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_float(mapping: dict[str, Any], *keys: str) -> float | None:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return _float_or_none(current)


def _lookup_metric(mapping: dict[str, Any], metric: str) -> float | None:
    current: Any = mapping
    for part in metric.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return _float_or_none(current)


def _top_per_organs(
    per_organ: dict[str, Any],
    *,
    metric: str,
    largest: bool,
    limit: int,
) -> list[dict[str, float | str]]:
    rows: list[tuple[str, float]] = []
    for organ_name, organ_metrics in per_organ.items():
        value = _lookup_metric(organ_metrics, metric)
        if value is None:
            continue
        rows.append((organ_name, value))
    rows.sort(key=lambda item: item[1], reverse=largest)
    return [{"organ": organ, "value": value} for organ, value in rows[: max(0, int(limit))]]


if __name__ == "__main__":
    main()
