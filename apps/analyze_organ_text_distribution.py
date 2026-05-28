#!/usr/bin/env python3
"""Analyze per-organ raw finding text distribution in the Merlin metadata."""

import argparse
from collections import Counter, defaultdict
import difflib
import json
import math
import re
from pathlib import Path


DEFAULT_ORGANS = (
    "Spleen",
    "Kidneys",
    "Gallbladder",
    "Liver",
    "Stomach",
    "Pancreas",
    "Adrenal glands",
    "Small bowel",
    "Colon",
    "Urinary bladder",
    "Prostate",
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "no",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/net/storage/pr3/plgrid/plggjmiag/Merlin_converted")
    parser.add_argument("--output", default="")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--near-duplicate-top-n", type=int, default=160)
    parser.add_argument("--near-duplicate-ratio", type=float, default=0.84)
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.72)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    layout_root = resolve_layout_root(dataset_root)
    metadata_path = layout_root / "train" / "combined.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata_records = json.load(handle)

    metadata_by_id = {}
    for record in metadata_records:
        if isinstance(record, dict) and isinstance(record.get("study_id"), str):
            metadata_by_id[str(record["study_id"])] = record

    studies = collect_usable_studies(layout_root, metadata_by_id)

    result = {
        "dataset_root": str(dataset_root),
        "layout_root": str(layout_root),
        "usable_study_count": len(studies),
        "usable_studies_per_split": dict(Counter(study["split"] for study in studies)),
        "organ_text_distribution": {},
        "cross_organ_overlap": {},
        "global_summary": {},
    }

    global_text_to_organs = defaultdict(Counter)
    global_all_text_counter = Counter()

    for organ_name in DEFAULT_ORGANS:
        exact_counter = Counter()
        split_counter = defaultdict(Counter)
        raw_examples = {}
        for study in studies:
            finding_value = study["findings"].get(organ_name)
            if organ_name == "Kidneys" and isinstance(finding_value, dict):
                continue
            if not isinstance(finding_value, str):
                continue
            normalized = normalize_text(finding_value)
            if not normalized:
                continue
            exact_counter[normalized] += 1
            split_counter[study["split"]][normalized] += 1
            if normalized not in raw_examples:
                raw_examples[normalized] = finding_value
            global_text_to_organs[normalized][organ_name] += 1
            global_all_text_counter[normalized] += 1

        result["organ_text_distribution"][organ_name] = organ_summary(
            organ_name=organ_name,
            exact_counter=exact_counter,
            split_counter=split_counter,
            raw_examples=raw_examples,
            top_k=max(1, int(args.top_k)),
            near_duplicate_top_n=max(2, int(args.near_duplicate_top_n)),
            near_duplicate_ratio=float(args.near_duplicate_ratio),
            near_duplicate_jaccard=float(args.near_duplicate_jaccard),
        )

    overlap_rows = []
    for text, organ_counter in global_text_to_organs.items():
        if len(organ_counter) <= 1:
            continue
        overlap_rows.append(
            {
                "text": text,
                "organ_count": len(organ_counter),
                "total_count": int(sum(organ_counter.values())),
                "per_organ_count": dict(sorted(organ_counter.items())),
            }
        )
    overlap_rows.sort(key=lambda row: (-row["organ_count"], -row["total_count"], row["text"]))

    result["cross_organ_overlap"] = {
        "shared_text_count": len(overlap_rows),
        "shared_text_top": overlap_rows[: max(10, int(args.top_k))],
        "unremarkable_cross_organ": next((row for row in overlap_rows if row["text"] == "unremarkable"), None),
    }

    global_unremarkable = int(global_all_text_counter.get("unremarkable", 0))
    global_total = int(sum(global_all_text_counter.values()))
    result["global_summary"] = {
        "total_organ_text_mentions": global_total,
        "global_unique_text_count": len(global_all_text_counter),
        "global_unremarkable_mentions": global_unremarkable,
        "global_unremarkable_fraction": 0.0 if global_total == 0 else float(global_unremarkable) / float(global_total),
        "top_global_texts": [
            {"text": text, "count": int(count)}
            for text, count in global_all_text_counter.most_common(max(10, int(args.top_k)))
        ],
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")


def resolve_layout_root(dataset_root):
    if (dataset_root / "train").is_dir() and (dataset_root / "val").is_dir():
        return dataset_root
    if (dataset_root / "dataset_split" / "train").is_dir() and (dataset_root / "dataset_split" / "val").is_dir():
        return dataset_root / "dataset_split"
    raise IOError("Could not locate train/val split directories under {}".format(dataset_root))


def collect_usable_studies(layout_root, metadata_by_id):
    studies = []
    seen_ids = set()
    for split in ("train", "val"):
        split_dir = layout_root / split
        for case_dir in split_dir.iterdir():
            if not case_dir.is_dir():
                continue
            study_id = case_dir.name
            if study_id in seen_ids:
                continue
            seen_ids.add(study_id)
            scan_path = case_dir / "{}_resampled.nii.gz".format(study_id)
            seg_path = case_dir / "{}_seg_resampled.nii.gz".format(study_id)
            if not scan_path.is_file() or not seg_path.is_file():
                continue
            record = metadata_by_id.get(study_id)
            if record is None:
                continue
            findings = record.get("findings")
            if not isinstance(findings, dict):
                continue
            studies.append({"study_id": study_id, "split": split, "findings": findings})
    return studies


def normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def tokenize_for_similarity(text, organ_name):
    normalized = text.lower().replace(organ_name.lower(), " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", " <num> ", normalized)
    return set(token for token in normalized.split() if token not in STOPWORDS)


def organ_summary(
    organ_name,
    exact_counter,
    split_counter,
    raw_examples,
    top_k,
    near_duplicate_top_n,
    near_duplicate_ratio,
    near_duplicate_jaccard,
):
    total = int(sum(exact_counter.values()))
    unique = len(exact_counter)
    singleton_count = sum(1 for count in exact_counter.values() if count == 1)
    top_items = exact_counter.most_common(top_k)
    top5_mass = sum(count for _, count in exact_counter.most_common(5))
    top10_mass = sum(count for _, count in exact_counter.most_common(10))
    entropy = counter_entropy(exact_counter)
    unremarkable_exact = int(exact_counter.get("unremarkable", 0))
    contains_unremarkable = sum(count for text, count in exact_counter.items() if "unremarkable" in text)
    starts_with_normal = sum(
        count for text, count in exact_counter.items() if text.startswith("normal") or text.startswith("the ") or text.startswith("no ")
    )

    return {
        "total_mentions": total,
        "unique_text_count": unique,
        "unique_fraction": 0.0 if total == 0 else float(unique) / float(total),
        "singleton_text_count": singleton_count,
        "singleton_fraction_of_unique": 0.0 if unique == 0 else float(singleton_count) / float(unique),
        "top1_fraction": 0.0 if total == 0 or not top_items else float(top_items[0][1]) / float(total),
        "top5_fraction": 0.0 if total == 0 else float(top5_mass) / float(total),
        "top10_fraction": 0.0 if total == 0 else float(top10_mass) / float(total),
        "entropy_bits": entropy,
        "effective_class_count": 0.0 if entropy <= 0.0 else 2 ** entropy,
        "unremarkable_exact_count": unremarkable_exact,
        "unremarkable_exact_fraction": 0.0 if total == 0 else float(unremarkable_exact) / float(total),
        "contains_unremarkable_count": contains_unremarkable,
        "contains_unremarkable_fraction": 0.0 if total == 0 else float(contains_unremarkable) / float(total),
        "starts_with_normal_or_the_or_no_fraction": 0.0 if total == 0 else float(starts_with_normal) / float(total),
        "text_length_stats": length_stats(exact_counter),
        "top_texts": [
            {
                "text": text,
                "count": int(count),
                "fraction": 0.0 if total == 0 else float(count) / float(total),
                "raw_example": raw_examples.get(text, text),
                "train_count": int(split_counter.get("train", {}).get(text, 0)),
                "val_count": int(split_counter.get("val", {}).get(text, 0)),
            }
            for text, count in top_items
        ],
        "near_duplicate_clusters": find_near_duplicate_clusters(
            organ_name,
            exact_counter,
            raw_examples,
            near_duplicate_top_n,
            near_duplicate_ratio,
            near_duplicate_jaccard,
        ),
    }


def find_near_duplicate_clusters(organ_name, exact_counter, raw_examples, top_n, ratio_threshold, jaccard_threshold):
    candidates = [text for text, _ in exact_counter.most_common(top_n)]
    if len(candidates) < 2:
        return []

    token_cache = dict((text, tokenize_for_similarity(text, organ_name)) for text in candidates)
    adjacency = dict((text, set()) for text in candidates)
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            ratio = difflib.SequenceMatcher(a=left, b=right).ratio()
            jaccard = jaccard_similarity(token_cache[left], token_cache[right])
            if ratio >= ratio_threshold or jaccard >= jaccard_threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited = set()
    clusters = []
    for seed in candidates:
        if seed in visited or not adjacency[seed]:
            continue
        stack = [seed]
        component = []
        visited.add(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component) <= 1:
            continue
        component.sort(key=lambda text: (-exact_counter[text], text))
        total_count = sum(exact_counter[text] for text in component)
        clusters.append(
            {
                "cluster_size": len(component),
                "total_count": int(total_count),
                "members": [
                    {
                        "text": text,
                        "count": int(exact_counter[text]),
                        "fraction_within_cluster": 0.0 if total_count == 0 else float(exact_counter[text]) / float(total_count),
                        "raw_example": raw_examples.get(text, text),
                    }
                    for text in component[:8]
                ],
            }
        )
    clusters.sort(key=lambda row: (-row["total_count"], -row["cluster_size"]))
    return clusters[:10]


def counter_entropy(counter):
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = float(count) / float(total)
        if probability > 0.0:
            value -= probability * math.log(probability, 2)
    return value


def length_stats(counter):
    lengths = []
    for text, count in counter.items():
        token_len = len(text.split())
        lengths.extend([token_len] * int(count))
    if not lengths:
        return {"mean_tokens": 0.0, "p50_tokens": 0.0, "p90_tokens": 0.0}
    lengths.sort()
    return {
        "mean_tokens": float(sum(lengths)) / float(len(lengths)),
        "p50_tokens": float(lengths[len(lengths) // 2]),
        "p90_tokens": float(lengths[min(len(lengths) - 1, int(math.floor(0.9 * (len(lengths) - 1))))]),
    }


def jaccard_similarity(left, right):
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


if __name__ == "__main__":
    main()
