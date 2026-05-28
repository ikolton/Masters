from __future__ import annotations

import json

import pandas as pd

from organ_seg_clip.evaluation.radgpt_oncology import (
    _raise_if_transport_failed,
    _resolve_client_base_url,
    build_radgpt_report_rows,
    compare_radgpt_labels,
    load_or_label_radgpt_reports,
)


def test_build_radgpt_report_rows_filters_to_supported_organs() -> None:
    raw_rows = [
        {"study_id": "AC1", "organ": "Liver", "target": "Liver lesion.", "generated": "Generated liver."},
        {"study_id": "AC1", "organ": "Spleen", "target": "Normal spleen.", "generated": "Generated spleen."},
        {"study_id": "AC2", "organ": "Kidneys", "target": "", "generated": "Generated kidneys."},
        {"study_id": "AC3", "organ": "Pancreas", "target": "Pancreatic cyst.", "generated": "Generated pancreas."},
    ]

    rows = build_radgpt_report_rows(raw_rows, text_field="target")

    assert rows == [
        {
            "sample_id": "AC1::Liver::0",
            "study_id": "AC1",
            "organ": "Liver",
            "radgpt_organ": "liver",
            "report_text": "Liver lesion.",
        },
        {
            "sample_id": "AC3::Pancreas::3",
            "study_id": "AC3",
            "organ": "Pancreas",
            "radgpt_organ": "pancreas",
            "report_text": "Pancreatic cyst.",
        },
    ]


def test_compare_radgpt_labels_computes_expected_metrics() -> None:
    reference_df = pd.DataFrame(
        [
            {"sample_id": "A", "study_id": "S1", "organ": "Liver", "radgpt_organ": "liver", "report_text": "", "tumor_label": 1, "malignancy_label": 1, "tumor_answer": "", "malignancy_answer": "", "warning": ""},
            {"sample_id": "B", "study_id": "S2", "organ": "Kidneys", "radgpt_organ": "kidney", "report_text": "", "tumor_label": 1, "malignancy_label": 1, "tumor_answer": "", "malignancy_answer": "", "warning": ""},
            {"sample_id": "C", "study_id": "S3", "organ": "Pancreas", "radgpt_organ": "pancreas", "report_text": "", "tumor_label": 1, "malignancy_label": 1, "tumor_answer": "", "malignancy_answer": "", "warning": ""},
        ]
    )
    generated_df = pd.DataFrame(
        [
            {"sample_id": "A", "study_id": "S1", "organ": "Liver", "radgpt_organ": "liver", "report_text": "", "tumor_label": 1, "malignancy_label": 1, "tumor_answer": "", "malignancy_answer": "", "warning": ""},
            {"sample_id": "B", "study_id": "S2", "organ": "Kidneys", "radgpt_organ": "kidney", "report_text": "", "tumor_label": 0, "malignancy_label": 0, "tumor_answer": "", "malignancy_answer": "", "warning": ""},
            {"sample_id": "C", "study_id": "S3", "organ": "Pancreas", "radgpt_organ": "pancreas", "report_text": "", "tumor_label": 1, "malignancy_label": 0, "tumor_answer": "", "malignancy_answer": "", "warning": ""},
        ]
    )

    result = compare_radgpt_labels(reference_df, generated_df)

    assert result["per_organ"]["Liver"]["tumor"]["f1"] == 1.0
    assert result["per_organ"]["Kidneys"]["tumor"]["accuracy"] == 0.0
    assert result["per_organ"]["Pancreas"]["malignancy"]["accuracy"] == 0.0
    assert result["overall"]["tumor_macro_f1"] == (1.0 + 0.0 + 1.0) / 3.0
    assert result["overall"]["malignancy_macro_f1"] == (1.0 + 0.0 + 0.0) / 3.0


def test_load_or_label_radgpt_reports_resumes_missing_rows(tmp_path) -> None:
    report_rows = [
        {"sample_id": "AC1::Liver::0", "study_id": "AC1", "organ": "Liver", "radgpt_organ": "liver", "report_text": "A"},
        {"sample_id": "AC2::Pancreas::1", "study_id": "AC2", "organ": "Pancreas", "radgpt_organ": "pancreas", "report_text": "B"},
    ]
    calls: list[list[str]] = []

    def fake_labeler(rows, **kwargs):
        del kwargs
        calls.append([row["sample_id"] for row in rows])
        return pd.DataFrame(
            [
                {
                    "sample_id": row["sample_id"],
                    "study_id": row["study_id"],
                    "organ": row["organ"],
                    "radgpt_organ": row["radgpt_organ"],
                    "report_text": row["report_text"],
                    "tumor_label": 1,
                    "malignancy_label": 0,
                    "tumor_answer": "tumor",
                    "malignancy_answer": "mal",
                    "warning": "",
                }
                for row in rows
            ]
        )

    metadata = {"schema_version": 1, "input_digest": "abc"}
    cache_dir = tmp_path / "radgpt_cache"

    first_df = load_or_label_radgpt_reports(
        report_rows,
        cache_dir=cache_dir,
        metadata=metadata,
        base_url="http://0.0.0.0:8000/v1",
        fast=True,
        force=False,
        quiet=True,
        progress_prefix="test",
        progress_every=10,
        radgpt_root="/tmp/radgpt",
        labeler=fake_labeler,
    )
    assert len(first_df) == 2
    assert calls == [["AC1::Liver::0", "AC2::Pancreas::1"]]

    labels_path = cache_dir / "labels.csv"
    partial_df = pd.read_csv(labels_path).iloc[:1]
    partial_df.to_csv(labels_path, index=False)
    (cache_dir / "metadata.json").write_text(json.dumps({**metadata, "row_count": 2}, indent=2), encoding="utf-8")

    second_df = load_or_label_radgpt_reports(
        report_rows,
        cache_dir=cache_dir,
        metadata=metadata,
        base_url="http://0.0.0.0:8000/v1",
        fast=True,
        force=False,
        quiet=True,
        progress_prefix="test",
        progress_every=10,
        radgpt_root="/tmp/radgpt",
        labeler=fake_labeler,
    )
    assert len(second_df) == 2
    assert calls[-1] == ["AC2::Pancreas::1"]


def test_resolve_client_base_url_rewrites_wildcard_host() -> None:
    assert _resolve_client_base_url("http://0.0.0.0:8000/v1") == "http://127.0.0.1:8000/v1"
    assert _resolve_client_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_raise_if_transport_failed_aborts_all_connection_failures() -> None:
    records = [
        {
            "tumor_answer": "",
            "malignancy_answer": "",
            "warning": "APIConnectionError: Connection error.",
        }
        for _ in range(3)
    ]
    try:
        _raise_if_transport_failed(records, base_url="http://127.0.0.1:8000/v1", progress_prefix="reference")
    except RuntimeError as exc:
        assert "API connection failures" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError for all-transport-failure RadGPT pass.")
