from pathlib import Path

from .types import RunSummary


def write_summary_markdown(path: Path, summary: RunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantic Tagging Run Summary",
        "",
        f"- dataset_id: `{summary.dataset_id}`",
        f"- run_id: `{summary.run_id}`",
        f"- source_row_count: `{summary.source_row_count}`",
        f"- unique_text_count: `{summary.unique_text_count}`",
        f"- validated_decision_count: `{summary.validated_decision_count}`",
        f"- provisional_subtype_count: `{summary.provisional_subtype_count}`",
        f"- row_level_tag_count: `{summary.row_level_tag_count}`",
        f"- loss_target_count: `{summary.loss_target_count}`",
        "",
        "## Organ Counts",
        "",
    ]
    for organ, count in sorted(summary.organ_counts.items()):
        lines.append(f"- `{organ}`: `{count}`")
    lines.extend(["", "## Decision Status Counts", ""])
    for status, count in sorted(summary.status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
