from .types import LossReadyTarget, RowLevelTag


def materialize_loss_targets(rows: list[RowLevelTag]) -> list[LossReadyTarget]:
    targets: list[LossReadyTarget] = []
    for row in rows:
        contradiction_flags = tuple(flag for flag in row.validation_flags if "contradiction" in flag or "with_" in flag)
        targets.append(
            LossReadyTarget(
                study_id=row.study_id,
                split=row.split,
                organ=row.organ,
                raw_text=row.raw_text,
                normality=row.normality,
                polarity=row.polarity,
                certainty=row.certainty,
                primary_subtype=row.primary_subtype,
                secondary_subtypes=row.secondary_subtypes,
                confidence_weight=float(row.confidence),
                contradiction_flags=contradiction_flags,
                provenance=row.decision_source,
                lesion_label=row.lesion_label,
                lesion_mask=row.lesion_mask,
                organ_abnormal_label=row.organ_abnormal_label,
            )
        )
    return targets
