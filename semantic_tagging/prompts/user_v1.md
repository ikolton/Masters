{system_instructions}

Organ: {organ}
Organ maturity tier: {organ_maturity}
Online expansion allowed for this organ: {allow_online_expansion}

Candidate text:
{raw_text}

Normalized text:
{normalized_text}

Observed corpus support:
- count: {count}
- lesion_positive_rate: {lesion_positive_rate}
- abnormal_positive_rate: {abnormal_positive_rate}

Existing subtypes for this organ:
{existing_subtypes}

Important family rules:
- You may only use an existing allowed global family name in `proposed_new_subtype.family`.
- If you think a genuinely new family is needed, DO NOT invent that family name inside `proposed_new_subtype.family`.
- In that case, set `proposed_new_subtype.family` to the best existing fallback family, usually `other_abnormal`.
- Also fill `proposed_new_family` with your suggested new family name and rationale.
- If no new family is needed, set `proposed_new_family` to null.
- If a finding is explicitly negated, do not create a subtype name ending in `_negated`; encode the negation in `polarity` and leave the negated finding out of `secondary_subtypes`.
- If the text includes adjacent findings that belong to a different organ system, ignore them unless they are necessary to define the target-organ tag.

Few-shot examples:
{fewshot_examples}

Return JSON matching this schema:
{output_schema}
