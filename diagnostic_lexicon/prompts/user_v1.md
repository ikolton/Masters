{system_instructions}

Organ: {organ}
Subtype: {subtype}
Family: {family}
Canonical label: {canonical_label}
Subtype maturity tier: {maturity_tier}

Subtype positive examples:
{positive_examples}

Subtype contrast examples:
{contrast_examples}

Real mined candidate phrases with support:
{candidate_phrases}

Instructions:
- Prefer phrases supported by the real examples.
- Merge obvious duplicates and wording variants.
- Put target-supporting wording into `positive_lexicalizations`.
- Put explicit negations or absence phrases into `negative_lexicalizations`.
- Put uncertain or equivocal wording into `uncertain_lexicalizations`.
- Put phrases that look similar but should not be treated as evidence into `confusers`.
- Put phrases that are low-quality, over-broad, or misleading into `discouraged_lexicalizations`.
- Be conservative.

Return JSON matching this schema:
{output_schema}
