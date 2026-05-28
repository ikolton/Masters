# Architecture

The semantic tagging subproject is intentionally self-contained.

Main layers:
- dataset adapter
- ontology registry
- prompt compiler
- backend abstraction
- validation and repair
- online subtype proposal handling
- consolidation
- row-level propagation
- loss-target materialization

Primary processing unit:
- unique `(organ, raw_text)` records

Primary truth artifact:
- structured tag record

Derived artifacts:
- row-level tags
- loss-ready targets
