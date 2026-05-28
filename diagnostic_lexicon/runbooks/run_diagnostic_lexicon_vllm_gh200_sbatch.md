# Queueing Plan For Diagnostic Lexicon

This subproject is intended to be queueable in the same style as `semantic_tagging`, but it is currently at the **designed/scaffolded** stage.

Planned batch flow:

1. allocate GH200 node
2. activate dedicated inference environment
3. launch local vLLM server
4. wait for authenticated readiness
5. run lexical pipeline config
6. write outputs under `outputs/diagnostic_lexicon/...`

Planned wrapper template:

- [examples/run_diagnostic_lexicon_vllm_gh200.sbatch](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/examples/run_diagnostic_lexicon_vllm_gh200.sbatch:1)

This template intentionally exits with a message until the actual CLI is implemented.
