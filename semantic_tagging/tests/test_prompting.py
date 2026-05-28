from semantic_tagging.config import PromptConfig, OntologyConfig
from semantic_tagging.ontology import OntologyRegistry
from semantic_tagging.paths import subproject_root
from semantic_tagging.prompting import PromptCompiler
from semantic_tagging.types import UniqueTextRecord


def test_prompt_compiler_injects_organ_and_subtypes() -> None:
    root = subproject_root()
    registry = OntologyRegistry(ontology_root=root / "ontology", config=OntologyConfig())
    compiler = PromptCompiler(
        prompt_root=root / "prompts",
        config=PromptConfig(
            system_template="system_v1.md",
            user_template="user_v1.md",
            output_schema="output_schema_v1.json",
            fewshot_dir="fewshot",
            max_fewshot_examples=2,
            include_existing_subtypes=True,
        ),
        ontology=registry,
    )
    record = UniqueTextRecord(
        organ="Pancreas",
        raw_text="The pancreas demonstrates normal attenuation. No pancreatic duct dilatation or focal masses.",
        normalized_text="the pancreas demonstrates normal attenuation. no pancreatic duct dilatation or focal masses.",
        count=10,
        split_counts={"train": 7, "val": 3},
        abnormal_positive_count=0,
        abnormal_negative_count=10,
        lesion_labeled_count=10,
        lesion_positive_count=0,
        lesion_positive_rate=0.0,
        abnormal_positive_rate=0.0,
    )
    request = compiler.compile_request(record, request_id="abc")
    assert "Organ: Pancreas" in request.prompt_text
    assert "pancreas_mass" in request.prompt_text
    assert "Example 1" in request.prompt_text
