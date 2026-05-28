from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from merlin_ablation.config import load_config
from merlin_ablation.data import build_datasets


def test_smoke_config_loads():
    config = load_config(ROOT / "configs" / "smoke_ce_only.yaml")
    assert config.train.run_id == "smoke_ce_only"
    assert config.losses.ce_weight == 1.0
    assert config.model.append_eos_to_target is True


def test_dataset_records_have_expected_fields():
    config = load_config(ROOT / "configs" / "smoke_sem_family.yaml")
    bundle = build_datasets(config)
    assert bundle.train_records
    row = bundle.train_records[0]
    assert Path(row["image"]).is_file()
    assert row["prompt"].startswith("Generate a radiology report for ")
    assert row["full_text"].startswith(row["prompt"])
    assert "semantic_family_targets" in row
