from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from merlin_ablation.config import LossConfig
from merlin_ablation.losses import AuxiliaryDiagnosticLosses


def test_auxiliary_losses_forward():
    config = LossConfig(lexical_weight=0.002, semantic_weight=0.005, semantic_variant="family")
    module = AuxiliaryDiagnosticLosses(config, hidden_size=8, family_count=3, subtype_count=4)
    hidden = torch.randn(2, 8)
    batch = {
        "lexical_label": torch.tensor([1.0, 0.0]),
        "lexical_available": torch.tensor([True, True]),
        "semantic_available": torch.tensor([True, False]),
        "semantic_weight": torch.tensor([0.9, 0.0]),
        "semantic_normality": torch.tensor([1, -100]),
        "semantic_polarity": torch.tensor([0, -100]),
        "semantic_family_targets": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "semantic_family_allowed": torch.tensor([[True, True, False], [True, True, False]]),
        "semantic_subtype_targets": torch.zeros(2, 4),
        "semantic_subtype_allowed": torch.ones(2, 4, dtype=torch.bool),
    }
    output = module(hidden, batch)
    assert output.total.ndim == 0
    assert output.lexical_count == 2
    assert output.semantic_count == 1

