"""Regression tests for the orchestrator's YAML loader.

PyYAML's default float resolver follows YAML 1.1, which does not match
scientific notation lacking a decimal point (e.g. ``3e-4``). LightningCLI
sidesteps this by using jsonargparse, but our orchestrator loads YAML
directly, so we install a custom resolver that matches the YAML 1.2 /
jsonargparse behavior. These tests pin that.
"""

import sys
from pathlib import Path

# Make `scripts/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kfold_label_audit import _load_config  # noqa: E402


def test_load_config_parses_scientific_no_decimal_as_float(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "model:\n"
        "  init_args:\n"
        "    optimizer_opts:\n"
        "      lr: 3e-4\n"
        "      weight_decay: 1e-5\n"
        "      betas: [0.9, 0.999]\n"
    )
    cfg = _load_config(cfg_path)
    opts = cfg["model"]["init_args"]["optimizer_opts"]
    assert isinstance(opts["lr"], float)
    assert opts["lr"] == 3e-4
    assert isinstance(opts["weight_decay"], float)
    assert opts["weight_decay"] == 1e-5


def test_load_config_parses_scientific_with_decimal_as_float(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model:\n  init_args:\n    lr: 3.0e-4\n")
    cfg = _load_config(cfg_path)
    assert cfg["model"]["init_args"]["lr"] == 3.0e-4


def test_load_config_keeps_strings_as_strings(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "model:\n"
        "  class_path: src.models.smp.SMPMulticlassSegmentationModel\n"
        "  init_args:\n"
        "    architecture: DPT\n"
    )
    cfg = _load_config(cfg_path)
    assert cfg["model"]["class_path"] == "src.models.smp.SMPMulticlassSegmentationModel"
    assert cfg["model"]["init_args"]["architecture"] == "DPT"
