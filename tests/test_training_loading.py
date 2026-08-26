from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from causalpfn.models import TabDPTLongContextModel, loading


def make_predictive_checkpoint(path: Path) -> None:
    model = TabDPTLongContextModel(
        dropout=0.0,
        n_out=4,
        nhead=1,
        nhid=8,
        ninp=4,
        nlayers=1,
        num_features=4,
        nbins=8,
    )
    config = OmegaConf.create(
        {
            "training": {"dropout": 0.0},
            "model": {
                "max_num_classes": 4,
                "nhead": 1,
                "emsize": 4,
                "nhid_factor": 2,
                "nlayers": 1,
                "max_num_features": 4,
                "nbins": 8,
            },
        }
    )
    torch.save({"cfg": config, "model": model.state_dict()}, path)


def test_local_checkpoint_takes_precedence_and_is_loaded_once(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "warm.ckpt"
    make_predictive_checkpoint(checkpoint_path)
    original_load = torch.load
    load_count = 0

    def counted_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(loading.torch, "load", counted_load)
    monkeypatch.setattr(loading, "hf_hub_download", lambda **kwargs: pytest.fail("local path must take precedence"))

    model = loading.load_pretrained_in_context_model(
        ckpt_path=checkpoint_path,
        repo_id="unused/repo",
        filename="unused.ckpt",
        revision="unused-revision",
        sigma=0.01,
    )

    assert load_count == 1
    assert model.model_config["model"]["nbins"] == 8
    assert model.sigma == 0.01


def test_hugging_face_resolution_uses_pinned_revision(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "warm.ckpt"
    checkpoint_path.touch()
    calls = []
    monkeypatch.setattr(loading, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or str(checkpoint_path))

    resolved = loading.resolve_pretrained_checkpoint(
        ckpt_path=None,
        repo_id="vdblm/causalpfn",
        filename="tabdpt_long_context.ckpt",
        revision="immutable-commit",
    )

    assert resolved == str(checkpoint_path)
    assert calls == [
        {
            "repo_id": "vdblm/causalpfn",
            "filename": "tabdpt_long_context.ckpt",
            "revision": "immutable-commit",
        }
    ]


def test_checkpoint_resolution_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="warm-start checkpoint not found"):
        loading.resolve_pretrained_checkpoint(tmp_path / "missing.ckpt", None, None, None)

    with pytest.raises(ValueError, match="pinned revision"):
        loading.resolve_pretrained_checkpoint(None, "vdblm/causalpfn", "tabdpt_long_context.ckpt", None)
