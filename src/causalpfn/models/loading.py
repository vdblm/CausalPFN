from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from .model import TabDPTLongContextModel


class DictToObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    else:
        return obj


def load_pretrained_tabdpt_model(ckpt_path: str | None, ckpt: dict | None = None) -> TabDPTLongContextModel:
    assert ckpt_path is not None or ckpt is not None, "Either ckpt_path or ckpt must be provided."
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu") if ckpt is None else ckpt
    config = DictToObject(checkpoint["cfg"])
    model = TabDPTLongContextModel.load(
        model_state=checkpoint["model"],
        config=config,
    )
    return model


def load_pretrained_tabdpt_config(ckpt_path: str | None, ckpt: dict | None = None) -> dict:
    assert ckpt_path is not None or ckpt is not None, "Either ckpt_path or ckpt must be provided."
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu") if ckpt is None else ckpt
    config = checkpoint["cfg"]
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return object_to_dict(config)


def resolve_pretrained_checkpoint(
    ckpt_path: str | Path | None,
    repo_id: str | None,
    filename: str | None,
    revision: str | None,
) -> str:
    """Resolve a local warm start or download an immutable Hugging Face revision."""
    if ckpt_path is not None:
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Predictive warm-start checkpoint not found: {path}")
        return str(path)

    if not repo_id or not filename or not revision:
        raise ValueError("repo_id, filename, and a pinned revision are required when ckpt_path is not set.")
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def load_pretrained_in_context_model(
    ckpt_path: str | Path | None = None,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
    sigma: float = 0.5,
):
    """Load one predictive checkpoint and wrap it for causal in-context training."""
    from .icl_model import InContextModel

    resolved_path = resolve_pretrained_checkpoint(ckpt_path, repo_id, filename, revision)
    checkpoint = torch.load(resolved_path, weights_only=False, map_location="cpu")
    model_config = load_pretrained_tabdpt_config(ckpt_path=None, ckpt=checkpoint)
    model = load_pretrained_tabdpt_model(ckpt_path=None, ckpt=checkpoint)
    return InContextModel(model=model, model_config=model_config, sigma=sigma)
