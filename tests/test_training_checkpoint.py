import torch

from causalpfn.training.callbacks import Checkpoint


class CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.model_config = {"model_type": "test", "sigma": 0.01}


def test_checkpoint_contains_resumable_state(tmp_path):
    model = CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    callback = Checkpoint(
        checkpoint_root=str(tmp_path),
        callback_name="checkpoint",
        frequency=1,
        checkpoint_dir_name="test-run",
        top_k=1,
    )

    callback(
        callback_idx=0,
        model=model,
        optimizer=optimizer,
        epoch=0,
        train_loss=0.5,
        device="cpu",
        rank=0,
        wandb_enabled=False,
        lr_scheduler=scheduler,
    )

    saved = torch.load(tmp_path / "test-run" / "latest.pt", weights_only=False, map_location="cpu")
    assert saved["epoch"] == 0
    assert saved["train_loss"] == 0.5
    assert saved["model_config"] == model.model_config
    assert "weight" in saved["model_state_dict"]
    assert saved["optimizer_state_dict"]["param_groups"]
    assert saved["lr_scheduler_state_dict"] == scheduler.state_dict()
    assert saved["checkpoint_dir_name"] == "test-run"
    assert saved["top_scores"] == [0.5]
    assert saved["top_epochs"] == [0]


def test_checkpoint_resume_preserves_custom_directory_and_bounded_top_k(tmp_path):
    model = CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    callback = Checkpoint(
        checkpoint_root=str(tmp_path),
        callback_name="checkpoint",
        frequency=1,
        checkpoint_dir_name="custom-directory",
        top_k=1,
    )
    callback(0, model, optimizer, 0, 0.5, "cpu", 0, False)
    first = torch.load(tmp_path / "custom-directory" / "latest.pt", weights_only=False)

    resumed = Checkpoint(
        checkpoint_root=str(tmp_path),
        callback_name="checkpoint",
        frequency=1,
        checkpoint_dir_name=None,
        top_k=1,
    )
    resumed.load_state(first)
    resumed(0, model, optimizer, 1, 0.4, "cpu", 0, False)

    checkpoint_dir = tmp_path / "custom-directory"
    assert resumed.checkpoint_dir_name == "custom-directory"
    assert [path.name for path in checkpoint_dir.glob("epoch_*.pt")] == ["epoch_0001.pt"]
    latest = torch.load(checkpoint_dir / "latest.pt", weights_only=False)
    assert latest["top_scores"] == [0.4]
    assert latest["top_epochs"] == [1]


def test_checkpoint_accepts_legacy_state(tmp_path):
    callback = Checkpoint(
        checkpoint_root=str(tmp_path),
        callback_name="checkpoint",
        frequency=1,
        checkpoint_dir_name=None,
        top_k=2,
    )
    callback.load_state({"run_name": "legacy-run"})

    assert callback.checkpoint_dir_name == "legacy-run"
    assert callback.top_scores == [float("inf"), float("inf")]
    assert callback.top_epochs == [-1, -1]
