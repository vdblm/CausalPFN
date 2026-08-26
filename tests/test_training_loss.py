import torch
from torch.utils.data import IterableDataset

from causalpfn.training.trainer import calculate_loss, distributed_mean, train


class ScalarLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.model_config = {"model_type": "test"}

    def get_param_groups(self):
        return self.parameters()

    def forward(self, X_context, **kwargs):
        return self.weight.square().expand(X_context.shape[0])


class NonFiniteLossModel(ScalarLossModel):
    def forward(self, X_context, **kwargs):
        return (self.weight * torch.full((), float("nan"))).expand(X_context.shape[0])


class TinyMetaDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield {
                "X": torch.randn(12, 4),
                "t": torch.tensor([0.0, 1.0] * 6),
                "y": torch.randn(12),
                "E_y0": torch.randn(12),
                "E_y1": torch.randn(12),
            }


def make_batch():
    return {
        "X": torch.randn(2, 12, 4),
        "t": torch.tensor([[0.0, 1.0] * 6, [1.0, 0.0] * 6]),
        "y": torch.randn(2, 12),
        "E_y0": torch.randn(2, 12),
        "E_y1": torch.randn(2, 12),
    }


def test_calculate_loss_is_differentiable():
    model = ScalarLossModel()
    loss = calculate_loss(model, "cpu", make_batch(), 0.5, 0.5, overlap_threshold=0.1)
    loss.backward()
    assert loss.item() == 1.0
    assert model.weight.grad.item() == 2.0


def test_calculate_loss_keeps_graph_when_every_table_is_invalid():
    model = NonFiniteLossModel()
    loss = calculate_loss(model, "cpu", make_batch(), 0.5, 0.5, overlap_threshold=0.1)
    loss.backward()
    assert loss.item() == 0.0
    assert model.weight.grad is not None
    assert model.weight.grad.item() == 0.0


def test_one_cpu_optimizer_update_with_no_workers():
    model = ScalarLossModel()
    initial_weight = model.weight.detach().clone()

    train(
        model=model,
        max_epochs=1,
        num_agg=1,
        num_model_updates=1,
        optimizer_partial=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        train_meta_dataset=TinyMetaDataset(),
        callbacks=[],
        lr_scheduler_partial=None,
        compile=False,
        num_workers=0,
        prefetch_factor=2,
        checkpoint=None,
        wandb_enabled=False,
        batch_size=2,
        grad_clip=1.0,
        device="cpu",
        min_train_data_split=0.5,
        max_train_data_split=0.5,
        overlap_threshold=0.1,
        rank=0,
        using_dist=False,
        world_size=1,
    )

    assert not torch.equal(model.weight.detach(), initial_weight)


def test_distributed_mean_uses_all_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op: tensor.mul_(2))
    assert distributed_mean(3.0, "cpu", using_dist=True, world_size=2) == 3.0


def test_training_restores_and_continues_scheduler_state():
    source_model = ScalarLossModel()
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1, gamma=0.5)
    source_optimizer.step()
    source_scheduler.step()
    checkpoint = {
        "optimizer_state_dict": source_optimizer.state_dict(),
        "lr_scheduler_state_dict": source_scheduler.state_dict(),
        "epoch": 0,
    }
    observed = {}

    def callback(**kwargs):
        observed["scheduler"] = kwargs["lr_scheduler"].state_dict()

    train(
        model=ScalarLossModel(),
        max_epochs=2,
        num_agg=1,
        num_model_updates=1,
        optimizer_partial=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        train_meta_dataset=TinyMetaDataset(),
        callbacks=[callback],
        lr_scheduler_partial=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        compile=False,
        num_workers=0,
        prefetch_factor=2,
        checkpoint=checkpoint,
        wandb_enabled=False,
        batch_size=2,
        grad_clip=1.0,
        device="cpu",
        min_train_data_split=0.5,
        max_train_data_split=0.5,
        overlap_threshold=0.1,
        rank=0,
        using_dist=False,
        world_size=1,
    )

    assert observed["scheduler"]["last_epoch"] == source_scheduler.last_epoch + 1
