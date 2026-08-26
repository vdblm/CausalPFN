import math
from contextlib import nullcontext
from typing import Any, Callable, Dict

import torch
import wandb
from tqdm import tqdm

from causalpfn.models import InContextModel

from .priors.meta_dataset import MetaDataset


def distributed_mean(value: float, device: str, using_dist: bool, world_size: int) -> float:
    if not using_dist:
        return value
    reduced = torch.tensor(value, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    return (reduced / world_size).item()


def calculate_loss(
    model: InContextModel,
    device: str,
    batch: dict,
    min_train_data_split: float,
    max_train_data_split: float,
    # Ignore a table when either treatment arm has fewer than this fraction of context samples.
    overlap_threshold: float = 0.05,
) -> torch.Tensor:
    """
    This function uses an in-context model to compute the loss for training the causal (foundation) model.
    `model` itself is a PFN-style model that takes in `t_train` `X_train, y_train` and an `X_query` and `t_query`
    and then produces expected potential outcome predictions.

    *Note*:
        The final loss averages the valid table losses in a batch. Tables with non-finite losses or insufficient treatment
        overlap are ignored.

    Args:
        model: The in-context model to use for training.
        device: The device to use for training.
        batch: The batch of data to use for training.
        min_train_data_split: The minimum fraction of training data to use for context.
        max_train_data_split: The maximum fraction of training data to use for context.
        overlap_threshold: The threshold for the overlap condition.
    Returns:
        avg_total_loss: The average total loss for the batch.
    """
    # shuffle the covariate columns to induce column permutation invariance
    X = batch["X"].to(device, non_blocking=True)  # shape: (batch_size, num_rows, num_features)
    idx = torch.randperm(batch["X"].shape[-1], device=device)  # create idx on GPU
    X = X[:, :, idx]  # shape: (batch_size, num_rows, num_features)
    split_pos = int(
        X.shape[1] * (torch.rand(()) * (max_train_data_split - min_train_data_split) + min_train_data_split)
    )
    t = batch["t"].to(device, non_blocking=True)  # shape: (batch_size, num_rows)
    y = batch["y"].to(device, non_blocking=True)  # shape: (batch_size, num_rows)
    E_y0, E_y1 = batch["E_y0"].to(device, non_blocking=True), batch["E_y1"].to(
        device, non_blocking=True
    )  # shape: (batch_size, num_rows)
    # compute the cepo loss
    cepo_losses = model(
        X_context=X[:, :split_pos],
        t_context=t[:, :split_pos],
        y_context=y[:, :split_pos],
        X_query=X[:, split_pos:],
        E_y0_query=E_y0[:, split_pos:],
        E_y1_query=E_y1[:, split_pos:],
    )
    # ignore the the tables where the treatment or control group has less than threshold samples
    # This basically ignores the tables where the overlap condition is not satisfied
    t_context = t[:, :split_pos]
    treated_counts = (t_context == 1).long().sum(dim=1)
    control_counts = (t_context == 0).long().sum(dim=1)
    # some tables might be ignored in the loss calculation here
    valid_mask = torch.ones(batch["X"].shape[0], dtype=torch.bool, device=device)
    valid_mask &= treated_counts >= overlap_threshold * split_pos
    valid_mask &= control_counts >= overlap_threshold * split_pos
    valid_mask &= torch.isfinite(cepo_losses)
    valid_losses = cepo_losses[valid_mask]
    if valid_losses.numel() > 0:
        return valid_losses.mean()

    # A standalone zero would bypass DDP's gradient hooks and can deadlock when
    # another rank has valid tables. Attach a finite zero to every trainable
    # parameter so all ranks participate with zero gradients for this batch.
    zero_loss = cepo_losses.new_zeros(())
    for parameter in model.parameters():
        if parameter.requires_grad:
            zero_loss = zero_loss + parameter.reshape(-1)[0] * 0.0
    return zero_loss


def train(
    model: InContextModel,
    max_epochs: int,
    num_agg: int,
    num_model_updates: int,
    optimizer_partial: Callable[[Any], torch.optim.Optimizer],
    train_meta_dataset: MetaDataset,
    callbacks: list,
    lr_scheduler_partial: Callable[[torch.optim.Optimizer], Any] | None,
    compile: bool,
    num_workers: int,
    prefetch_factor: int,
    checkpoint: Dict[str, Any] | None,
    wandb_enabled: bool,
    batch_size: int,
    grad_clip: float | None,
    device: str,
    min_train_data_split: float,
    max_train_data_split: float,
    overlap_threshold: float,
    rank: int,
    using_dist: bool,
    world_size: int,
):
    """
    Takes a model designed for prior-fitting (`model`) and then trains CATE on a given set of datasets.

    Evaluation is handled through callbacks in ``causalpfn.training.callbacks``.
    """

    barrier_kwargs = {"device_ids": [torch.device(device).index]} if device.startswith("cuda") else {}

    model.to(device)
    model.train()

    if compile:
        # Dynamic shapes are required because the context/query split changes each batch.
        model = torch.compile(model, dynamic=True)

    if using_dist:
        ddp_kwargs = {}
        if device.startswith("cuda"):
            local_device_index = torch.device(device).index
            ddp_kwargs = {"device_ids": [local_device_index], "output_device": local_device_index}
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        optimizer = optimizer_partial(model.module.get_param_groups())
    else:
        optimizer = optimizer_partial(model.get_param_groups())

    if checkpoint:  # load optimizer state if resume = True
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
        lr_scheduler_partial(optimizer) if lr_scheduler_partial is not None else None
    )
    if checkpoint and lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = torch.utils.data.DataLoader(train_meta_dataset, **loader_kwargs)
    train_loader_iterator = iter(train_loader)

    # create three progress bars
    pbar_train = tqdm(
        range(max_epochs * num_model_updates * num_agg * world_size), desc="Train Batches", disable=(rank != 0)
    )

    print(f"Effective batch size is: {num_agg * batch_size * world_size}")

    for epoch in range(max_epochs):
        if epoch < start_epoch:
            pbar_train.update(num_model_updates * num_agg * world_size)
            continue

        model.train()
        if hasattr(optimizer, "train"):  # for schedulefree
            optimizer.train()

        # run the model for num_agg * num_model_updates iterations
        total_loss = 0.0
        for batch_counter in range(num_agg * num_model_updates):
            # accumate the loss gradients over num_agg batches for larger effective batch size
            train_batch = next(train_loader_iterator)
            last_micro_batch = ((batch_counter + 1) % num_agg) == 0
            sync_context = model.no_sync() if using_dist and not last_micro_batch else nullcontext()
            with sync_context:
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16,
                ):
                    loss = calculate_loss(
                        model,
                        device,
                        batch=train_batch,
                        min_train_data_split=min_train_data_split,
                        max_train_data_split=max_train_data_split,
                        overlap_threshold=overlap_threshold,
                    )
                    loss = loss / num_agg
                    total_loss += loss.item()
                    loss.backward()

            if last_micro_batch:
                # short-circuit if the gradients are messed up
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip or float("inf")
                ).item()
                # A generated task can occasionally be numerically degenerate; discard its update on every rank.
                skip_local = (grad_norm == 0) or math.isinf(grad_norm) or math.isnan(grad_norm)
                skip_flag = torch.tensor(skip_local, device=device, dtype=torch.uint8)
                if using_dist:
                    torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)

                pbar_train.update(num_agg * world_size)

                if skip_flag.item():
                    optimizer.zero_grad(set_to_none=True)  # discard the bad step
                    print("[Warning] non-finite, NaN, or empty gradients – skipping update.")
                    continue

                if wandb_enabled and rank == 0:
                    with torch.no_grad():
                        weight_norm = torch.linalg.vector_norm(
                            torch.stack([p.norm() for p in model.parameters()])
                        ).item()
                    wandb.log(
                        {
                            f"weights/grad_norm": grad_norm,
                            f"weights/weight_norm": weight_norm,
                            f"weights/avg_lr": sum(
                                [optimizer.param_groups[i]["lr"] for i in range(len(optimizer.param_groups))]
                            )
                            / len(optimizer.param_groups),
                        },
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()

        # update the loss reports
        total_loss /= num_model_updates
        total_loss = distributed_mean(total_loss, device, using_dist, world_size)
        loss_report = {"train/loss": total_loss}

        # log the loss values
        pbar_train.set_postfix(loss_report)
        if wandb_enabled:
            wandb.log(loss_report)

        if using_dist:
            torch.distributed.barrier(**barrier_kwargs)

        # make the optimizer eval mode
        if hasattr(optimizer, "eval"):
            optimizer.eval()

        # run the __call__ method of each callback
        for i, callback in enumerate(callbacks):
            pbar_train.set_description(f"Total Epochs (Callback [{i+1}/{len(callbacks)}])")
            callback(
                callback_idx=i,
                model=model if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=total_loss,
                device=device,
                rank=rank,
                wandb_enabled=wandb_enabled,
                lr_scheduler=lr_scheduler,
            )
            pbar_train.set_description("Total Epochs (Training ...)")

        # bring back the optimizer to train mode
        if hasattr(optimizer, "train"):
            optimizer.train()

        if using_dist:
            torch.distributed.barrier(**barrier_kwargs)
