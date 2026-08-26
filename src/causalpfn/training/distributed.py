import datetime
import os
import random
import signal
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist


def cleanup() -> None:
    """Destroy the distributed process group when one is active."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def signal_handler(*_) -> None:
    """Clean up distributed state after an interrupt."""
    print("Received Ctrl+C, exiting...")
    cleanup()
    raise KeyboardInterrupt


def install_signal_handler() -> None:
    signal.signal(signal.SIGINT, signal_handler)


def init_dist(device: str) -> Tuple[bool, int, str, int]:
    """Initialize torch.distributed from torchrun environment variables."""
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, device, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        resolved_device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        resolved_device = "cpu"
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=600),
        world_size=world_size,
        rank=rank,
    )
    barrier_kwargs = {"device_ids": [local_rank]} if backend == "nccl" else {}
    dist.barrier(**barrier_kwargs)
    return True, rank, resolved_device, world_size


def seed_everything(base_seed: int, using_dist: bool, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch, using a distinct seed per rank."""
    if using_dist:
        seed_source = random.Random(base_seed)
        seeds = [seed_source.randint(0, 2**32 - 1) for _ in range(int(os.environ["WORLD_SIZE"]))]
        seed = seeds[rank]
    else:
        seed = base_seed

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
