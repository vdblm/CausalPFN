"""
A callback designed for evaluating the model performance on a test set.
"""

from typing import Any, Callable, Dict, Protocol

import torch
import wandb
from torch.optim import Optimizer
from tqdm import tqdm

from causalpfn.causal_estimator import CATEEstimator
from causalpfn.evaluation import calculate_pehe
from causalpfn.models import InContextModel

from .base import Callback


class EvalCATE(Callback):
    """
    Evaluates the trained CATE model on a test set of causal datasets
    """

    def __init__(
        self,
        eval_datasets: Dict[str, "EvaluationDataset"],
        cate_estimator_partial: Callable[[InContextModel], CATEEstimator],
        callback_name: str | None,
        frequency: int = 1,
        estimator_fitting_kwargs: Dict | None = None,
    ):
        super().__init__(callback_name=callback_name)
        self.frequency = frequency
        self.eval_datasets = eval_datasets
        self.cate_estimator_partial = cate_estimator_partial
        self.num_evaluations = sum(len(eval_dataset) for eval_dataset in eval_datasets.values())
        self.pbar = None
        self.estimator_fitting_kwargs = estimator_fitting_kwargs or {}

    @torch.no_grad()
    def __call__(
        self,
        callback_idx: int,
        model: InContextModel,
        optimizer: Optimizer,
        epoch: int,
        train_loss: float,
        device: str,
        rank: int,
        wandb_enabled: bool,
        lr_scheduler=None,
    ):
        if rank != 0:
            return

        if (epoch + 1) % self.frequency == 0:
            if self.pbar is None:
                self.pbar = tqdm(range(self.num_evaluations), desc="Evaluating CATE")
            model.eval()
            report = {}
            for eval_dataset_name, eval_dataset in self.eval_datasets.items():
                all_cate_pehe = []
                for i in range(len(eval_dataset)):
                    cate_estimator: CATEEstimator = self.cate_estimator_partial(icl_model=model, device=device)
                    cate_data, ate_data = eval_dataset[i]
                    X_train, t_train, y_train, X_test = (
                        cate_data.X_train,
                        cate_data.t_train,
                        cate_data.y_train,
                        cate_data.X_test,
                    )
                    cate_estimator.fit(
                        X=X_train,
                        t=t_train,
                        y=y_train,
                        **self.estimator_fitting_kwargs,
                    )
                    cate_estimate = cate_estimator.estimate_cate(
                        X=X_test,
                    )
                    all_cate_pehe.append(calculate_pehe(cate_data.true_cate, cate_estimate))
                    self.pbar.update(1)
                report |= {f"{eval_dataset_name}/CATE PEHE": sum(all_cate_pehe) / len(all_cate_pehe)}
            self.pbar.reset()

            self.pbar.set_postfix(report)

            if wandb_enabled:
                wandb.log(
                    dict([(f"callback{callback_idx}:{self.callback_name}/{k}", v) for k, v in report.items()]),
                )

            model.train()


class EvaluationDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Any, Any]: ...
