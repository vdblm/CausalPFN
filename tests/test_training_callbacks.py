from types import SimpleNamespace

import numpy as np
import torch

from causalpfn.training.callbacks import EvalCATE


class OneEvaluationDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        cate = SimpleNamespace(
            X_train=np.zeros((4, 2)),
            t_train=np.array([0, 1, 0, 1]),
            y_train=np.zeros(4),
            X_test=np.zeros((2, 2)),
            true_cate=np.zeros(2),
        )
        return cate, SimpleNamespace()


class FakeEstimator:
    def fit(self, **kwargs):
        return self

    def estimate_cate(self, X):
        return np.zeros(len(X))


def test_nonzero_rank_does_not_evaluate_or_change_model_mode():
    calls = []
    callback = EvalCATE(
        eval_datasets={"tiny": OneEvaluationDataset()},
        cate_estimator_partial=lambda **kwargs: calls.append(kwargs),
        callback_name="eval",
    )
    model = torch.nn.Linear(2, 1)
    model.train()

    callback(0, model, None, 0, 0.0, "cuda:1", 1, False)

    assert calls == []
    assert callback.pbar is None
    assert model.training


def test_rank_zero_evaluation_uses_resolved_runtime_device():
    calls = []

    def make_estimator(**kwargs):
        calls.append(kwargs)
        return FakeEstimator()

    callback = EvalCATE(
        eval_datasets={"tiny": OneEvaluationDataset()},
        cate_estimator_partial=make_estimator,
        callback_name="eval",
    )
    model = torch.nn.Linear(2, 1)

    callback(0, model, None, 0, 0.0, "cuda:3", 0, False)
    callback.pbar.close()

    assert calls[0]["device"] == "cuda:3"
    assert calls[0]["icl_model"] is model
    assert model.training
