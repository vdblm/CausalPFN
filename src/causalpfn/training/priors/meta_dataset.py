from abc import ABC, abstractmethod

from torch.utils.data import IterableDataset

from causalpfn.models.utils import pad_x
from causalpfn.training.priors.utils import PriorGenerationError


class MetaDataset(IterableDataset, ABC):
    """
    Data used for training: for each call, it will return a set of samples from a random data-generating process.
    It will output variables like covariates X, treatments T, CEPOs E[Y_t|X], potential outcomes Y_t,
    observed outcomes Y, etc; anything needed for training the in-context model.
    """

    def __init__(self, name: str, post_padding_n_cols: int):
        self.name = name

        # padding the number of columns so that tables can be batched together for training
        self.post_padding_n_cols = post_padding_n_cols

        # the set of all of the warnings that have been raised
        self.prior_generation_warnings = set()

    @abstractmethod
    def get_sample(self) -> dict:
        raise NotImplementedError("Implement generating samples separately")

    def __iter__(self):
        while True:
            try:
                sample = self.get_sample()
            except PriorGenerationError as error:
                # This warning can be emitted once per worker.
                message = str(error)
                if message not in self.prior_generation_warnings:
                    self.prior_generation_warnings.add(message)
                    print(f"[Prior Warning] {message}")
                continue

            sample["X"] = pad_x(sample["X"], num_features=self.post_padding_n_cols)
            yield sample
