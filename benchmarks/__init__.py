from importlib import import_module

from .polynomial import PolynomialDataset
from .sinusoidal import SinusoidalDataset

_OPTIONAL_DATASETS = {
    "ACIC2016Dataset": (".acic2016", "ACIC2016Dataset"),
    "CriteoDataset": (".criteo", "CriteoDataset"),
    "HillstromDataset": (".hillstrom", "HillstromDataset"),
    "IHDPDataset": (".ihdp", "IHDPDataset"),
    "LentaDataset": (".lenta", "LentaDataset"),
    "MegafonDataset": (".megafon", "MegafonDataset"),
    "RealCauseLalondeCPSDataset": (".realcause", "RealCauseLalondeCPSDataset"),
    "RealCauseLalondePSIDDataset": (".realcause", "RealCauseLalondePSIDDataset"),
    "X5Dataset": (".retail_hero", "X5Dataset"),
}


def __getattr__(name):
    if name not in _OPTIONAL_DATASETS:
        raise AttributeError(name)
    module_name, class_name = _OPTIONAL_DATASETS[name]
    value = getattr(import_module(module_name, __name__), class_name)
    globals()[name] = value
    return value


__all__ = [
    "ACIC2016Dataset",
    "IHDPDataset",  # healthcare dataset
    "RealCauseLalondePSIDDataset",
    "RealCauseLalondeCPSDataset",
    "HillstromDataset",  # marketting dataset
    "CriteoDataset",  # marketting dataset
    "LentaDataset",  # marketting dataset
    "X5Dataset",  # marketting dataset
    "MegafonDataset",  # marketting dataset
    "PolynomialDataset",  # synthetic dataset
    "SinusoidalDataset",  # synthetic dataset
]
