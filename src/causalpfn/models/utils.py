import torch


def pad_x(X: torch.Tensor, num_features=100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    if n_features > num_features:
        raise ValueError(f"Cannot pad {n_features} features to the smaller size {num_features}.")
    feature_padding = torch.zeros((*X.shape[:-1], num_features - n_features), dtype=X.dtype, device=X.device)
    return torch.cat([X, feature_padding], dim=-1)
