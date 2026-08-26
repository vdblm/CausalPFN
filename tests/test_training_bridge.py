import pytest
import torch

from causalpfn.models import InContextModel, TabDPTLongContextModel
from causalpfn.models.utils import pad_x


def test_pad_x_uses_zero_padding_and_preserves_dtype():
    values = torch.ones(2, 2, dtype=torch.float64)
    padded = pad_x(values, num_features=5)
    assert padded.shape == (2, 5)
    assert padded.dtype == values.dtype
    assert torch.equal(padded[:, :2], values)
    assert torch.equal(padded[:, 2:], torch.zeros(2, 3, dtype=values.dtype))


def test_pad_x_rejects_smaller_target():
    with pytest.raises(ValueError, match="smaller size"):
        pad_x(torch.ones(2, 3), num_features=2)


def test_in_context_model_exposes_optimizer_groups():
    inner = TabDPTLongContextModel(
        dropout=0.0,
        n_out=2,
        nhead=1,
        nhid=8,
        ninp=4,
        nlayers=1,
        num_features=4,
        nbins=8,
    )
    config = {"model": {"nbins": 8, "max_num_features": 4}}
    model = InContextModel(inner, config)
    groups = model.get_param_groups()
    assert len(groups) == 2
    assert groups[1]["weight_decay"] == 0.0


def test_in_context_model_prepares_input_without_instance_closure():
    inner = TabDPTLongContextModel(
        dropout=0.0,
        n_out=2,
        nhead=1,
        nhid=8,
        ninp=4,
        nlayers=1,
        num_features=4,
        nbins=8,
    )
    model = InContextModel(inner, {"model": {"nbins": 8, "max_num_features": 4}})
    x = torch.ones(2, 3)
    y = torch.arange(2)

    padded_x, returned_y = model.prepare_input(x, y)

    assert "prepare_input" not in model.__dict__
    assert padded_x.shape == (2, 4)
    assert torch.equal(padded_x[:, -1], torch.zeros(2))
    assert returned_y is y
