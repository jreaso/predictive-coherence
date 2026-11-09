import pytest
import torch


@pytest.fixture(autouse=True)
def seed_and_float64():
    torch.manual_seed(0)
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
