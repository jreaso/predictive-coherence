import pytest
import torch
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True)
def seed_and_float64():
    torch.manual_seed(0)
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    # One thread, as in a worker process: on problems this small the learners that thread
    # through OpenMP spend far longer synchronising than working.
    with threadpool_limits(1):
        yield
    torch.set_default_dtype(dtype)
