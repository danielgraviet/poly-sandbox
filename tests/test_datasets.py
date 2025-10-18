import pytest
from hf_datasets import mbpp_loader

@pytest.mark.hf
def test_load_mbpp():
    ds = mbpp_loader.load_mbpp()
    sample = ds[0]
    assert len(ds) == 120
    assert sample is not None