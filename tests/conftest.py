"""Make every test deterministic regardless of the order they run in.

The augmentation tests draw from the global random module, so a test's
behavior otherwise depends on how many random numbers the tests before it
happened to consume. That produces failures that reproduce in a full run and
vanish when the test is run alone, which is the most expensive kind to
diagnose and exactly what a benchmark repository should not have.
"""

import random

import numpy as np
import pytest
import torch

from segmentation_benchmark._config import SEED


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Reseed before each test, so order cannot change any outcome."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    yield
