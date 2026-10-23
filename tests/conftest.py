"""Shared fixtures for the offline test suite.

Every test runs on synthetic data: no files under data/, no network, no API keys.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def quiet_logger():
    """Plain logger with no file handler (src.utils.logger.get_logger writes to data/metadata)."""
    logger = logging.getLogger("tests")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
