import warnings

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs (deselect with -m 'not slow')")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
