"""Load the compiled library before pytest collects the data-vector checks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cocoa_test_utils as u


def pytest_configure():
    """Limit BLAS to one thread and set the requested native OpenMP team."""
    u._cct_freeze.configure_nmodels_threads(u=u, allow_missing=True)
