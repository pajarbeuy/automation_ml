"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

# Add ml_engine package to sys.path for all tests
_ML_ENGINE = str(Path(__file__).parent / "packages" / "ml_engine")
if _ML_ENGINE not in sys.path:
    sys.path.insert(0, _ML_ENGINE)
