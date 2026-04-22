"""Print the default benchmark suite for the canonical test case.

Run: ``python experiments/run_benchmarks.py``.
"""
from __future__ import annotations

from pprint import pprint

from _common import _ROOT  # noqa: F401  (ensures sys.path injection)
from asian_option_pricer.benchmarks import benchmark_suite


if __name__ == "__main__":
    pprint(benchmark_suite())
