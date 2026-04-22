import numpy as np
from .models import AsianOptionParams


def monitoring_times(params: AsianOptionParams) -> np.ndarray:
    return np.arange(1, params.N + 1) * (params.T / params.N)


def discount_factor(params: AsianOptionParams) -> float:
    return float(np.exp(-params.r * params.T))
