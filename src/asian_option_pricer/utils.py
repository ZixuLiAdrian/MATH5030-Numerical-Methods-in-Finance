import numpy as np

from .models import AsianOptionParams


def monitoring_times(params: AsianOptionParams) -> np.ndarray:
    """
    Return equally spaced monitoring dates between T1 and T2.
    """

    params.validate()

    T2 = params.averaging_end

    dt = (T2 - params.T1) / params.N

    return params.T1 + np.arange(1, params.N + 1) * dt


def discount_factor(params: AsianOptionParams) -> float:
    return float(np.exp(-params.r * params.T))
