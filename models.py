from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AsianOptionParams:
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    N: int
    option_type: Literal["call", "put"] = "call"

    def validate(self) -> None:
        if self.S0 <= 0 or self.K <= 0:
            raise ValueError("S0 and K must be positive.")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")
