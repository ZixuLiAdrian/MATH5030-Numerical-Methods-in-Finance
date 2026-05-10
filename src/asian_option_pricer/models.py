from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class AsianOptionParams:
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    N: int
    option_type: Literal["call", "put"] = "call"

    # NEW
    T1: float = 0.0
    T2: Optional[float] = None

    @property
    def averaging_end(self) -> float:
        return self.T if self.T2 is None else self.T2

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

        T2 = self.averaging_end

        if self.T1 < 0:
            raise ValueError("T1 must be non-negative.")

        if T2 <= self.T1:
            raise ValueError("T2 must be greater than T1.")

        if T2 > self.T:
            raise ValueError("T2 cannot exceed maturity T.")
