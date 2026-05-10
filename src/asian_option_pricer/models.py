"""Dataclasses and parameter validation for Asian option pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class AsianOptionParams:
    """
    Parameters for discretely monitored Asian options under GBM.

    Supports generalized averaging windows [T1, T2].

    Default behavior:
        T1 = 0
        T2 = T
    """

    # underlying
    S0: float

    # strike
    K: float

    # risk-free rate
    r: float

    # volatility
    sigma: float

    # maturity
    T: float

    # number of monitoring dates
    N: int

    # option style
    option_type: Literal["call", "put"] = "call"

    # averaging window start
    T1: float = 0.0

    # averaging window end
    T2: Optional[float] = None

    @property
    def averaging_end(self) -> float:
        """
        Return effective averaging end time.

        If T2 is None:
            averaging ends at maturity T.
        """

        return (
            self.T
            if self.T2 is None
            else self.T2
        )

    def validate(self) -> None:
        """
        Validate all model parameters.
        """

        # positivity checks
        if self.S0 <= 0:
            raise ValueError(
                "S0 must be positive."
            )

        if self.K <= 0:
            raise ValueError(
                "K must be positive."
            )

        if self.T <= 0:
            raise ValueError(
                "T must be positive."
            )

        if self.N <= 0:
            raise ValueError(
                "N must be positive."
            )

        # volatility can be zero
        if self.sigma < 0:
            raise ValueError(
                "sigma must be non-negative."
            )

        # option type
        if self.option_type not in {
            "call",
            "put",
        }:
            raise ValueError(
                "option_type must be 'call' or 'put'."
            )

        T2 = self.averaging_end

        # averaging window validation
        if self.T1 < 0:
            raise ValueError(
                "T1 must be non-negative."
            )

        if T2 <= self.T1:
            raise ValueError(
                "T2 must be greater than T1."
            )

        if T2 > self.T:
            raise ValueError(
                "T2 cannot exceed maturity T."
            )
