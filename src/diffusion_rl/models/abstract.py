from typing import Protocol


def SDE(Protocol):
    a: float  # diffusion coefficient

    def drift(xt, ts):
        """
        
        Args:
            xt: (n, dim) vector of xt
            ts: (n, 1) vector of t

        Returns:
            drift: (n, dim) vector of drift
        """
        ...