"""Execution algorithms for a large parent order, run against the simulated book."""
from __future__ import annotations

import numpy as np

from .simulate import Market


def twap_schedule(total: int, n: int) -> np.ndarray:
    base = np.full(n, total // n)
    base[: total - base.sum()] += 1
    return base


def vwap_schedule(total: int, volume_profile: np.ndarray) -> np.ndarray:
    w = np.asarray(volume_profile, float) / np.sum(volume_profile)
    raw = np.floor(total * w).astype(int)
    raw[np.argsort(-(total * w - raw))[: total - raw.sum()]] += 1
    return raw


def almgren_chriss_schedule(total: int, n: int, risk_aversion: float, sigma: float, eta: float) -> np.ndarray:
    """Optimal liquidation trajectory x_k = X sinh(κ(T−t_k)) / sinh(κT) with κ = sqrt(λσ²/η).
    Higher risk aversion front-loads the trades; λ → 0 recovers TWAP."""
    if risk_aversion <= 0:
        return twap_schedule(total, n)
    kappa = np.sqrt(risk_aversion * sigma**2 / eta)
    t = np.arange(n + 1)
    holdings = total * np.sinh(kappa * (n - t)) / np.sinh(kappa * n)
    trades = -np.diff(holdings)
    out = np.floor(trades).astype(int)
    out[np.argsort(-(trades - out))[: total - out.sum()]] += 1
    return out


def execute(schedule: np.ndarray, side: str = "B", interval: float = 5.0, seed: int = 0, warmup: float = 30.0) -> dict:
    """Send each slice as a market order at the start of its interval. Cost = implementation shortfall
    versus the arrival mid, in basis points (positive = worse than arrival)."""
    m = Market(seed=seed)
    m.run_until(warmup)
    while m.book.mid() is None:  # make sure both sides are quoted at arrival
        m.step()
    arrival = m.book.mid()
    paid, filled = 0.0, 0
    for k, q in enumerate(schedule):
        m.run_until(warmup + k * interval)
        if q > 0:
            for tr in m._market(side, int(q), tag="algo"):
                paid += tr.price * tr.qty
                filled += tr.qty
    avg = paid / max(filled, 1)
    sign = 1 if side == "B" else -1
    return {"filled": filled, "avg_price": avg, "arrival": arrival,
            "shortfall_bps": sign * (avg - arrival) / arrival * 1e4}
