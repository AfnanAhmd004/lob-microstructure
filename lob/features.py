"""Microstructure features sampled on a fixed time grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import Market


def microprice(bb, ba, qb, qa):
    """Mid weighted towards the side with less queue: where the next trade is more likely to move the price."""
    return (ba * qb + bb * qa) / (qb + qa)


def collect(market: Market, duration: float, dt: float = 1.0) -> pd.DataFrame:
    """Run the market and record per-interval features, including Cont–Kukanov–Stoikov order-flow imbalance."""
    rows, state = [], {"prev": None, "ofi": 0.0, "buy_vol": 0, "sell_vol": 0, "n_trades": 0}
    next_t = dt

    def on_event(m: Market):
        nonlocal next_t
        b = m.book
        bb, ba = b.best_bid(), b.best_ask()
        if bb is None or ba is None:
            return
        qb, qa = b.level_qty("B", bb), b.level_qty("S", ba)
        if state["prev"] is not None:
            pbb, pqb, pba, pqa = state["prev"]
            e = (qb if bb >= pbb else 0) - (pqb if bb <= pbb else 0) - (qa if ba <= pba else 0) + (pqa if ba >= pba else 0)
            state["ofi"] += e
        state["prev"] = (bb, qb, ba, qa)
        while m.t >= next_t:
            dq = b.depth("B", 5), b.depth("S", 5)
            depth_b, depth_a = sum(q for _, q in dq[0]), sum(q for _, q in dq[1])
            rows.append({"t": next_t, "mid": (bb + ba) / 2, "spread": ba - bb, "microprice": microprice(bb, ba, qb, qa),
                         "imb_l1": (qb - qa) / (qb + qa), "imb_l5": (depth_b - depth_a) / (depth_b + depth_a),
                         "ofi": state["ofi"], "fair": m.fair})
            state["ofi"] = 0.0
            next_t += dt

    market.run_until(duration, on_event)
    df = pd.DataFrame(rows).set_index("t")
    df["dmid_next"] = df["mid"].shift(-1) - df["mid"]  # what we try to predict, known only after the interval
    df["dmid"] = df["mid"].diff()
    return df


def ofi_regression(df: pd.DataFrame) -> dict[str, float]:
    """Contemporaneous Δmid ~ OFI (Cont et al. 2014) and predictive Δmid(next) ~ features at t."""
    d = df.dropna()
    beta = np.polyfit(d["ofi"], d["dmid"], 1)[0]
    r2_contemp = np.corrcoef(d["ofi"], d["dmid"])[0, 1] ** 2
    out = {"ofi_beta": float(beta), "r2_contemporaneous": float(r2_contemp)}
    for f in ["ofi", "imb_l1", "imb_l5"]:
        out[f"corr_next_{f}"] = float(np.corrcoef(d[f], d["dmid_next"])[0, 1])
    out["corr_next_microprice_dev"] = float(np.corrcoef(d["microprice"] - d["mid"], d["dmid_next"])[0, 1])
    return out
