"""Event-driven order-flow simulator: zero-intelligence liquidity plus informed traders.

Liquidity providers post limit orders at random distances from the opposite best quote and cancel at
random; noise traders send random market orders; informed traders know a latent fair value (a random
walk) and buy when it is above the mid and sell when below. Informed flow is what makes order-flow
features predictive of short-term price moves, which is the effect microstructure research tries to
measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .book import OrderBook


@dataclass
class FlowParams:
    limit_rate: float = 10.0      # limit orders per second
    cancel_rate: float = 9.0
    noise_mkt_rate: float = 2.0
    informed_rate: float = 2.0
    fair_vol: float = 1.0         # ticks per sqrt(second) of the latent fair value
    max_depth_ticks: int = 8
    qty_range: tuple[int, int] = (1, 10)


class Market:
    def __init__(self, seed: int = 0, params: FlowParams = FlowParams(), start_price: int = 10_000):
        self.rng = np.random.default_rng(seed)
        self.p = params
        self.book = OrderBook()
        self.t, self.fair, self.next_id = 0.0, float(start_price), 1
        self.messages: list[tuple] = []
        for k in range(1, 6):  # seed both sides
            for _ in range(3):
                self._add("B", start_price - k, int(self.rng.integers(*self.p.qty_range)))
                self._add("S", start_price + k, int(self.rng.integers(*self.p.qty_range)))

    def _add(self, side, price, qty):
        oid = self.next_id; self.next_id += 1
        self.messages.append(("A", self.t, oid, side, price, qty))
        self.book.add_limit(oid, side, price, qty, self.t)

    def _market(self, side, qty, tag="noise"):
        self.messages.append(("M", self.t, side, qty))
        return self.book.market(side, qty, self.t)

    def step(self):
        """Advance to the next event and apply it."""
        p = self.p
        total = p.limit_rate + p.cancel_rate + p.noise_mkt_rate + p.informed_rate
        dt = self.rng.exponential(1 / total)
        self.t += dt
        self.fair += p.fair_vol * np.sqrt(dt) * self.rng.standard_normal()
        u = self.rng.random() * total
        bb, ba = self.book.best_bid(), self.book.best_ask()
        mid = self.book.mid() or self.fair
        qty = int(self.rng.integers(*p.qty_range))
        if bb is None or ba is None or u < p.limit_rate:
            # liquidity providers quote around the current mid, sometimes improving the inside quote
            side = "B" if bb is None else "S" if ba is None else self.rng.choice(["B", "S"])
            d = int(self.rng.geometric(0.4)) - 1
            d = min(d, p.max_depth_ticks)
            price = int(np.floor(mid - 0.5)) - d if side == "B" else int(np.ceil(mid + 0.5)) + d
            self._add(side, price, qty)
        elif u < p.limit_rate + p.cancel_rate:
            if self.book.orders:
                oid = list(self.book.orders)[int(self.rng.integers(len(self.book.orders)))]
                self.messages.append(("C", self.t, oid, None))
                self.book.cancel(oid)
        elif u < p.limit_rate + p.cancel_rate + p.noise_mkt_rate:
            self._market(self.rng.choice(["B", "S"]), qty)
        else:
            if abs(self.fair - mid) > 0.5:
                self._market("B" if self.fair > mid else "S", qty, tag="informed")

    def run_until(self, t_end: float, on_event=None):
        while self.t < t_end:
            self.step()
            if on_event:
                on_event(self)
