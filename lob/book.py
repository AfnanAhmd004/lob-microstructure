"""Limit order book with price-time priority, built from an exchange-style message stream."""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

from sortedcontainers import SortedDict


@dataclass
class Trade:
    ts: float
    price: int
    qty: int
    aggressor: str  # "B" buyer-initiated, "S" seller-initiated
    maker_id: int


class OrderBook:
    """Prices are integer ticks. Bids are kept highest-first, asks lowest-first.
    Each price level is a FIFO queue (price-time priority)."""

    def __init__(self):
        self.bids: SortedDict = SortedDict(lambda p: -p)
        self.asks: SortedDict = SortedDict()
        self.orders: dict[int, tuple[str, int]] = {}  # id -> (side, price)
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------ views
    def best_bid(self):
        return next(iter(self.bids), None)

    def best_ask(self):
        return next(iter(self.asks), None)

    def level_qty(self, side: str, price: int) -> int:
        book = self.bids if side == "B" else self.asks
        return sum(book[price].values()) if price in book else 0

    def depth(self, side: str, levels: int = 5) -> list[tuple[int, int]]:
        book = self.bids if side == "B" else self.asks
        return [(p, sum(q.values())) for p, q in list(book.items())[:levels]]

    def mid(self):
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else (b + a) / 2

    # ------------------------------------------------------------------ messages
    def add_limit(self, oid: int, side: str, price: int, qty: int, ts: float = 0.0) -> list[Trade]:
        """Marketable limit orders match first; any remainder rests on the book."""
        trades = self._match(side, qty, ts, limit=price)
        remaining = qty - sum(t.qty for t in trades)
        if remaining > 0:
            book = self.bids if side == "B" else self.asks
            book.setdefault(price, OrderedDict())[oid] = remaining
            self.orders[oid] = (side, price)
        return trades

    def market(self, side: str, qty: int, ts: float = 0.0) -> list[Trade]:
        return self._match(side, qty, ts, limit=None)

    def cancel(self, oid: int, qty: int | None = None) -> int:
        """Cancel all (or part) of a resting order. Returns the quantity removed."""
        if oid not in self.orders:
            return 0
        side, price = self.orders[oid]
        book = self.bids if side == "B" else self.asks
        level = book[price]
        removed = level[oid] if qty is None else min(qty, level[oid])
        level[oid] -= removed
        if level[oid] == 0:
            del level[oid]
            del self.orders[oid]
        if not level:
            del book[price]
        return removed

    def _match(self, side: str, qty: int, ts: float, limit: int | None) -> list[Trade]:
        opp = self.asks if side == "B" else self.bids
        trades = []
        while qty > 0 and opp:
            price = next(iter(opp))
            if limit is not None and ((side == "B" and price > limit) or (side == "S" and price < limit)):
                break
            level = opp[price]
            while qty > 0 and level:
                maker, avail = next(iter(level.items()))
                fill = min(qty, avail)
                trades.append(Trade(ts, price, fill, side, maker))
                qty -= fill
                level[maker] -= fill
                if level[maker] == 0:
                    del level[maker]
                    del self.orders[maker]
            if not level:
                del opp[price]
        self.trades.extend(trades)
        return trades


def replay(messages, book: OrderBook | None = None, on_event=None) -> OrderBook:
    """Rebuild a book from messages: ("A", ts, id, side, price, qty) add, ("C", ts, id, qty) cancel,
    ("M", ts, side, qty) market order. `on_event(book, msg)` is called after each message."""
    book = book or OrderBook()
    for m in messages:
        kind = m[0]
        if kind == "A":
            _, ts, oid, side, price, qty = m
            book.add_limit(oid, side, price, qty, ts)
        elif kind == "C":
            _, ts, oid, qty = m
            book.cancel(oid, qty)
        elif kind == "M":
            _, ts, side, qty = m
            book.market(side, qty, ts)
        if on_event:
            on_event(book, m)
    return book
