# lob-microstructure

**Market microstructure research toolkit**: a price-time-priority **limit order book** rebuilt from an exchange-style message stream, an **agent-based order-flow simulator** with informed traders, **order-flow imbalance** features, and **execution algorithms** (TWAP, VWAP, Almgren–Chriss) evaluated by implementation shortfall.

## Components

| Module | What it does |
|---|---|
| `book.py` | integer-tick order book with FIFO queues per level; limit (including marketable), market, partial and full cancel; `replay()` rebuilds the book from `A`/`C`/`M` messages, ITCH-style |
| `simulate.py` | Poisson event flow: liquidity providers quote around the mid, random cancels, noise market orders, and **informed traders** who know a latent fair value (a random walk) |
| `features.py` | spread, L1 and L5 depth imbalance, **microprice**, and **order-flow imbalance (OFI)** as defined by Cont, Kukanov & Stoikov (2014), sampled on a fixed time grid |
| `execution.py` | TWAP, VWAP against a U-shaped volume profile, and the **Almgren–Chriss** optimal trajectory `x_k = X·sinh(κ(T−t_k))/sinh(κT)`, executed as child market orders against the live book |

## Run

```bash
pip install -e ".[dev]"
python examples/research.py
pytest
```

```
replayed 13,474 messages -> book matches live state; 2,983 trades

Δmid ~ OFI (same interval): beta 0.080 ticks/share, R² 0.31
correlation with NEXT-second Δmid:
  ofi             -0.131
  imb_l1          +0.047
  imb_l5          -0.083
  microprice_dev  +0.015
mean spread 3.53 ticks

algorithm                mean IS (bps)  std (bps)
single market order               2.85       1.33
TWAP                              1.98       6.19
VWAP (U-shaped)                   1.82       5.73
Almgren-Chriss λ=1e-3             2.22       4.26
Almgren-Chriss λ=1e-2             2.15       2.73
```

### Findings

- **OFI explains price moves within the same interval**, with R² ≈ 0.31 and a linear price-impact slope, the empirical regularity documented by Cont et al.
- **Predicting the next second is a different problem.** Lagged OFI is *negatively* correlated with the next move: after strong buying, the mid tends to bounce back as liquidity refills (spread bounce / transient impact). A feature that explains moves is not the same as a feature that predicts them.
- **Execution trades cost against risk.** Splitting the order lowers mean shortfall but adds timing risk (higher standard deviation). Raising Almgren–Chriss risk aversion front-loads the schedule and cuts the standard deviation from 6.2 to 2.7 bps at a small cost in mean. Choosing λ is a business decision about risk appetite, not a free parameter.

The simulator is deliberately simple (no queue-reactive liquidity, no hidden orders, no latency). Its job is to make the mechanisms observable and the research code testable before applying it to real exchange data.

## Tests

Price-time priority and partial fills, marketable limit orders, cancels, book reconstruction from messages matching the live book, OFI explaining contemporaneous moves, and schedule invariants (slices sum to the parent order; risk-averse schedules are front-loaded).

## License

MIT
