"""1) Rebuild the book from messages. 2) Test order-flow features. 3) Compare execution algorithms."""
import numpy as np

from lob import Market, almgren_chriss_schedule, collect, execute, ofi_regression, replay, twap_schedule, vwap_schedule

# 1. book reconstruction from the message log
m = Market(seed=1)
m.run_until(600)
rebuilt = replay(m.messages)
assert rebuilt.depth("B", 10) == m.book.depth("B", 10) and rebuilt.depth("S", 10) == m.book.depth("S", 10)
print(f"replayed {len(m.messages):,} messages -> book matches live state; {len(m.book.trades):,} trades\n")

# 2. order-flow imbalance
df = collect(Market(seed=2), duration=3600, dt=1.0)
res = ofi_regression(df)
print(f"Δmid ~ OFI (same interval): beta {res['ofi_beta']:.3f} ticks/share, R² {res['r2_contemporaneous']:.2f}")
print("correlation with NEXT-second Δmid:")
for k in ["ofi", "imb_l1", "imb_l5", "microprice_dev"]:
    print(f"  {k:<16}{res['corr_next_' + k]:+.3f}")
print(f"mean spread {df['spread'].mean():.2f} ticks\n")

# 3. execution of a 400-share buy over 20 slices x 5 s
Q, N = 400, 20
profile = 1 + 0.8 * np.cos(np.linspace(0, 2 * np.pi, N))  # U-shaped expected volume
algos = {"single market order": np.r_[Q, np.zeros(N - 1, int)], "TWAP": twap_schedule(Q, N),
         "VWAP (U-shaped)": vwap_schedule(Q, profile),
         "Almgren-Chriss λ=1e-3": almgren_chriss_schedule(Q, N, 1e-3, sigma=3.0, eta=0.05),
         "Almgren-Chriss λ=1e-2": almgren_chriss_schedule(Q, N, 1e-2, sigma=3.0, eta=0.05)}
print(f"{'algorithm':<24}{'mean IS (bps)':>14}{'std (bps)':>11}")
for name, sched in algos.items():
    costs = [execute(sched, seed=s)["shortfall_bps"] for s in range(40)]
    print(f"{name:<24}{np.mean(costs):>14.2f}{np.std(costs):>11.2f}")
