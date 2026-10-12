"""
Is the Nifty 50 a random walk?

Usage:  python nifty_random_walk_tests.py [data_folder] [output_folder] [start_date end_date]
Defaults: data folder = ./data, output folder = current folder.

Loads the yearly NSE "NIFTY_50_Historical_PR_*.csv" files, builds daily log returns
r_t = ln(P_t / P_{t-1}), and runs three standard checks:
  1. Autocorrelation of returns (lags 1..10)
  2. Lo-MacKinlay variance ratio test, VR(q) for q = 2, 5, 10, 20
  3. Runs test on the sign of returns
It also overlays a simulated Gaussian random walk with the same mean and volatility.

If a calendar year is missing, the data is split into contiguous blocks and the
tests use the longest block (so we never compute a "return" across a gap).
"""
import glob
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

# ---------- 1. load ----------
files = sorted(glob.glob(f"{FOLDER}/NIFTY_50_Historical_PR_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = [c.strip() for c in df.columns]
df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d %b %Y")
df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ""), errors="coerce")
df = df.dropna(subset=["Close"]).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
if len(sys.argv) > 4:  # optional: restrict to a sub-period, e.g.  ... 2021-01-01 2026-12-31
    df = df[(df["Date"] >= sys.argv[3]) & (df["Date"] <= sys.argv[4])].reset_index(drop=True)
print(f"Files loaded: {len(files)}   Trading days: {len(df)}   "
      f"{df['Date'].min().date()} to {df['Date'].max().date()}")

# ---------- 2. split into contiguous blocks (gap > 7 calendar days = missing data) ----------
gap = df["Date"].diff().dt.days.fillna(0) > 7
df["block"] = gap.cumsum()
blocks = df.groupby("block")
print("Contiguous blocks:")
for b, g in blocks:
    print(f"  block {b}: {g['Date'].min().date()} to {g['Date'].max().date()}  ({len(g)} days)")
main = max((g for _, g in blocks), key=len).reset_index(drop=True)
print(f"Using the longest block: {main['Date'].min().date()} to {main['Date'].max().date()}\n")

x = np.log(main["Close"].values)        # log prices
r = np.diff(x)                           # daily log returns
n = len(r)
mu, sd = r.mean(), r.std(ddof=1)

# ---------- 3. basic stats ----------
print("=== Basic statistics of daily log returns ===")
print(f"n = {n}, mean = {mu:.5f}, std = {sd:.5f}  (annualised vol ~ {sd*np.sqrt(252):.1%})")
print(f"skewness = {stats.skew(r):.3f}, excess kurtosis = {stats.kurtosis(r):.3f}  (0 for a normal)")

# ---------- 4. autocorrelation ----------
def acf(v, k):
    v = v - v.mean()
    return (v[k:] * v[:-k]).sum() / (v * v).sum()

band = 1.96 / np.sqrt(n)
lags = np.arange(1, 11)
rho = np.array([acf(r, k) for k in lags])
print(f"\n=== Autocorrelation of returns (95% band for pure noise = +/-{band:.3f}) ===")
for k, v in zip(lags, rho):
    flag = "  <-- outside band" if abs(v) > band else ""
    print(f"lag {k:2d}: {v:+.3f}{flag}")
lb = n * (n + 2) * np.sum(rho ** 2 / (n - lags))
print(f"Ljung-Box (10 lags): Q = {lb:.2f}, p = {1 - stats.chi2.cdf(lb, 10):.3f}")

# ---------- 5. Lo-MacKinlay variance ratio ----------
def variance_ratio(x, q):
    """x = log prices (length n+1). Returns VR, heteroskedasticity-robust z*, p-value."""
    r = np.diff(x); n = len(r); m = r.mean()
    s_a = ((r - m) ** 2).sum() / (n - 1)
    rq = x[q:] - x[:-q]
    mm = q * (n - q + 1) * (1 - q / n)
    s_c = ((rq - q * m) ** 2).sum() / mm
    vr = s_c / s_a
    d = (r - m) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = (d[j:] * d[:-j]).sum() / (d.sum() ** 2)
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta)
    return vr, z, 2 * (1 - stats.norm.cdf(abs(z)))

print("\n=== Variance ratio test (VR = 1 for a random walk) ===")
vr_rows = []
for q in (2, 5, 10, 20):
    vr, z, p = variance_ratio(x, q)
    vr_rows.append((q, vr, z, p))
    print(f"q = {q:2d}: VR = {vr:.3f}, z* = {z:+.2f}, p = {p:.3f}")

# ---------- 6. runs test on signs ----------
sign = r > 0
runs = 1 + (sign[1:] != sign[:-1]).sum()
n1, n2 = sign.sum(), (~sign).sum()
exp_runs = 1 + 2 * n1 * n2 / (n1 + n2)
var_runs = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
zr = (runs - exp_runs) / np.sqrt(var_runs)
print("\n=== Runs test on up/down days ===")
print(f"up days = {n1}, down days = {n2}, runs = {runs}, expected = {exp_runs:.1f}, "
      f"z = {zr:+.2f}, p = {2 * (1 - stats.norm.cdf(abs(zr))):.3f}")

# ---------- 7. how unusual are the VR values? compare with simulated random walks ----------
rng = np.random.default_rng(7)
sims = 2000
vr_sim = {q: [] for q in (2, 5, 10, 20)}
for _ in range(sims):
    xs = np.concatenate([[0], np.cumsum(rng.normal(mu, sd, n))])
    for q in vr_sim:
        vr_sim[q].append(variance_ratio(xs, q)[0])
print("\n=== Simulated Gaussian random walks (same mean/vol): spread of VR(q) ===")
for q, vr, z, p in vr_rows:
    lo, hi = np.percentile(vr_sim[q], [2.5, 97.5])
    print(f"q = {q:2d}: real VR = {vr:.3f}; 95% of simulated random walks give {lo:.3f} to {hi:.3f}")

# ---------- 8. figures ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(main["Date"], x - x[0], label="Nifty 50 (log price)", lw=1.3)
for i in range(3):
    xs = np.concatenate([[0], np.cumsum(rng.normal(mu, sd, n))])
    ax[0].plot(main["Date"], xs, lw=0.8, alpha=0.6, label="simulated random walk" if i == 0 else None, color="grey")
ax[0].set_title("Real log price vs simulated random walks"); ax[0].legend(); ax[0].set_ylabel("log price change")

ax[1].bar(lags, rho, color="#2f5496")
ax[1].axhline(band, color="r", ls="--"); ax[1].axhline(-band, color="r", ls="--")
ax[1].set_title("Autocorrelation of daily returns"); ax[1].set_xlabel("lag (days)")

qs = [row[0] for row in vr_rows]
ax[2].plot(qs, [row[1] for row in vr_rows], "o-", color="#2f5496", label="real VR(q)")
ax[2].fill_between(qs, [np.percentile(vr_sim[q], 2.5) for q in qs],
                   [np.percentile(vr_sim[q], 97.5) for q in qs], color="grey", alpha=0.3,
                   label="95% range, random walk")
ax[2].axhline(1, color="k", lw=0.8)
ax[2].set_title("Variance ratio VR(q)"); ax[2].set_xlabel("q (days)"); ax[2].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/nifty_random_walk_tests.png", dpi=140)
print(f"\nFigure saved to {OUT}/nifty_random_walk_tests.png")
