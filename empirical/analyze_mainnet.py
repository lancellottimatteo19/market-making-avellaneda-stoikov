"""Lead-lag between Binance and Hyperliquid, toxicity of Hyperliquid trades and model calibration.

Reads the hourly files written by ``extract_mainnet.py`` and writes small CSV tables to ``empirical/results``:

* ``calibration.csv``: hours of data, Hyperliquid quoted spread, delivery delays to the recording server,
  volatility of Binance, and the parameters of the price model of ``two_venue_sim.py`` (diffusion plus
  information events) estimated from one-second Binance returns;
* ``lead_lag.csv``: correlation between 100 ms Binance returns and Hyperliquid returns ``lag`` seconds later,
  on the server's receive clock and on the exchanges' own clocks;
* ``gap_closure.csv``: how much of the gap between the two mids each venue closes over the next seconds;
* ``toxicity.csv``: markout of the liquidity provider on Hyperliquid trades, by how much Binance moved in the
  taker's direction in the second before the trade;
* ``maker_markout.csv``: average markout of the liquidity provider over all trades.

    python empirical/analyze_mainnet.py <cache folder> [results folder]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEP = 100                                              # ms, grid of the price series
COINS = ("BTC", "ETH", "HYPE")
LAGS = range(-20, 21)                                   # in grid steps
GAP_HORIZONS = (5, 10, 20, 50, 100)                     # in grid steps
MARKOUT_S = (1, 10, 60)
BUCKETS = [-np.inf, -3, -1, 1, 3, np.inf]
BUCKET_LABELS = ["against taker, >3 bp", "against taker, 1-3 bp", "flat (within 1 bp)", "with taker, 1-3 bp", "with taker, >3 bp"]


def at(t_src, v_src, t):
    """Last known value at each time t (forward fill), NaN before the first observation."""
    i = np.searchsorted(t_src, t, side="right") - 1
    return np.where(i >= 0, v_src[np.clip(i, 0, None)], np.nan)


def sessions(files, coin, clock):
    """Per hour: 100 ms grid, log mids of both venues, Hyperliquid spread, trades and delivery delays."""
    tc = 0 if clock == "receive" else 1
    for f in files:
        z = np.load(f)
        hl, bn, tr = z[f"{coin}_hl"], z[f"{coin}_bn"], z[f"{coin}_tr"]
        if len(hl) < 1000 or len(bn) < 1000:
            continue
        hl = hl[np.argsort(hl[:, tc], kind="stable")]
        bn = bn[np.argsort(bn[:, tc], kind="stable")]
        delay_hl, delay_bn = hl[:, 0] - hl[:, 1], bn[:, 0] - bn[:, 1]
        trades = tr[:, [1, 2, 3, 4]].copy()             # exchange time, price, size, side
        if tc == 0:                                     # place trades on the receive clock with the typical delay
            trades[:, 0] += np.median(delay_hl)
        start, end = max(hl[0, tc], bn[0, tc]) + 5_000, min(hl[-1, tc], bn[-1, tc])
        grid = np.arange(start, end, STEP)
        hl_mid, bn_mid = (hl[:, 2] + hl[:, 3]) / 2, (bn[:, 2] + bn[:, 3]) / 2
        yield dict(grid=grid, lh=np.log(at(hl[:, tc], hl_mid, grid)), lb=np.log(at(bn[:, tc], bn_mid, grid)),
                   spread=at(hl[:, tc], (hl[:, 3] - hl[:, 2]) / hl_mid * 1e4, grid),
                   quotes=np.column_stack([hl[:, tc], hl[:, 2], hl[:, 3]]), trades=trades,
                   delay_hl=delay_hl, delay_bn=delay_bn)


def corr(pairs):
    a = np.concatenate([p[0] for p in pairs]); b = np.concatenate([p[1] for p in pairs])
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def slope(pairs):
    y = np.concatenate([p[0] for p in pairs]); x = np.concatenate([p[1] for p in pairs])
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0]


def price_model(r1, k=4.0):
    """Diffusion plus events from one-second returns in bp.

    The diffusion is the standard deviation of the returns after iteratively excluding those beyond ``k``
    standard deviations (more than half of the one-second returns are exactly zero, so the median absolute
    deviation is zero and cannot be used); the excluded tail gives the rate and size of the events.
    """
    sigma = r1.std()
    for _ in range(50):
        new = r1[np.abs(r1) <= k * sigma].std()
        if abs(new - sigma) < 1e-9:
            break
        sigma = new
    tail = np.abs(r1) > k * sigma
    rate = tail.mean()                                  # events per second
    size = np.sqrt(max(np.mean(r1[tail] ** 2) - sigma ** 2, 0.0)) if tail.any() else 0.0
    return dict(sigma_bp_sqrt_s=sigma, event_rate_per_s=rate, event_sd_bp=size,
                total_sd_bp_1s=r1.std())


def analyse(files, clock):
    lead_lag, gap, tox, cal = [], [], [], []
    for coin in COINS:
        xc = {lag: [] for lag in LAGS}
        fut = {h: ([], []) for h in GAP_HORIZONS}
        r1s, spreads, d_hl, d_bn, hours, frames = [], [], [], [], 0.0, []
        for s in sessions(files, coin, clock):
            grid, lh, lb = s["grid"], s["lh"], s["lb"]
            hours += len(grid) * STEP / 3.6e6
            spreads.append(np.nanmedian(s["spread"]))
            d_hl.append(s["delay_hl"]); d_bn.append(s["delay_bn"])
            rh, rb = np.diff(lh) * 1e4, np.diff(lb) * 1e4
            for lag in LAGS:                            # lag > 0: Binance first, Hyperliquid lag steps later
                xc[lag].append((rb[:len(rb) - lag], rh[lag:]) if lag >= 0 else (rb[-lag:], rh[:len(rh) + lag]))
            r1s.append(((lb[10:] - lb[:-10]) * 1e4)[::10])
            g = (lb - lh) * 1e4                         # gap net of the session's slowly moving basis
            g = g - pd.Series(g).rolling(3000, min_periods=600, center=True).median().to_numpy()
            for h in GAP_HORIZONS:
                fut[h][0].append(((lh[h:] - lh[:-h]) * 1e4, g[:len(g) - h]))
                fut[h][1].append(((lb[h:] - lb[:-h]) * 1e4, g[:len(g) - h]))
            t, px, sz, side = s["trades"].T
            keep = (t > grid[0] + 1000) & (t < grid[-1] - 60_000)
            t, px, sz, side = t[keep], px[keep], sz[keep], side[keep]
            pre = (at(grid, lb, t) - at(grid, lb, t - 1000)) * 1e4 * side
            q = s["quotes"]
            frame = dict(coin=coin, pre=pre, notional=px * sz, side=side, px=px,
                         bid=at(q[:, 0], q[:, 1], t - 1), ask=at(q[:, 0], q[:, 2], t - 1))
            for h in MARKOUT_S:                         # liquidity provider's markout in bp
                frame[f"m{h}"] = -side * (np.exp(at(grid, lh, t + h * 1000)) - px) / px * 1e4
            frames.append(pd.DataFrame(frame))
        for lag, pairs in xc.items():
            lead_lag.append(dict(clock=clock, coin=coin, lag_s=lag * STEP / 1000, corr=corr(pairs)))
        for h, (hl_pairs, bn_pairs) in fut.items():
            b_hl, b_bn = slope(hl_pairs), slope(bn_pairs)
            gap.append(dict(clock=clock, coin=coin, horizon_s=h * STEP / 1000, beta_HL=b_hl, beta_Binance=b_bn,
                            share_HL=b_hl / (b_hl - b_bn)))
        r1 = np.concatenate(r1s); r1 = r1[np.isfinite(r1)]
        dh, db = np.concatenate(d_hl), np.concatenate(d_bn)
        cal.append(dict(clock=clock, coin=coin, hours=hours, sessions=len(spreads),
                        hl_spread_median_bp=np.median(spreads),
                        delivery_HL_ms_median=np.median(dh), delivery_HL_ms_p90=np.percentile(dh, 90),
                        delivery_Binance_ms_median=np.median(db), delivery_Binance_ms_p90=np.percentile(db, 90),
                        share_1s_moves_over_3bp=np.mean(np.abs(r1) > 3), share_1s_moves_over_5bp=np.mean(np.abs(r1) > 5),
                        **price_model(r1)))
        tox.append(pd.concat(frames, ignore_index=True))
    return pd.DataFrame(lead_lag), pd.DataFrame(gap), pd.DataFrame(cal), pd.concat(tox, ignore_index=True)


def toxicity_tables(t):
    t = t[np.isfinite(t.pre) & np.isfinite(t.m60)].copy()
    side_check = {coin: dict(buy_at_ask=((g.side > 0) & (g.px >= g.ask)).sum() / (g.side > 0).sum(),
                             sell_at_bid=((g.side < 0) & (g.px <= g.bid)).sum() / (g.side < 0).sum())
                  for coin, g in t.groupby("coin")}
    t["binance_before"] = pd.cut(t.pre, BUCKETS, labels=BUCKET_LABELS)
    rows = []
    for (coin, bucket), g in t.groupby(["coin", "binance_before"], observed=True):
        row = dict(coin=coin, binance_before=bucket, trades=len(g), volume_usd=g.notional.sum())
        for h in MARKOUT_S:
            row[f"median_{h}s"] = g[f"m{h}"].median()
            row[f"mean_{h}s"] = g[f"m{h}"].mean()
            row[f"volume_weighted_{h}s"] = np.average(g[f"m{h}"], weights=g.notional)
        rows.append(row)
    tox = pd.DataFrame(rows)
    tox["volume_share"] = tox.volume_usd / tox.groupby("coin").volume_usd.transform("sum")
    overall = pd.DataFrame([dict(coin=coin, trades=len(g), volume_usd=g.notional.sum(),
                                 **{f"median_{h}s": g[f"m{h}"].median() for h in MARKOUT_S},
                                 **{f"volume_weighted_{h}s": np.average(g[f"m{h}"], weights=g.notional) for h in MARKOUT_S},
                                 **side_check[coin])
                            for coin, g in t.groupby("coin")])
    return tox, overall


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in cache.glob("stream_*.npz") if ".tmp" not in p.name)
    results = {clock: analyse(files, clock) for clock in ("receive", "exchange")}
    pd.concat([results[c][0] for c in results]).to_csv(out / "lead_lag.csv", index=False)
    pd.concat([results[c][1] for c in results]).to_csv(out / "gap_closure.csv", index=False)
    results["receive"][2].to_csv(out / "calibration.csv", index=False)
    tox, overall = toxicity_tables(results["receive"][3])
    tox.to_csv(out / "toxicity.csv", index=False)
    overall.to_csv(out / "maker_markout.csv", index=False)
    pd.set_option("display.width", 220)
    print(results["receive"][2].round(3).to_string())
    print(overall.round(2).to_string())
    print(tox[["coin", "binance_before", "volume_share", "median_1s", "median_10s", "median_60s"]].round(2).to_string())
