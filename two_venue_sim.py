"""Market making on a lagging venue: Avellaneda-Stoikov quotes against noise traders and arbitrageurs.

The efficient price is discovered on a *leader* venue (for crypto, typically Binance). Our market maker quotes
on a *follower* venue (for example Hyperliquid) and sees the leader with a delay of ``latency`` seconds.
Two kinds of counterparties trade against the quotes:

* noise traders, uninformed, who arrive as in Avellaneda & Stoikov (2008): the probability that a quote at
  distance ``delta`` from the venue's mid (the price we see) is hit within ``dt`` is ``A exp(-k delta) dt``;
* arbitrageurs, who see the leader without delay and take any quote that is stale by more than their
  trading cost ``arb_cost``: they buy our ask when the efficient price is above it, sell to our bid when the
  efficient price is below it.

The efficient price is a Brownian motion plus *information events*: at rate ``event_rate`` the price moves by
``N(0, event_sd)`` spread evenly over ``event_duration`` seconds, as when the leader reprices over a few seconds.
The quotes are the Avellaneda-Stoikov reservation price and spread computed on the price the market maker
sees, optionally with two defences:

* ``widen``: extra half-spread added on both sides;
* the *guard*: when the price seen moved by more than ``guard_bps`` over the last ``guard_window`` seconds,
  the side the arbitrageurs would hit (the ask after a rise, the bid after a fall) is withdrawn for
  ``guard_pause`` seconds.

Prices are in USD on an asset worth 100 at the start, so 0.01 is one basis point, and every order is for one
unit. Every function is vectorised over Monte Carlo paths.
"""
from dataclasses import dataclass

import numpy as np

BP = 0.01                                              # one basis point on a price of 100
WHO = ("noise", "arbitrage")


@dataclass(frozen=True)
class Market:
    """The efficient price and the flow of noise traders and arbitrageurs."""
    s0: float = 100.0
    horizon: float = 1800.0                            # seconds
    dt: float = 0.1                                    # seconds
    sigma: float = 0.005                               # diffusion, price units per sqrt(second): 0.5 bp/sqrt(s)
    event_rate: float = 1 / 30                         # information events per second
    event_sd: float = 0.10                             # size of an event: 10 bp
    event_duration: float = 3.0                        # seconds over which the leader reprices
    noise_A: float = 0.5                               # noise intensity at zero distance, per second and side
    noise_k: float = 100.0                             # decay of the noise intensity, per price unit (1/k = 1 bp)
    arb_cost: float = 0.03                             # arbitrageur's cost (taker fees on both venues): 3 bp
    arbitrage: bool = True                             # False: only noise traders, the textbook case

    @property
    def steps(self):
        return int(round(self.horizon / self.dt))

    @property
    def total_variance(self):
        """Variance of the efficient price per second, diffusion plus events."""
        return self.sigma ** 2 + self.event_rate * self.event_sd ** 2


@dataclass(frozen=True)
class MarketMaker:
    """Avellaneda-Stoikov quotes on the price seen with a delay, with optional defences."""
    gamma: float = 0.05                                # risk aversion
    latency: float = 1.0                               # seconds between the leader and what we see
    max_inventory: int = 10                            # no quote that would push |q| beyond this
    widen: float = 0.0                                 # extra half-spread, price units
    guard_bps: float = np.inf                          # move that triggers the guard, basis points (inf: off)
    guard_window: float = 1.0                          # seconds over which the move is measured
    guard_pause: float = 3.0                           # seconds the exposed side stays withdrawn
    fee_bp: float = 0.0                                # maker fee per fill, bp of notional (negative: rebate)


def efficient_price(market, n_paths, rng):
    """Efficient-price paths, shape (steps + 1, n_paths): diffusion plus information events."""
    n, dt = market.steps, market.dt
    inc = market.sigma * np.sqrt(dt) * rng.standard_normal((n, n_paths))
    ramp = max(1, int(round(market.event_duration / dt)))
    n_events = rng.poisson(market.event_rate * market.horizon, size=n_paths)
    for path in range(n_paths):
        starts = rng.integers(0, n, size=n_events[path])
        sizes = rng.normal(0.0, market.event_sd, size=n_events[path])
        for start, size in zip(starts, sizes):
            inc[start:start + ramp, path] += size / ramp
    price = np.empty((n + 1, n_paths))
    price[0] = market.s0
    price[1:] = market.s0 + np.cumsum(inc, axis=0)
    return price


def as_half_spread(market, mm, tau):
    """Half of the Avellaneda-Stoikov optimal spread with tau seconds to the horizon."""
    var = market.total_variance
    return 0.5 * (mm.gamma * var * tau + (2 / mm.gamma) * np.log(1 + mm.gamma / market.noise_k))


def simulate(market, mm, n_paths=200, seed=0, markout_horizons=(1.0, 10.0, 60.0), price=None, trace=False):
    """Run the market maker on ``n_paths`` paths.

    Every fill is valued against the efficient price at the time of the fill (``edge``: what we earn if the
    price then stood still) and ``h`` seconds later for each markout horizon (edge plus the move that
    followed). The final PnL splits exactly into the edge of all fills plus the inventory PnL minus fees.
    With ``trace=True`` the quotes and fills of the first path are recorded for plotting.
    ``price`` can be passed to run several market makers on the same efficient-price paths.
    """
    rng = np.random.default_rng(seed)
    if price is None:
        price = efficient_price(market, n_paths, rng)
    n, dt = market.steps, market.dt
    lag = int(round(mm.latency / dt))
    window = max(1, int(round(mm.guard_window / dt)))
    pause = int(round(mm.guard_pause / dt))
    horizons = {h: int(round(h / dt)) for h in markout_horizons}
    var = market.total_variance

    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    fees = np.zeros(n_paths)
    inv_pnl = np.zeros(n_paths)
    no_ask_until = np.full(n_paths, -1)
    no_bid_until = np.full(n_paths, -1)
    stats = {who: dict(fills=np.zeros(n_paths), edge=np.zeros(n_paths),
                       markout={h: np.zeros(n_paths) for h in horizons}) for who in WHO}
    tr = dict(seen=np.empty(n), bid=np.full(n, np.nan), ask=np.full(n, np.nan), q=np.empty(n), fills=[]) if trace else None

    def book(who, mask, signed_qty, fill_price, i):
        """signed_qty: +1 when we buy, -1 when we sell."""
        if not mask.any():
            return
        s = stats[who]
        s["fills"] += mask
        fees[:] += np.where(mask, mm.fee_bp * 1e-4 * fill_price, 0.0)
        s["edge"] += np.where(mask, signed_qty * (price[i] - fill_price), 0.0)
        for h, steps in horizons.items():
            s["markout"][h] += np.where(mask, signed_qty * (price[min(i + steps, n)] - fill_price), 0.0)
        if trace and mask[0]:
            tr["fills"].append((i * dt, float(np.broadcast_to(fill_price, mask.shape)[0]), who, signed_qty))

    for i in range(n):
        p = price[i]
        seen = price[max(i - lag, 0)]
        tau = market.horizon - i * dt
        reservation = seen - q * mm.gamma * var * tau
        half = as_half_spread(market, mm, tau) + mm.widen
        bid, ask = reservation - half, reservation + half

        # guard: a move seen over the last window withdraws the side the arbitrageurs would hit
        if np.isfinite(mm.guard_bps):
            move = seen - price[max(i - lag - window, 0)]
            no_ask_until = np.where(move > mm.guard_bps * BP, i + pause, no_ask_until)
            no_bid_until = np.where(move < -mm.guard_bps * BP, i + pause, no_bid_until)
        bid_on = (q < mm.max_inventory) & (i > no_bid_until)
        ask_on = (q > -mm.max_inventory) & (i > no_ask_until)
        if trace:
            tr["seen"][i], tr["q"][i] = seen[0], q[0]
            tr["bid"][i] = bid[0] if bid_on[0] else np.nan
            tr["ask"][i] = ask[0] if ask_on[0] else np.nan

        # arbitrageurs first: they take any quote stale by more than their cost
        if market.arbitrage:
            arb_buy = ask_on & (p - ask > market.arb_cost)       # they buy our ask, we sell
            arb_sell = bid_on & (bid - p > market.arb_cost)      # they sell to our bid, we buy
            book("arbitrage", arb_buy, -1.0, ask, i)
            book("arbitrage", arb_sell, +1.0, bid, i)
            q += arb_sell.astype(float) - arb_buy
            cash += np.where(arb_buy, ask, 0.0) - np.where(arb_sell, bid, 0.0)
            ask_on &= ~arb_buy
            bid_on &= ~arb_sell

        # noise traders on what is left: uninformed, so their intensity depends on the distance of the
        # quote from the venue's own mid (what we see), as in Avellaneda-Stoikov, not on the efficient price
        p_bid = np.clip(market.noise_A * np.exp(-market.noise_k * (seen - bid)) * dt, 0.0, 1.0)
        p_ask = np.clip(market.noise_A * np.exp(-market.noise_k * (ask - seen)) * dt, 0.0, 1.0)
        noise_sell = bid_on & (rng.random(n_paths) < p_bid)
        noise_buy = ask_on & (rng.random(n_paths) < p_ask)
        book("noise", noise_sell, +1.0, bid, i)
        book("noise", noise_buy, -1.0, ask, i)
        q += noise_sell.astype(float) - noise_buy
        cash += np.where(noise_buy, ask, 0.0) - np.where(noise_sell, bid, 0.0)

        inv_pnl += q * (price[i + 1] - p)

    edge = stats["noise"]["edge"] + stats["arbitrage"]["edge"]
    return dict(pnl=cash + q * price[n] - fees, edge=edge, inventory_pnl=inv_pnl, fees=fees, final_q=q.copy(),
                stats=stats, price=price, trace=tr, market=market, mm=mm)


def markout_bp(result, who, horizon):
    """Mean markout per fill of one counterparty at one horizon, in basis points."""
    s = result["stats"][who]
    fills = s["fills"].sum()
    return s["markout"][horizon].sum() / fills / BP if fills else np.nan


def summary(result, horizon=10.0):
    """Per-path means in USD (orders of one unit worth 100 USD) and markouts per fill in basis points."""
    s = result["stats"]
    row = {"PnL": result["pnl"].mean(), "PnL sd": result["pnl"].std(),
           "edge": result["edge"].mean(), "inventory": result["inventory_pnl"].mean(),
           "fees": result["fees"].mean()}
    for who in WHO:
        row[f"{who} fills"] = s[who]["fills"].mean()
        row[f"{who} markout {horizon:g}s (bp)"] = markout_bp(result, who, horizon)
    return row
