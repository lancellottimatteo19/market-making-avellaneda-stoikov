# Market Making: from Avellaneda-Stoikov to Adverse Selection

How much does a market maker lose to faster traders, and what can it do about it? Three notebooks:

1. a replication of the **Avellaneda & Stoikov (2008)** optimal market-making model;
2. the same market maker on a **lagging venue**, where the price is discovered elsewhere and **arbitrageurs** pick off stale quotes — the problem we met in the [Perpetual Alpha](https://github.com/lancellottimatteo19/perpetual-alpha) trading competition;
3. the same mechanism measured on **49 hours of real Hyperliquid and Binance data**, with the model calibrated on them.

**Key results on real data (BTC, ETH, HYPE, 10–12 November 2026)**

| | BTC | ETH | HYPE |
|---|---:|---:|---:|
| Delay of Hyperliquid behind Binance (exchange clocks) | 0.5 s | 0.5 s | 0.2 s |
| Share of volume that follows a Binance move in the previous second | 33% | 47% | 57% |
| Maker's 10 s markout on those trades, when the move exceeds 3 bp | −2.8 bp | −2.7 bp | −2.0 bp |
| Maker's 10 s markout when Binance is flat | +0.07 bp | +0.24 bp | +0.07 bp |

The two-venue model, calibrated on Binance volatility and Hyperliquid spreads, reproduces these markouts without being fitted to them (+0.07 and −3.0 bp against +0.07 and −2.8 bp). It shows that on these venues a market maker's profit depends on **latency and fees**, not on the spread formula.

## 1. The Avellaneda-Stoikov model — [`avellaneda_stoikov.ipynb`](avellaneda_stoikov.ipynb)

* **Reservation price** $r = s - q \gamma \sigma^2 (T-t)$: quotes are skewed against the inventory to bring it back to zero.
* **Optimal spread** $\delta^a + \delta^b = \gamma \sigma^2 (T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$.
* **Mid-price** as a random walk $s_{t+dt} = s_t \pm \sigma\sqrt{dt}$ and **fills** with Poisson intensity $\lambda(\delta) = A e^{-k\delta}$, with the parameters of Section 4 of the paper.
* **Monte Carlo comparison** (1000 simulations) with a strategy quoting symmetrically around the mid, for three levels of risk aversion $\gamma$.

| $\gamma$ | Strategy | Mean PnL | PnL std. dev. | Final inventory std. dev. |
|---|---|---:|---:|---:|
| 0.01 | Avellaneda-Stoikov | 68.35 | 8.91 | 5.20 |
| 0.01 | Symmetric | 68.26 | 13.28 | 8.53 |
| 0.1 | Avellaneda-Stoikov | 64.90 | 6.52 | 2.89 |
| 0.1 | Symmetric | 67.39 | 12.90 | 8.31 |
| 0.5 | Avellaneda-Stoikov | 48.59 | 5.79 | 1.93 |
| 0.5 | Symmetric | 58.32 | 11.29 | 6.99 |

Inventory management roughly halves the dispersion of the PnL and cuts that of the inventory by a factor of three to four, at the cost of part of the mean profit; the higher $\gamma$, the sharper the trade-off.

## 2. Adverse selection on a lagging venue — [`two_venue_adverse_selection.ipynb`](two_venue_adverse_selection.ipynb)

In real markets the price is discovered across several venues: in crypto, Binance usually leads and other venues follow with a delay. The model has:

* an **efficient price** on the leading venue: Brownian motion plus information events incorporated over a few seconds;
* the **market maker** on the lagging venue, which sees the leading price with a delay and quotes with the Avellaneda-Stoikov formulas;
* uninformed **noise traders**, as in the paper, and **arbitrageurs** who see the leading venue and take any quote that is stale by more than their cost;
* two defences: a **wider spread**, and a **guard** that withdraws the exposed side for an instant after a price move.

| Scenario (half an hour, orders of 1 unit worth 100 USD) | Mean PnL (USD) |
|---|---:|
| Textbook: no arbitrageurs | 4.47 |
| 1 s delay, with arbitrageurs | 0.09 |
| 1 s + 1 bp wider spread | 0.15 |
| 1 s + guard (1 bp threshold, 0.25 s pause) | 2.10 |
| 200 ms delay | 4.47 |

Arbitrageurs make up 13% of the fills, but each costs −13 bp at 10 seconds on average, against +1.9 bp for a fill against a noise trader. Speed removes the problem. Without speed, the guard recovers almost half of the lost profit, while widening the spread barely helps. The guard's threshold and pause are chosen on paths different from those used for evaluation.

The simulator is [`two_venue_sim.py`](two_venue_sim.py): vectorised over Monte Carlo paths, it splits the PnL exactly into the edge of the fills, the inventory PnL and fees, and measures markouts by counterparty.

## 3. Real data: Hyperliquid versus Binance — [`hyperliquid_binance_data.ipynb`](hyperliquid_binance_data.ipynb)

Public feeds recorded from 10 to 12 November 2026: every best bid and ask update and every trade on Hyperliquid (1.76 million trades, 9.8 billion USD of volume) and the Binance futures book every 100 ms, for BTC, ETH and HYPE.

- **Binance leads and Hyperliquid follows.** Hyperliquid closes almost all of the gap between the two venues; Binance does not move.
- **The flow that follows Binance is toxic** and makes up a third of the volume or more (table above).
- **Latency and fees decide.** With one-tick spreads, a BTC market maker only makes money with less than about 100–200 ms of delay behind Binance and fees close to zero; at the base fee of around 1.5 bp it loses in any case.

The scripts are in [`empirical/`](empirical/): `extract_mainnet.py` extracts the data from the recordings and `analyze_mainnet.py` writes the aggregated results to [`empirical/results/`](empirical/results/), which the notebook reads. The raw data are not included: they amount to tens of GB and exchange feeds restrict redistribution.

## Where this sits in the literature

The model combines three known strands: adverse selection (Glosten and Milgrom, 1985; Copeland and Galai, 1983), optimal market making (Avellaneda and Stoikov, 2008; Cartea, Jaimungal and Ricci, 2014) and latency arbitrage on stale quotes (Budish, Cramton and Shim, 2015; Foucault, Kozhan and Tham, 2017). Its DeFi counterpart is the *loss-versus-rebalancing* of liquidity providers in automated market makers (Milionis et al., 2022). The contribution here is not theoretical: it makes the Budish-Cramton-Shim mechanism measurable in a simulation, links it to what we observed in the competition and checks it against real data. The full discussion and references are at the end of [`two_venue_adverse_selection.ipynb`](two_venue_adverse_selection.ipynb).

### Main references

* Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*.
* Budish, E., Cramton, P. and Shim, J. (2015). The high-frequency trading arms race: frequent batch auctions as a market design response. *Quarterly Journal of Economics*.
* Cartea, Á., Jaimungal, S. and Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
* Foucault, T., Kozhan, R. and Tham, W. W. (2017). Toxic arbitrage. *Review of Financial Studies*.
* Glosten, L. and Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*.
* Milionis, J., Moallemi, C. C., Roughgarden, T. and Zhang, A. L. (2022). Automated market making and loss-versus-rebalancing. arXiv preprint.

## How to run

Requires Python 3.9 or later.

```bash
git clone https://github.com/lancellottimatteo19/market-making-avellaneda-stoikov.git
cd market-making-avellaneda-stoikov
pip install -r requirements.txt
jupyter lab
```

Simulator tests:

```bash
python -m pytest tests
```

The notebooks are saved with their outputs, so charts and results can be read directly on GitHub.
