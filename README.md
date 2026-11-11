# Market Making: modello di Avellaneda-Stoikov

Simulazione della strategia di market making ottimale di **Avellaneda & Stoikov (2008)**, *High-frequency trading in a limit order book*, con i parametri della Sezione 4 dell'articolo.

## Contenuto — [`avellaneda_stoikov.ipynb`](avellaneda_stoikov.ipynb)

* **Prezzo di riserva** $r = s - q \gamma \sigma^2 (T-t)$: le quote si spostano contro l'inventario per riportarlo verso zero.
* **Spread ottimale** $\delta^a + \delta^b = \gamma \sigma^2 (T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$.
* **Mid-price** come passeggiata aleatoria $s_{t+dt} = s_t \pm \sigma\sqrt{dt}$ ed **esecuzioni** con intensità di Poisson $\lambda(\delta) = A e^{-k\delta}$.
* **Confronto Monte Carlo** (1000 simulazioni) con una strategia simmetrica attorno al mid-price, per tre livelli di avversione al rischio $\gamma$.

## Risultati

| $\gamma$ | Strategia | PnL medio | Dev. std PnL | Dev. std inventario finale |
|---|---|---:|---:|---:|
| 0.01 | Avellaneda-Stoikov | 68.35 | 8.91 | 5.20 |
| 0.01 | Simmetrica | 68.26 | 13.28 | 8.53 |
| 0.1 | Avellaneda-Stoikov | 64.90 | 6.52 | 2.89 |
| 0.1 | Simmetrica | 67.39 | 12.90 | 8.31 |
| 0.5 | Avellaneda-Stoikov | 48.59 | 5.79 | 1.93 |
| 0.5 | Simmetrica | 58.32 | 11.29 | 6.99 |

La strategia con gestione dell'inventario dimezza circa la dispersione del PnL e riduce di tre o quattro volte quella dell'inventario, rinunciando a una parte del guadagno medio; più alto è $\gamma$, più netto è il compromesso.

## ▶️ Esecuzione

Serve Python 3.9 o successivo.

```bash
git clone https://github.com/lancellottimatteo19/market-making-avellaneda-stoikov.git
cd market-making-avellaneda-stoikov
pip install -r requirements.txt
jupyter lab
```

I notebook sono salvati con gli output, quindi grafici e risultati si vedono anche direttamente su GitHub.
