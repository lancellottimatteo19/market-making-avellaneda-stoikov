# Market Making: da Avellaneda-Stoikov alla selezione avversa

Due notebook sul market making ottimale:

1. la replica del modello di **Avellaneda & Stoikov (2008)**, *High-frequency trading in a limit order book*;
2. lo stesso market maker su una **borsa che segue**, dove il prezzo si forma altrove e gli **arbitraggisti** colpiscono le quote rimaste indietro. È il problema che abbiamo incontrato nella gara di trading [Perpetual Alpha](https://github.com/lancellottimatteo19/perpetual-alpha).

## 1. Il modello di Avellaneda-Stoikov — [`avellaneda_stoikov.ipynb`](avellaneda_stoikov.ipynb)

* **Prezzo di riserva** $r = s - q \gamma \sigma^2 (T-t)$: le quote si spostano contro l'inventario per riportarlo verso zero.
* **Spread ottimale** $\delta^a + \delta^b = \gamma \sigma^2 (T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)$.
* **Mid-price** come passeggiata aleatoria $s_{t+dt} = s_t \pm \sigma\sqrt{dt}$ ed **esecuzioni** con intensità di Poisson $\lambda(\delta) = A e^{-k\delta}$, con i parametri della Sezione 4 dell'articolo.
* **Confronto Monte Carlo** (1000 simulazioni) con una strategia simmetrica attorno al mid-price, per tre livelli di avversione al rischio $\gamma$.

| $\gamma$ | Strategia | PnL medio | Dev. std PnL | Dev. std inventario finale |
|---|---|---:|---:|---:|
| 0.01 | Avellaneda-Stoikov | 68.35 | 8.91 | 5.20 |
| 0.01 | Simmetrica | 68.26 | 13.28 | 8.53 |
| 0.1 | Avellaneda-Stoikov | 64.90 | 6.52 | 2.89 |
| 0.1 | Simmetrica | 67.39 | 12.90 | 8.31 |
| 0.5 | Avellaneda-Stoikov | 48.59 | 5.79 | 1.93 |
| 0.5 | Simmetrica | 58.32 | 11.29 | 6.99 |

La strategia con gestione dell'inventario dimezza circa la dispersione del PnL e riduce di tre o quattro volte quella dell'inventario, rinunciando a una parte del guadagno medio; più alto è $\gamma$, più netto è il compromesso.

## 2. Selezione avversa su una borsa che segue — [`two_venue_adverse_selection.ipynb`](two_venue_adverse_selection.ipynb)

Sui mercati reali il prezzo si forma su più borse: per le crypto di solito guida Binance e le altre seguono con un ritardo. Il modello ha:

* un **prezzo efficiente** sulla borsa leader: moto browniano più eventi informativi che la borsa incorpora in pochi secondi;
* il **market maker** sulla borsa che segue, che vede il prezzo leader con un ritardo e quota con le formule di Avellaneda-Stoikov;
* **noise trader** non informati, come nell'articolo, e **arbitraggisti** che vedono la borsa leader e prendono ogni quota rimasta indietro di più del loro costo;
* due difese: **spread più largo** e una **guardia** che ritira il lato esposto per un istante dopo un movimento del prezzo.

| Scenario (mezz'ora, ordini da 1 unità a 100 USD) | PnL medio (USD) |
|---|---:|
| Manuale: nessun arbitraggista | 4,47 |
| 1 s di ritardo, con arbitraggisti | 0,09 |
| 1 s + spread più largo di 1 bp | 0,15 |
| 1 s + guardia (soglia 1 bp, pausa 0,25 s) | 2,10 |
| 250 ms di ritardo | 4,47 |

Gli arbitraggisti sono il 13% delle esecuzioni, ma ciascuna costa in media −13 bp a 10 secondi, contro +1,9 bp di un'esecuzione con un noise trader. La velocità elimina il problema. Senza velocità la guardia recupera quasi metà del guadagno perso, mentre allargare lo spread quasi non aiuta. La soglia e la pausa della guardia sono scelte su traiettorie diverse da quelle della valutazione.

Il simulatore è in [`two_venue_sim.py`](two_venue_sim.py): vettorizzato sulle traiettorie Monte Carlo, scompone il PnL in edge delle esecuzioni e PnL dell'inventario e misura il markout per tipo di controparte.

## ▶️ Esecuzione

Serve Python 3.9 o successivo.

```bash
git clone https://github.com/lancellottimatteo19/market-making-avellaneda-stoikov.git
cd market-making-avellaneda-stoikov
pip install -r requirements.txt
jupyter lab
```

Test del simulatore:

```bash
python -m pytest tests
```

I notebook sono salvati con gli output, quindi grafici e risultati si vedono anche direttamente su GitHub.
