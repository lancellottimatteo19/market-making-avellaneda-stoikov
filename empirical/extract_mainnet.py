"""Extract Hyperliquid and Binance quotes and Hyperliquid trades for BTC, ETH and HYPE from recorded sessions.

Input: one folder per recorded hour (``stream_YYYYMMDD_HHMMSS``) as written by the mainnet recorder of
https://github.com/lancellottimatteo19/perpetual-alpha (``recorder/registra_mainnet.py``): gzipped JSON lines
from the public WebSocket feeds, each message with the server's ``received_ms``:

* ``hl_*.jsonl.gz``: Hyperliquid ``bbo`` (best bid and ask, one update per block) and ``trades``;
* ``binance_*.jsonl.gz``: Binance USD-M futures ``depth5@100ms``.

Output: one ``<cache>/<session>.npz`` per hour with, for each coin,

* ``<coin>_hl``: received_ms, exchange time, bid, ask (Hyperliquid best quotes)
* ``<coin>_bn``: received_ms, exchange event time E, bid, ask (Binance best level of depth5)
* ``<coin>_tr``: received_ms, exchange time, price, size, side (+1 taker buy 'B', -1 taker sell 'A')

Trades are deduplicated by trade id keeping the first receipt: subscriptions and reconnections replay recent
trades, so the receipt time of a trade can be seconds after it happened; the analysis places trades at their
exchange time.

    python empirical/extract_mainnet.py <folder with the sessions> <cache folder>
"""
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

COINS = {"BTC": "btcusdt", "ETH": "ethusdt", "HYPE": "hypeusdt"}
BN2COIN = {v: k for k, v in COINS.items()}


def process(sess, cache):
    out = cache / f"{sess.name}.npz"
    if out.exists() or not (sess / "COMPLETE.json").exists():
        return
    d = {c: dict(hl=[], bn=[], tr={}) for c in COINS}
    for f in sorted(glob.glob(str(sess / "hl_*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if '"kind":"bbo"' in line:
                    if '"coin":"BTC"' not in line and '"coin":"ETH"' not in line and '"coin":"HYPE"' not in line:
                        continue
                    r = json.loads(line)
                    x = r["payload"]["data"]
                    b, a = x["bbo"]
                    if b is None or a is None:
                        continue
                    d[x["coin"]]["hl"].append((r["received_ms"], x["time"], float(b["px"]), float(a["px"])))
                elif '"kind":"trades"' in line:
                    r = json.loads(line)
                    for x in r["payload"]["data"]:
                        c = x["coin"]
                        if c in d and x["tid"] not in d[c]["tr"]:
                            d[c]["tr"][x["tid"]] = (r["received_ms"], x["time"], float(x["px"]), float(x["sz"]),
                                                    1 if x["side"] == "B" else -1)
    for f in sorted(glob.glob(str(sess / "binance_*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if '"kind":"depth5"' not in line:
                    continue
                k = line.find('"stream":"')
                sym = line[k + 10:line.find("@", k)]
                if sym not in BN2COIN:
                    continue
                r = json.loads(line)
                x = r["payload"]["data"]
                d[BN2COIN[sym]]["bn"].append((r["received_ms"], x["E"], float(x["b"][0][0]), float(x["a"][0][0])))
    arrays = {}
    for c, v in d.items():
        hl = np.array(sorted(v["hl"]), dtype=float).reshape(-1, 4)
        bn = np.array(sorted(v["bn"]), dtype=float).reshape(-1, 4)
        tr = np.array(sorted(v["tr"].values(), key=lambda r: (r[1], r[0])), dtype=float).reshape(-1, 5)
        arrays.update({f"{c}_hl": hl, f"{c}_bn": bn, f"{c}_tr": tr})
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    print(sess.name, {c: (len(d[c]["hl"]), len(d[c]["bn"]), len(d[c]["tr"])) for c in COINS}, flush=True)


if __name__ == "__main__":
    from multiprocessing import Pool
    src, cache = Path(sys.argv[1]), Path(sys.argv[2])
    cache.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in src.iterdir() if p.is_dir())
    with Pool(6) as pool:
        pool.starmap(process, [(s, cache) for s in sessions])
    print("done:", len(list(cache.glob("*.npz"))), "sessions")
