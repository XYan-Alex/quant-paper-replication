import numpy as np, sys, os, json
from backtest import *
def z(S):
    return (S - np.nanmean(S, 1, keepdims=True)) / (np.nanstd(S, 1, keepdims=True) + 1e-9)
for u in sys.argv[1:]:
    X, Y, idx, dates, groups, names = panel(u); bench = P["i_" + u][t0:].astype(np.float64)
    SC = {"ew": ew_score(X, groups).astype(np.float64), "linear": linear_score(X, Y, idx, dates)}
    SC["ew"][idx < 0] = np.nan
    for k in ("dmfm", "lite", "mlp"):
        if os.path.exists(f"{D}/score_{u}_{k}.npy"): SC[k] = np.load(f"{D}/score_{u}_{k}.npy")
    combos = {"ew": ["ew"], "ew+dmfm": ["ew", "dmfm"], "ew+lite": ["ew", "lite"], "ew+linear": ["ew", "linear"], "ew+dmfm+lite": ["ew", "dmfm", "lite"], "ew+dmfm+linear": ["ew", "dmfm", "linear"], "all4": ["ew", "dmfm", "lite", "linear"]}
    print("==", u)
    for name, ks in combos.items():
        if not all(k in SC for k in ks): continue
        S = sum(z(SC[k]) for k in ks); out = []
        for s, e in (("2016-01-01", "2022-07-01"), ("2022-07-01", None)):
            nv, bn, to = backtest(S, idx, dates, bench, start=s, end=e); m = metrics(nv, bn); ic, icir, _ = ic_stats(S, Y, idx, dates, s, e); out.append(f"α {m['alpha']*100:6.2f} IR {m['IR']:5.2f} ic {ic*100:5.2f} icir {icir:4.2f}")
        print(f"{name:16s} paper: {out[0]} | after: {out[1]}")
    if "lite" in SC:                                  # 片中“复现版本”：63 个因子等权 + lite，各自截面标准化后相加
        np.save(f"{D}/score_{u}_ours.npy", (z(SC["ew"]) + z(SC["lite"])).astype(np.float32))
