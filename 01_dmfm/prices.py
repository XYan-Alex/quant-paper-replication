"""读日线二进制 → 宽表（日期 × 股票）。股票池：历史上进过沪深300/中证500/中证1000 的全部股票。"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import numpy as np, pandas as pd, os
Q = os.path.join(DATA, "qlib_bin")
cal = pd.to_datetime(pd.read_csv(f"{Q}/calendars/day.txt", header=None)[0])
START = "2008-06-01"
i0 = int(np.searchsorted(cal.values, np.datetime64(START))); dates = cal[i0:].reset_index(drop=True); T = len(dates)
mem = {u: pd.read_csv(f"{Q}/instruments/{u}.txt", sep="\t", header=None, names=["code", "s", "e"], parse_dates=["s", "e"]) for u in ("csi300", "csi500", "csi1000")}
codes = sorted(set().union(*[set(m.code) for m in mem.values()])); N = len(codes); cix = {c: i for i, c in enumerate(codes)}
def read(code, field):
    p = f"{Q}/features/{code.lower()}/{field}.day.bin"
    out = np.full(T, np.nan, np.float32)
    if not os.path.exists(p): return out
    a = np.fromfile(p, "<f4"); s = int(a[0]); v = a[1:]
    lo = max(s, i0); hi = min(s + len(v), i0 + T)
    if hi > lo: out[lo - i0:hi - i0] = v[lo - s:hi - s]
    return out
P = {}
for f in ("close", "open", "high", "low", "volume", "amount", "factor", "vwap"):
    P[f] = np.stack([read(c, f) for c in codes], 1); print(f, P[f].shape, np.isnan(P[f]).mean().round(3), flush=True)
M = {}
for u, m in mem.items():
    A = np.zeros((T, N), bool)
    for c, s, e in m.itertuples(index=False):
        a, b = np.searchsorted(dates.values, np.datetime64(s)), np.searchsorted(dates.values, np.datetime64(e), "right")
        A[a:b, cix[c]] = True
    first = np.argmax(A.any(1)); A[:first] = A[first]          # 指数发布前：沿用首期成分（口径说明见文档）
    M[u] = A; print(u, "first", dates[first].date(), "members/day", A.sum(1).min(), A.sum(1).max())
idx = {k: read(c, "close") for k, c in (("csi300", "SH000300"), ("csi500", "SH000905"), ("csi1000", "SH000852"))}
for k, v in idx.items(): print(k, "index nan", np.isnan(v).mean().round(3), v[-1] / v[np.searchsorted(dates.values, np.datetime64("2016-01-04"))])
np.savez_compressed(os.path.join(DATA, "prices.npz"), dates=dates.values, codes=np.array(codes), **P, **{"m_" + u: a for u, a in M.items()}, **{"i_" + k: v for k, v in idx.items()})
c = P["close"][:, cix["SH600000"]]; f = P["factor"][:, cix["SH600000"]]; print("600000 last adj/raw", c[-1], c[-1] / f[-1], "amount", P["amount"][-1, cix["SH600000"]], "vol", P["volume"][-1, cix["SH600000"]], "vwap", P["vwap"][-1, cix["SH600000"]])
