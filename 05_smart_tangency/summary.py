"""汇总 smart_tangency.py 的各轮结果：6 个智能体、集成（验证期夏普最高者）、三种切点组合基准、等权、SPY。
用法：python3 summary.py etf"""
import os, sys, json, glob, numpy as np, pandas as pd
sys.argv = sys.argv[:2] if len(sys.argv) > 1 else sys.argv + ["etf"]
from smart_tangency import *
if __name__ == "__main__":
    cyc = {}
    for f in sorted(glob.glob(os.path.join(DATA, "us", f"st_{U}_*_*.json"))):
        d = json.load(open(f)); k = tuple(d["cycle"]); cyc.setdefault(k, dict(dates=d["dates"], res={}))["res"].update(d["res"])
    S = {}; picks = []; acts = {}; RS = {}
    for k in sorted(cyc):
        c = cyc[k]; a, b = int(np.searchsorted(dates.values, np.datetime64(c["dates"][0]))), int(np.searchsorted(dates.values, np.datetime64(c["dates"][-1]))) + 1; idx = dates[a:b]
        if len(c["res"]) < 6: print("轮次不完整，跳过", k); continue
        for name, v in c["res"].items(): S.setdefault(name, []).append(pd.Series(v["test"], idx)); acts.setdefault(name, []).extend(v["acts"])
        best = max(c["res"], key=lambda n: c["res"][n]["val_sharpe"]); picks.append((k[2], best, round(c["res"][best]["val_sharpe"], 2))); S.setdefault("ensemble", []).append(pd.Series(c["res"][best]["test"], idx))
        for kind in KINDS: S.setdefault("bench_" + kind, []).append(pd.Series(tangency(kind, a, b), idx))
        for kind in KINDS:                                                                       # 对照：λ、h 完全随机选（20 个随机种子的平均日收益）
            rr = []
            for sd in range(20):
                g = np.random.default_rng(sd); t = a; o = np.zeros(b - a)
                while t < b - 1:
                    r_, e = hold(W[kind][t, int(g.integers(0, len(LAMS)))], t, int(HS[int(g.integers(0, len(HS)))]), b); o[t + 1 - a:e + 1 - a] = r_; t = e
                rr.append(o)
            S.setdefault("random_" + kind, []).append(pd.Series(np.mean(rr, 0), idx)); RS.setdefault(kind, []).append(np.array(rr))
        S.setdefault("equal_weight", []).append(pd.Series(np.concatenate([[0], R[a + 1:b].mean(1)]), idx)); S.setdefault("SPY", []).append(pd.Series(np.concatenate([[0], spy[a + 1:b]]), idx))
    S = pd.DataFrame({k: pd.concat(v) for k, v in S.items()}); out = {"ensemble_picks": picks}
    for tag, (s, e) in {"paper_2007_2022": ("2007-01-01", "2022-12-31"), "after_2023_2026": ("2023-01-01", "2026-12-31"), "all": ("2007-01-01", "2026-12-31")}.items():
        x = S.loc[s:e]
        if len(x) < 100: continue
        out[tag] = {c: stat(x[c].values) for c in x}; print("==", tag, str(x.index[0])[:10], str(x.index[-1])[:10])
        for c in x: print(f"  {c:14s}", out[tag][c])
    cut = int(np.searchsorted(S.index.values, np.datetime64("2023-01-01")))
    out["random_single_seed_2007_2022"] = {k: dict(ann_mean=round(float(np.mean([stat(x[:cut])["ann"] for x in np.concatenate(v, 1)])), 2), ann_min=round(float(np.min([stat(x[:cut])["ann"] for x in np.concatenate(v, 1)])), 2), ann_max=round(float(np.max([stat(x[:cut])["ann"] for x in np.concatenate(v, 1)])), 2),
                                           sharpe_mean=round(float(np.mean([stat(x[:cut])["sharpe"] for x in np.concatenate(v, 1)])), 2), sharpe_max=round(float(np.max([stat(x[:cut])["sharpe"] for x in np.concatenate(v, 1)])), 2)) for k, v in RS.items()}
    print("随机策略（单个种子）2007—2022：", out["random_single_seed_2007_2022"])
    out["avg_action"] = {k: dict(lam=round(float(np.mean([a[1] for a in v])), 1), h=round(float(np.mean([a[2] for a in v])), 1)) for k, v in acts.items()}
    print(picks); print(out["avg_action"])
    json.dump(out, open(os.path.join(OUT, f"result_{U}.json"), "w"), ensure_ascii=False, indent=1); np.cumprod(1 + S).round(4).to_csv(os.path.join(OUT, f"nav_{U}.csv"))
