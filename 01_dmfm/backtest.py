"""月度调仓回测（论文 4. Portfolio Construction）：每月最后一个交易日打分 → 选前 10% 等权 → 下一交易日收盘换仓，买卖各 4‰。
同时算两个不用训练的基线：EW（63 个因子等权）和 Linear（滚动线性回归）。"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import numpy as np, pandas as pd, json, sys, os
D = DATA; A = np.load(f"{D}/aux.npz"); Cl = A["C"]; traded = A["traded"]
P = np.load(f"{D}/prices.npz", allow_pickle=True); t0 = len(P["dates"]) - len(Cl)
SIGN = {"size": -1.0}
def panel(u):
    Z = np.load(f"{D}/panel_{u}.npz", allow_pickle=True); return Z["X"], Z["Y"], Z["idx"], pd.to_datetime(Z["dates"]), Z["groups"], Z["names"]
def ew_score(X, groups):
    s = np.array([SIGN.get(g, 1.0) for g in groups], np.float32); return (X * s).mean(-1)
def linear_score(X, Y, idx, dates):
    """每半年用到当时为止的数据（留 20 天间隔）回归一次 20 日收益，系数用于下一个半年"""
    T, n, m = X.shape; S = np.full((T, n), np.nan, np.float32); y = Y[:, :, 4]
    yz = (y - np.nanmean(y, 1, keepdims=True)) / (np.nanstd(y, 1, keepdims=True) + 1e-9)
    cuts = [np.searchsorted(dates.values, np.datetime64(f"{yy}-{mm}-01")) for yy in range(2016, 2028) for mm in ("01", "07")]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a >= T: break
        tr = slice(0, a - 130, 5); xs = X[tr].reshape(-1, m); ys = yz[tr].reshape(-1); v = np.isfinite(ys) & (idx[tr].reshape(-1) >= 0)
        beta = np.linalg.lstsq(xs[v].T @ xs[v] + 1e-3 * np.eye(m) * v.sum(), xs[v].T @ ys[v], rcond=None)[0]
        S[a:min(b, T)] = X[a:min(b, T)] @ beta
    return S
def backtest(S, idx, dates, bench, top=0.10, cost=0.004, start="2016-01-01", end=None):
    T = len(dates); a = np.searchsorted(dates.values, np.datetime64(start)); b = T if end is None else np.searchsorted(dates.values, np.datetime64(end))
    mend = [t for t in range(a - 1, b - 2) if dates[t].month != dates[t + 1].month]            # 月末信号日
    if a - 1 not in mend: mend = [a - 1] + mend
    ret = pd.DataFrame(Cl).pct_change(fill_method=None).fillna(0).values
    nav, w, cur, days, turn = 1.0, {}, [], [], []
    out = np.ones(b - a); k = 0
    for t in range(a, b):
        if w:                                                                                   # 持仓当日涨跌（买入持有，权重随价格漂移）
            tot = sum(w.values()); w = {c: v * (1 + ret[t, c]) for c, v in w.items()}; nav *= sum(w.values()) / tot
        if k < len(mend) and t == mend[k] + 1:                                                  # 信号日的下一交易日收盘换仓
            s = S[mend[k]]; v = np.isfinite(s) & (idx[mend[k]] >= 0); cand = idx[mend[k]][v]; sc = s[v]
            okk = traded[t, cand]; cand, sc = cand[okk], sc[okk]
            pick = cand[np.argsort(-sc)[:max(1, int(round(top * v.sum())))]]
            tot = sum(w.values()) if w else 1.0; old = {c: x / tot for c, x in w.items()}; new = {int(c): 1.0 / len(pick) for c in pick}
            tv = sum(abs(new.get(c, 0) - old.get(c, 0)) for c in set(new) | set(old))          # 买入 + 卖出的总比例
            nav *= 1 - cost * tv; turn.append(tv / 2); w = dict(new); k += 1
        out[t - a] = nav
    d = dates[a:b]; nv = pd.Series(out, d); bn = pd.Series(bench[a:b] / bench[a - 1], d) if a > 0 else pd.Series(bench[a:b] / bench[a], d)
    return nv, bn, float(np.mean(turn[1:]) * 12) if len(turn) > 1 else 0.0
def metrics(nv, bn, rf=0.0):
    r, rb = nv.pct_change().fillna(nv.iloc[0] - 1), bn.pct_change().fillna(bn.iloc[0] - 1); yrs = len(nv) / 244
    ann = nv.iloc[-1] ** (1 / yrs) - 1; annb = bn.iloc[-1] ** (1 / yrs) - 1; act = r - rb
    dd = (nv / nv.cummax() - 1).min(); ex = (1 + act).cumprod()
    return dict(ann=ann, bench=annb, alpha=ann - annb, IR=act.mean() / act.std() * np.sqrt(244), SR=(r.mean() * 244 - rf) / (r.std() * np.sqrt(244)), SRb=(rb.mean() * 244 - rf) / (rb.std() * np.sqrt(244)), mdd=dd, ex_mdd=(ex / ex.cummax() - 1).min(), total=nv.iloc[-1], total_b=bn.iloc[-1])
def ic_stats(S, Y, idx, dates, start, end=None):
    a = np.searchsorted(dates.values, np.datetime64(start)); b = len(dates) - 20 if end is None else np.searchsorted(dates.values, np.datetime64(end))
    ics = []
    for t in range(a, b, 20):                                                                  # 不重叠的 20 日窗口
        v = np.isfinite(S[t]) & np.isfinite(Y[t, :, 4])
        if v.sum() > 30: ics.append(pd.Series(S[t][v]).rank().corr(pd.Series(Y[t, v, 4]).rank()))
    ics = np.array(ics); return float(ics.mean()), float(ics.mean() / ics.std()), float((ics > 0).mean())
if __name__ == "__main__":
    res = {}; curves = {}
    PERIODS = {"paper": ("2016-01-01", "2022-07-01"), "after": ("2022-07-01", None), "all": ("2016-01-01", None)}
    for u in sys.argv[1:]:
        X, Y, idx, dates, groups, names = panel(u); bench = P["i_" + u][t0:].astype(np.float64)
        SC = {"ew": ew_score(X, groups), "linear": linear_score(X, Y, idx, dates)}
        for k in ("mlp", "mgat", "dmfm", "dmfmf", "lite", "ours"):
            f = f"{D}/score_{u}_{k}.npy"
            if os.path.exists(f): SC[k] = np.load(f)
        for k, S in SC.items():
            for pn, (s, e) in PERIODS.items():
                nv, bn, to = backtest(S, idx, dates, bench, start=s, end=e); m = metrics(nv, bn); ic, icir, pos = ic_stats(S, Y, idx, dates, s, e)
                res[f"{u}|{k}|{pn}"] = dict(**{a: round(float(b), 4) for a, b in m.items()}, ic=round(ic, 4), icir=round(icir, 3), turnover=round(to, 2))
                if pn == "all": curves[f"{u}|{k}"] = [round(float(x), 4) for x in nv.values[::5]]; curves[f"{u}|bench"] = [round(float(x), 4) for x in bn.values[::5]]; curves[f"{u}|dates"] = [str(x.date()) for x in nv.index[::5]]
        print(f"== {u}  （α=年化主动收益；论文区间 2016-01~2022-06 | 论文之后 2022-07~2026-11）")
        for k in SC:
            a, b = res[f"{u}|{k}|paper"], res[f"{u}|{k}|after"]
            print(f"{k:7s} paper: ann {a['ann']*100:6.2f} α {a['alpha']*100:6.2f} IR {a['IR']:5.2f} SR {a['SR']:5.2f} ic {a['ic']*100:5.2f} icir {a['icir']:5.2f} mdd {a['mdd']*100:6.1f} to {a['turnover']:4.1f} | after: ann {b['ann']*100:6.2f} α {b['alpha']*100:6.2f} IR {b['IR']:5.2f} ic {b['ic']*100:5.2f}")
        print("bench paper ann", round(res[f"{u}|ew|paper"]["bench"] * 100, 2), "SR", res[f"{u}|ew|paper"]["SRb"], "| after ann", round(res[f"{u}|ew|after"]["bench"] * 100, 2))
    json.dump(dict(res=res, curves=curves), open(f"{D}/results_{'_'.join(sys.argv[1:])}.json", "w"))
