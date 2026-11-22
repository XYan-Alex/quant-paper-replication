"""Neural Shrinkage of Indefinite Pairwise Correlations（arXiv:2608.30446）在 A 股上的复现。
问题：股票上市时间不同、时有停牌，两两按“共同有数据的那段”算相关系数，拼出来的相关矩阵可能不是正定的，没法直接做最小方差组合。
论文做法：把带符号的特征值逐个喂给双向 GRU，输出一组正的“逆特征值”，端到端训练，目标是未来 5 天最小方差组合的实际方差最小。
和论文不同：A 股（论文是最多 1500 只美股、2000—2025 逐年扩展训练、券商级撮合仿真）；这里只训练一次（2010—2015），2016-01 起全部是样本外；
每期从中证1000 成分股里随机取 500 只、回看 500 天；不计交易成本；波动率分支加了一条乘法直连（输出 = 样本波动率 × 网络修正）。
用法：python3 neural_shrinkage.py train | eval"""
import os, sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "07_neural_shrinkage"); os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.default_rng(0)
Z = np.load(os.path.join(DATA, "prices.npz"), allow_pickle=True); D = pd.to_datetime(Z["dates"])
c = (Z["close"] * Z["factor"]).astype(np.float64); RET = c[1:] / c[:-1] - 1; RET = np.vstack([np.full((1, c.shape[1]), np.nan), RET])
RET[~(np.isfinite(Z["close"]) & (np.nan_to_num(Z["volume"]) > 0))] = np.nan; RET = np.clip(RET * 100, -21, 21)        # 百分数；停牌日记为缺失
MEM = Z["m_csi1000"] > 0; T, N = RET.shape; HOR = 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor([0.5, -2.0, -1.0, -2.0, -4.0]))                       # 滞后变换 5 个参数（softplus 后为正）
        self.vol = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(), nn.Linear(8, 1))
        self.gru = nn.GRU(4, 32, batch_first=True, bidirectional=True); self.proj = nn.Linear(64, 1)
    def forward(self, R, M):
        """R：Δt×n 收益（缺失处为 0，最后一行是最近一天），M：有效标记。返回 协方差估计 Σ（n×n）"""
        dt, n = R.shape; c = Fn.softplus(self.c); lag = torch.arange(dt, 0, -1, dtype=R.dtype).unsqueeze(1)
        a = c[0] * lag ** (-c[1]); b = c[2] + c[3] * (1 - torch.exp(-c[4] * lag)) + 1e-3          # b_t = c2 − c3·e^{−c4 t}（改写成恒正的形式）
        Rt = a * torch.tanh(b * R) / b * M
        Tii = M.sum(0).clamp(min=1); mu = Rt.sum(0) / Tii; sig = torch.sqrt((((Rt - mu) * M) ** 2).sum(0) / Tii + 1e-8)
        Zs = (Rt - mu) / sig * M; Tij = (M.T @ M).clamp(min=1); C = (Zs.T @ Zs) / Tij; C = C * ((M.T @ M) > 0); C = (C + C.T) / 2
        lam, Q = torch.linalg.eigh(C); Q2 = Q ** 2
        tau = ((Q2.T @ (M.T @ M)) * Q2.T).sum(1); x = torch.stack([lam, torch.full_like(lam, n ** 0.5 / 20), tau.sqrt() / 30, n / tau], 1)
        h, _ = self.gru(x.unsqueeze(0)); inv = Fn.softplus(self.proj(h[0]).squeeze(-1)) + 1e-4
        ev = 1 / inv; ev = ev / ev.mean()
        Qn = Q / torch.sqrt((Q2 * ev).sum(1, keepdim=True)); Cn = (Qn * ev) @ Qn.T
        s = sig / torch.sqrt((sig ** 2).mean()); vol = s * Fn.softplus(self.vol(torch.stack([s, Tii ** -0.5 * 10], 1)).squeeze(-1) + 0.55) / 1.0
        return Cn * vol.unsqueeze(0) * vol.unsqueeze(1), lam.detach(), ev.detach()
def panel(t, cols, dt):
    R = RET[t - dt + 1:t + 1][:, cols]; M = np.isfinite(R); return np.nan_to_num(R), M
def sample(lo, hi):
    while True:
        t = int(rng.integers(lo, hi - HOR)); dt = int(rng.integers(250, 751)); n = int(rng.integers(50, 251))
        ok = np.where((np.isfinite(RET[t - dt + 1:t + 1]).sum(0) >= 60) & np.isfinite(RET[t]) & np.isfinite(RET[t + 1:t + 1 + HOR]).all(0))[0]
        if len(ok) < n: continue
        cols = rng.choice(ok, n, replace=False); R, M = panel(t, cols, dt); return R, M, RET[t + 1:t + 1 + HOR][:, cols]
def train(steps=3000):
    lo = int(np.searchsorted(D.values, np.datetime64("2010-01-01"))) + 750; hi = int(np.searchsorted(D.values, np.datetime64("2016-01-01")))
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3); t0 = time.time(); run = []; base = []
    for it in range(steps):
        R, M, Ro = sample(max(lo, 760), hi); R, M, Ro = torch.tensor(R, dtype=torch.float32), torch.tensor(M, dtype=torch.float32), torch.tensor(Ro, dtype=torch.float32)
        S, _, _ = net(R, M); n = S.shape[0]; w = torch.linalg.solve(S, torch.ones(n)); w = w / w.sum()
        loss = n * ((Ro @ w) ** 2).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        run.append(loss.item()); base.append((n * ((Ro.mean(1)) ** 2).mean()).item())            # 对照：等权组合的同一个量
        if (it + 1) % 250 == 0: print(it + 1, "loss", round(np.mean(run[-250:]), 3), "等权", round(np.mean(base[-250:]), 3), round(time.time() - t0), flush=True)
    torch.save(net.state_dict(), os.path.join(DATA, "ns_net.pt")); json.dump(dict(loss=[round(x, 4) for x in run], ew=[round(x, 4) for x in base]), open(os.path.join(OUT, "train_log.json"), "w"))
# ---------- 评估用的对照估计量 ----------
def pairwise(R, M):
    Tii = M.sum(0); mu = R.sum(0) / Tii; sig = np.sqrt((((R - mu) * M) ** 2).sum(0) / Tii); Zs = (R - mu) / sig * M; Tij = M.T @ M
    C = np.where(Tij > 0, Zs.T @ Zs / np.maximum(Tij, 1), 0); return (C + C.T) / 2, sig
def nearest_corr(C, iters=30):
    """Higham (2002) 交替投影：最近的相关矩阵"""
    Yk = C.copy(); dS = np.zeros_like(C)
    for _ in range(iters):
        Rk = Yk - dS; lam, Q = np.linalg.eigh(Rk); Xk = (Q * np.maximum(lam, 1e-6)) @ Q.T; dS = Xk - Rk; Yk = Xk.copy(); np.fill_diagonal(Yk, 1)
    lam, Q = np.linalg.eigh(Yk); Yk = (Q * np.maximum(lam, 1e-6)) @ Q.T; d = np.sqrt(np.diag(Yk)); return Yk / d[:, None] / d[None]
def lw_shrink(C, q):
    """相关矩阵向单位阵线性收缩（强度按 n/有效样本 的启发式给出）"""
    a = min(1.0, q / (1 + q)); return (1 - a) * C + a * np.eye(len(C))
def gmv_long(S):
    import quadprog
    n = len(S); S = S + 1e-8 * np.trace(S) / n * np.eye(n); Cm = np.column_stack([np.ones(n), np.eye(n)])
    try: w = quadprog.solve_qp(S, np.zeros(n), Cm, np.concatenate([[1.0], np.zeros(n)]), 1)[0]
    except ValueError: w = 1 / np.diag(S)
    w = np.clip(w, 0, None); return w / w.sum()
def evaluate(nu=500, dt=500):
    net = Net(); net.load_state_dict(torch.load(os.path.join(DATA, "ns_net.pt"))); net.eval()
    a = int(np.searchsorted(D.values, np.datetime64("2016-01-01"))); names = ["神经网络收缩（论文）", "两两相关+最近相关矩阵", "两两相关+线性收缩", "只用波动率（对角）", "等权"]
    rets = {k: [] for k in names}; wprev = {k: None for k in names}; tov = {k: 0.0 for k in names}; neg = []; dts = []; r2 = np.random.default_rng(1); t0 = time.time()
    for t in range(a, T - HOR, HOR):
        ok = np.where(MEM[t] & np.isfinite(RET[t]) & (np.isfinite(RET[t - dt + 1:t + 1]).sum(0) >= 60))[0]; cols = r2.choice(ok, min(nu, len(ok)), replace=False)
        R, M = panel(t, cols, dt); Mf = M.astype(np.float64); C, sig = pairwise(R, Mf); lam = np.linalg.eigvalsh(C); neg.append(float((lam < -1e-8).mean())); Dg = np.outer(sig, sig)
        with torch.no_grad(): Snn = net(torch.tensor(R, dtype=torch.float32), torch.tensor(M, dtype=torch.float32))[0].numpy().astype(np.float64)
        W = {names[0]: gmv_long(Snn), names[1]: gmv_long(nearest_corr(C) * Dg), names[2]: gmv_long(lw_shrink(nearest_corr(C, 5), len(cols) / Mf.sum(0).mean()) * Dg), names[3]: (1 / sig ** 2) / (1 / sig ** 2).sum(), names[4]: np.ones(len(cols)) / len(cols)}
        fut = np.nan_to_num(RET[t + 1:t + 1 + HOR][:, cols]) / 100
        for k in names:
            w = W[k]; g = np.cumprod(1 + fut, 0) @ w; rets[k] += list(np.diff(np.concatenate([[1.0], g])) / np.concatenate([[1.0], g[:-1]]))
            full = np.zeros(N); full[cols] = w
            if wprev[k] is not None: tov[k] += np.abs(full - wprev[k]).sum() / 2
            wprev[k] = full
        dts += list(D[t + 1:t + 1 + HOR])
        if len(neg) % 50 == 0: print(str(D[t])[:10], len(neg), round(time.time() - t0), {k[:4]: round(np.std(v) * np.sqrt(244) * 100, 2) for k, v in rets.items()}, flush=True)
    res = {}
    for k in names:
        r = np.array(rets[k]); r5 = np.add.reduceat(r, np.arange(0, len(r), HOR)); nav = np.cumprod(1 + r)
        res[k] = dict(vol_5d=round(r5.std() * np.sqrt(244 / HOR) * 100, 2), vol_1d=round(r.std() * np.sqrt(244) * 100, 2), ann=round((nav[-1] ** (244 / len(r)) - 1) * 100, 2), sharpe=round(r.mean() / r.std() * np.sqrt(244), 2), mdd=round(float((nav / np.maximum.accumulate(nav) - 1).min()) * 100, 1), turnover=round(tov[k] / (len(r) / 244), 1))
        print(k, res[k], flush=True)
    res["_meta"] = dict(start=str(dts[0])[:10], end=str(dts[-1])[:10], rebalances=len(neg), n=nu, lookback=dt, share_negative_eigenvalues=round(float(np.mean(neg)) * 100, 2))
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), ensure_ascii=False, indent=1); pd.DataFrame(rets, index=dts).round(6).to_csv(os.path.join(OUT, "daily_returns.csv"))
if __name__ == "__main__":
    train(int(sys.argv[2]) if len(sys.argv) > 2 else 3000) if sys.argv[1] == "train" else evaluate()
