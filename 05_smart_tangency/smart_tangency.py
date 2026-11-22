"""Smart Tangency Portfolio（Int. J. Financial Stud. 2025, 13, 227）复现。
智能体不直接给权重，只选两个数：λ（有效前沿上的位置）和 h（多少天后再调仓，5~60 天）；权重交给传统优化器（frontier.py 预先算好）。
算法 PPO / A2C × 风险度量 方差 / 半方差 / CVaR = 6 个智能体；滚动：训练 2 年 → 验证 2 年 → 测试 2 年；
“集成”= 验证期夏普最高的那个智能体，拿去跑测试期。基准：三种风险度量的切点组合（252 天窗口、30 天重算）、等权、SPY。
和论文不同：9 只行业 ETF（论文 12 只）；λ、h 各按 5 一档；特征 11 个（论文 34 个）；每个智能体训练 STEPS 步、1 个随机种子。
用法：python3 smart_tangency.py etf [ppo,a2c] [起始轮] [结束轮]"""
import os, sys, json, numpy as np, pandas as pd, gymnasium as gym, torch, warnings
from stable_baselines3 import PPO, A2C
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "05_smart_tangency"); os.makedirs(OUT, exist_ok=True)
from frontier import load, LAMS, LOOK
U = sys.argv[1] if len(sys.argv) > 1 else "etf"; ALGOS = (sys.argv[2] if len(sys.argv) > 2 else "ppo,a2c").split(",")
STEPS = int(os.environ.get("STEPS", 12000)); HS = np.arange(5, 61, 5); KINDS = ["mv", "sv", "cvar"]
P, SPY = load(U); dates = P.index[1:]; Y = np.log(P).diff().dropna().values; R = np.expm1(Y); T, n = Y.shape
spy = SPY.pct_change().dropna().values
W = {k: np.load(os.path.join(DATA, "us", f"front_{U}_{k}.npy")) for k in KINDS}
def features():
    p = pd.DataFrame(np.cumsum(Y, 0)); r = pd.DataFrame(Y); F = []
    for k in (5, 20, 60, 120): F.append(p.diff(k))
    for k in (20, 60): F.append(r.rolling(k).std())
    up = r.clip(lower=0).rolling(14).mean(); dn = (-r.clip(upper=0)).rolling(14).mean(); F.append(up / (up + dn + 1e-12))       # RSI
    e12, e26 = np.exp(p).ewm(span=12).mean(), np.exp(p).ewm(span=26).mean(); macd = (e12 - e26) / e26; F.append(macd - macd.ewm(span=9).mean())
    for k in (20, 60): F.append(p - p.rolling(k).mean())
    F.append(p - p.rolling(60).max())
    X = np.stack([f.values for f in F], -1)                                                     # T × n × f
    Xd = pd.DataFrame(X.reshape(T, -1)); Zs = (Xd - Xd.rolling(LOOK, min_periods=120).mean()) / (Xd.rolling(LOOK, min_periods=120).std() + 1e-9)
    return np.nan_to_num(np.clip(Zs.values, -5, 5)).astype(np.float32)
X = features()
def hold(w, t, h, b):
    """t 日收盘按 w 建仓，持有到 min(t+h, b-1)，返回每天的组合收益"""
    e = min(t + h, b - 1); g = np.cumprod(1 + R[t + 1:e + 1], 0) @ w; return np.diff(np.concatenate([[1.0], g])) / np.concatenate([[1.0], g[:-1]]), e
class Env(gym.Env):
    def __init__(self, kind, a, b, random_start=True):
        self.kind, self.a, self.b, self.rs = kind, a, b, random_start
        self.observation_space = gym.spaces.Box(-5, 5, (X.shape[1],), np.float32); self.action_space = gym.spaces.MultiDiscrete([len(LAMS), len(HS)])
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.t = self.a + (int(self.np_random.integers(0, 60)) if self.rs else 0); return X[self.t], {}
    def step(self, act):
        w = W[self.kind][self.t, int(act[0])]; r, e = hold(w, self.t, int(HS[int(act[1])]), self.b); self.t = e
        return X[self.t], float(np.log1p(r).sum()), self.t >= self.b - 1, False, {}
def rollout(model, kind, a, b):
    t = a; out = np.zeros(b - a); acts = []; wprev = np.zeros(n); tov = 0.0
    while t < b - 1:
        act, _ = model.predict(X[t], deterministic=True); w = W[kind][t, int(act[0])]; r, e = hold(w, t, int(HS[int(act[1])]), b)
        out[t + 1 - a:e + 1 - a] = r; acts.append((str(dates[t])[:10], int(LAMS[int(act[0])]), int(HS[int(act[1])]))); tov += np.abs(w - wprev).sum() / 2; wprev = w * np.cumprod(1 + R[t + 1:e + 1], 0)[-1]; wprev /= wprev.sum(); t = e
    return out, acts, tov
def tangency(kind, a, b, every=30):
    t = a; out = np.zeros(b - a)
    while t < b - 1:
        y = Y[t - LOOK + 1:t + 1]; best, bw = -9, None
        for j in range(len(LAMS)):
            w = W[kind][t, j]; x = y @ w; s = x.mean() / (x.std() + 1e-12)
            if s > best: best, bw = s, w
        r, e = hold(bw, t, every, b); out[t + 1 - a:e + 1 - a] = r; t = e
    return out
def stat(r):
    r = np.asarray(r); nav = np.cumprod(1 + r); return dict(ann=round(r.mean() * 252 * 100, 2), vol=round(r.std() * np.sqrt(252) * 100, 2), sharpe=round(r.mean() / r.std() * np.sqrt(252), 2), mdd=round(float((nav / np.maximum.accumulate(nav) - 1).min()) * 100, 1))
def ix(y): return int(np.searchsorted(dates.values, np.datetime64(f"{y}-01-01")))
if __name__ == "__main__":
    y0 = 2003 if U == "etf" else 2005; cycles = [(y0 + 2 * k, y0 + 2 + 2 * k, y0 + 4 + 2 * k) for k in range(0, 11) if y0 + 4 + 2 * k <= 2025]
    c0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; c1 = int(sys.argv[4]) if len(sys.argv) > 4 else len(cycles)
    for ci, (ytr, yva, yte) in list(enumerate(cycles))[c0:c1]:
        a, b, c, d = ix(ytr), ix(yva), ix(yte), min(ix(yte + 2), T); a = max(a, LOOK + 1); res = {}
        for algo in ALGOS:
            for kind in KINDS:
                env = Env(kind, a, b); torch.manual_seed(0)
                m = PPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=6, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, seed=0, verbose=0, device="cpu") if algo == "ppo" else A2C("MlpPolicy", env, n_steps=16, learning_rate=7e-4, ent_coef=0.01, gamma=0.99, seed=0, verbose=0, device="cpu")
                m.learn(STEPS); rv, _, _ = rollout(m, kind, b, c); rt, acts, tov = rollout(m, kind, c, d)
                res[f"{algo}_{kind}"] = dict(val_sharpe=float(rv.mean() / (rv.std() + 1e-12) * np.sqrt(252)), test=[round(float(x), 6) for x in rt], acts=acts, turnover=round(tov / ((d - c) / 252), 2))
                print(U, ci, yte, algo, kind, "val", round(res[f"{algo}_{kind}"]["val_sharpe"], 2), "test", stat(rt), "avg λ", round(np.mean([x[1] for x in acts]), 1), "avg h", round(np.mean([x[2] for x in acts]), 1), flush=True)
        json.dump(dict(cycle=(ytr, yva, yte), dates=[str(x)[:10] for x in dates[c:d]], res=res), open(os.path.join(DATA, "us", f"st_{U}_{'-'.join(ALGOS)}_{ci}.json"), "w"))
