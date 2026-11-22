"""Deep Multi-Factor Model（arXiv:2210.12462）复现：模型、损失、半年滚动训练。
用法：python3 model.py csi300 [dmfm,mlp,mgat]
变体：FLIP_SIZE=1（规模因子取反，输出后缀 f）；MORE_EPOCHS=1（沪深300 更多训练，后缀 x）"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, sys, time, json
torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
U = sys.argv[1]; WHICH = (sys.argv[2] if len(sys.argv) > 2 else "dmfm,mlp,mgat").split(",")
D = DATA; Z = np.load(f"{D}/panel_{U}.npz", allow_pickle=True)
FLIP = os.environ.get("FLIP_SIZE") == "1"; MORE = os.environ.get("MORE_EPOCHS") == "1"; SUF = ("f" if FLIP else "") + ("x" if MORE else "")
Xn = Z["X"].copy()
if FLIP: Xn[:, :, Z["groups"] == "size"] *= -1                       # 变体 f：规模类因子取反
X = torch.from_numpy(Xn); Y = Z["Y"]; idx = Z["idx"]; dates = pd.to_datetime(Z["dates"]); indus = Z["indus"]
T, n, m = X.shape; KS = [3, 5, 10, 15, 20]; H = 32
valid = torch.from_numpy(idx >= 0)                                               # [T, n]
ind_id = np.where(idx >= 0, indus[np.clip(idx, 0, None)], -1)
Yz = np.full_like(Y, np.nan)                                                     # 未来收益：截面去极值 + 标准化
for j in range(5):
    y = Y[:, :, j]; lo, hi = np.nanpercentile(y, 1, axis=1, keepdims=True), np.nanpercentile(y, 99, axis=1, keepdims=True)
    y = np.clip(y, lo, hi); Yz[:, :, j] = (y - np.nanmean(y, 1, keepdims=True)) / (np.nanstd(y, 1, keepdims=True) + 1e-9)
Yz = torch.from_numpy(Yz); IND = torch.from_numpy(ind_id)

class GAT(nn.Module):                                                            # 单层图注意力：别的股票对我的“影响”
    def __init__(s, h, heads=2):
        super().__init__(); s.W = nn.Linear(h, h * heads, bias=False); s.a1 = nn.Parameter(torch.randn(heads, h) * .1); s.a2 = nn.Parameter(torch.randn(heads, h) * .1); s.heads = heads; s.h = h
    def forward(s, C, mask):                                                     # C [B,n,h]，mask [B,n,n] 谁和谁相连
        B, n, _ = C.shape; Wh = s.W(C).view(B, n, s.heads, s.h)
        e = Fn.leaky_relu((Wh * s.a1).sum(-1).permute(0, 2, 1).unsqueeze(-1) + (Wh * s.a2).sum(-1).permute(0, 2, 1).unsqueeze(-2), 0.2)   # [B,heads,n,n]
        att = torch.softmax(e.masked_fill(~mask.unsqueeze(1), -1e9), -1)
        return torch.matmul(att, Wh.permute(0, 2, 1, 3)).mean(1)

class DMFM(nn.Module):
    def __init__(s, m, h=H, kind="dmfm"):
        super().__init__(); s.kind = kind
        s.bn = nn.BatchNorm1d(m); s.enc = nn.Sequential(nn.Linear(m, h), nn.LeakyReLU(), nn.Linear(h, h))                              # 式(1)
        s.gi, s.gu = GAT(h), GAT(h)
        w = {"dmfm": 3, "mgat": 2, "mlp": 1}[kind]
        s.heads = nn.ModuleList([nn.Linear(h * w, 1) for _ in KS])                                                                      # 式(8)：每个持有期一个头
        s.fa = nn.ModuleList([nn.Linear(m, m) for _ in KS])                                                                             # 式(9)：因子注意力
    def forward(s, F, v, ind):
        B, n, m = F.shape; Fb = s.bn(F.reshape(-1, m)).view(B, n, m); C = s.enc(Fb)
        full = v.unsqueeze(1) & v.unsqueeze(2)
        if s.kind == "dmfm":
            same = (ind.unsqueeze(1) == ind.unsqueeze(2)) & full; same = same | torch.eye(n, dtype=torch.bool).unsqueeze(0)             # 式(2) 同行业才相连
            HI = s.gi(C, same); CI = C - HI                                                                                              # 式(3)(4) 行业中性
            HU = s.gu(CI, full | torch.eye(n, dtype=torch.bool).unsqueeze(0)); CU = CI - HU                                             # 式(5)(6) 全市场中性
            Z_ = torch.cat([C, CI, CU], -1)
        elif s.kind == "mgat":
            HU = s.gu(C, full | torch.eye(n, dtype=torch.bool).unsqueeze(0)); Z_ = torch.cat([C, HU], -1)
        else: Z_ = C
        f = torch.stack([Fn.leaky_relu(hd(Z_).squeeze(-1), 0.2) for hd in s.heads], -1)                                                  # [B,n,K]
        if s.kind != "dmfm": return f, None, None
        A = torch.stack([torch.softmax(Fn.leaky_relu(fa(Fb), 0.2), -1) for fa in s.fa], -2)                                              # [B,n,K,m] 式(10)
        abar = (A * v[..., None, None]).sum(1) / v.sum(1)[:, None, None]                                                                 # [B,K,m] 式(11)
        fhat = torch.einsum("bnm,bkm->bnk", Fb, abar)                                                                                    # 式(12)
        return f, fhat, abar

def zs(x, w):                                                                    # 带掩码的截面标准化
    cnt = w.sum(1, keepdim=True).clamp(min=1); mu = (x * w).sum(1, keepdim=True) / cnt; sd = (((x - mu) ** 2 * w).sum(1, keepdim=True) / cnt).sqrt() + 1e-6
    return (x - mu) / sd * w
def loss_fn(f, fhat, y, v):                                                      # 式(13)：d − 因子收益 − ICIR
    w = (v.unsqueeze(-1) & torch.isfinite(y)).float(); y0 = torch.nan_to_num(y) * w
    fz = zs(f, w); yz = zs(y0, w); ic = (fz * yz).sum(1) / w.sum(1).clamp(min=1)                                                        # [B,K]；收益已标准化 → 截面回归系数 b = IC
    icir = ic.mean(0) / (ic.std(0) + 1e-6); b = ic.mean(0)
    d = ((fz - zs(fhat, w)) ** 2).sum(1).div(w.sum(1).clamp(min=1)).sqrt().mean(0) if fhat is not None else torch.zeros_like(b)
    return (d - b - icir).mean(), ic.detach().mean(0), d.detach().mean()

def batches(days, bs):
    out = [days[i:i + bs] for i in range(0, len(days), bs)]
    if len(out) > 1 and len(out[-1]) < 8: out[-2] = np.concatenate([out[-2], out.pop()])          # 末尾不足 8 天的并进上一批（单日算不出 ICIR）
    return out
def run(model, days, bs, opt=None):
    tot, ics, ds, k = 0., 0., 0., 0
    for b in batches(days, bs):
        f, fhat, _ = model(X[b], valid[b], IND[b]); L, ic, d = loss_fn(f, fhat, Yz[b], valid[b])
        if opt and torch.isfinite(L): opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += L.item(); ics += ic[-1].item(); ds += d.item(); k += 1
    return tot / k, ics / k, ds / k

def di(s): return int(np.searchsorted(dates.values, np.datetime64(s)))
# 滚动分组：训练到半年末 → 验证 5 个月 → 隔 1 个月 → 测试半年（论文 4. Dataset Construction）
groups = []
for y in range(2015, 2027):
    for h in (1, 2):
        tr_end = f"{y}-07-01" if h == 1 else f"{y + 1}-01-01"; va_end = f"{y}-12-01" if h == 1 else f"{y + 1}-06-01"
        te0 = f"{y + 1}-01-01" if h == 1 else f"{y + 1}-07-01"; te1 = f"{y + 1}-07-01" if h == 1 else f"{y + 2}-01-01"
        if di(te0) < T: groups.append((tr_end, va_end, te0, te1, f"{y}S{h}"))
groups = groups[1:]                                                              # 第一组：训练到 2015-06-30，测试 2016 上半年
BS = {"csi300": 64, "csi500": 48, "csi1000": 24}[U]
STRIDE, EP0, EP = {"csi300": (2, 6, 2), "csi500": (4, 6, 1), "csi1000": (5, 6, 1)}[U]
if MORE and U == "csi300": STRIDE, EP0, EP = 1, 12, 4                       # 变体 x：沪深300 用全部交易日、训练更多轮      # 训练隔几天取一天 / 首组轮数 / 之后每组轮数（算力所限）
out = {}
for kind in WHICH:
    model = DMFM(m, kind=kind); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    S = np.full((T, n), np.nan, np.float32); AB = {}; log = []
    for gi, (tr_end, va_end, te0, te1, tag) in enumerate(groups):
        t0 = time.time(); tr = np.arange(0, di(tr_end) - 20); va = np.arange(di(tr_end), di(va_end) - 20); te = np.arange(di(te0), min(di(te1), T))
        best, best_state = 1e9, None
        for ep in range(EP0 if gi == 0 else EP):
            model.train(); sub = np.random.permutation(tr[np.random.randint(STRIDE)::STRIDE]); trl, tric, _ = run(model, sub, BS, opt)                # 隔一天取一天（相邻两天几乎一样）
            model.eval()
            with torch.no_grad(): val, vic, vd = run(model, va, BS)
            if val < best or best_state is None: best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval(); ab = []
        with torch.no_grad():
            for b in batches(te, BS):
                f, fhat, abar = model(X[b], valid[b], IND[b]); S[b] = f[:, :, -1].numpy()                                                    # 20 日头，给月度调仓用
                if abar is not None: ab.append(abar[:, -1].numpy())
            _, tic, _ = run(model, te[:-20] if len(te) > 40 else te, BS) if te[-1] + 20 < T else (0, float("nan"), 0)
        if ab: AB[tag] = np.concatenate(ab).mean(0).tolist()
        log.append(dict(tag=tag, test=te0, val_loss=round(best, 4), val_ic=round(vic, 4), d=round(vd, 4), test_ic=round(tic, 4), sec=round(time.time() - t0)))
        print(U, kind, log[-1], flush=True)
    S[~valid.numpy()] = np.nan; out[kind] = S
    np.save(f"{D}/score_{U}_{kind}{SUF}.npy", S); json.dump(dict(log=log, attn=AB), open(f"{D}/log_{U}_{kind}{SUF}.json", "w"))
    torch.save(model.state_dict(), f"{D}/model_{U}_{kind}{SUF}.pt")
print("DONE", U)
