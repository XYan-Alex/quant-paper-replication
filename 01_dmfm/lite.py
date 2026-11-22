"""我们的改进版（DMFM-lite）：保留论文的两步中性化和“练排名”的训练目标，但做三处调整——
1) 同行 / 全市场的“影响”不再用图注意力去学，直接取等权平均（快两个数量级，能训练得充分）；
2) 打分头旁边加一条线性直连（63 个因子直接加权），深度部分只负责补充；
3) 每组多训几轮、三个随机种子取平均。
用法：python3 lite.py csi1000"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, sys, time, json
torch.set_num_threads(1)
U = sys.argv[1]; SEEDS = [0, 1, 2]; TAG = sys.argv[2] if len(sys.argv) > 2 else "lite"
D = DATA; Z = np.load(f"{D}/panel_{U}.npz", allow_pickle=True)
X = torch.from_numpy(Z["X"]); Y = Z["Y"]; idx = Z["idx"]; dates = pd.to_datetime(Z["dates"]); indus = Z["indus"]
PRIOR = TAG.startswith("lite2"); W0 = torch.tensor([-1.0 if g == "size" else 1.0 for g in Z["groups"]])            # 等权先验：63 个因子各 1 票（规模类取反）
T, n, m = X.shape; KS = [3, 5, 10, 15, 20]; H = 64; NI = 33
valid = torch.from_numpy(idx >= 0); IND = torch.from_numpy(np.where(idx >= 0, indus[np.clip(idx, 0, None)], 32)).long()
Yz = np.full_like(Y, np.nan)
for j in range(5):
    y = Y[:, :, j]; lo, hi = np.nanpercentile(y, 1, axis=1, keepdims=True), np.nanpercentile(y, 99, axis=1, keepdims=True)
    y = np.clip(y, lo, hi); Yz[:, :, j] = (y - np.nanmean(y, 1, keepdims=True)) / (np.nanstd(y, 1, keepdims=True) + 1e-9)
Yz = torch.from_numpy(Yz)
LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
def group_mean(C, ind, v):                                         # 每天、每个行业的平均画像，再发回给每只股票
    B, n, h = C.shape; key = (ind + torch.arange(B)[:, None] * NI).reshape(-1); w = v.reshape(-1, 1).float()
    s = torch.zeros(B * NI, h).index_add_(0, key, C.reshape(-1, h) * w); c = torch.zeros(B * NI, 1).index_add_(0, key, w)
    return (s / c.clamp(min=1))[key].view(B, n, h)
class Lite(nn.Module):
    def __init__(s):
        super().__init__(); s.bn = nn.BatchNorm1d(m); s.enc = nn.Sequential(nn.Linear(m, H), nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(H, H))
        s.heads = nn.Linear(3 * H, len(KS)); s.skip = nn.Linear(m, len(KS))
        if PRIOR:                                                    # 从“63 个因子等权”出发，深度部分初始为 0，只学修正
            nn.init.zeros_(s.heads.weight); nn.init.zeros_(s.heads.bias); s.skip.weight.data = W0.repeat(len(KS), 1).clone(); nn.init.zeros_(s.skip.bias)
    def forward(s, F, v, ind):
        B, n, _ = F.shape; Fb = s.bn(F.reshape(-1, m)).view(B, n, m); C = s.enc(Fb); w = v.unsqueeze(-1).float()
        CI = C - group_mean(C, ind, v)                               # 行业中性：减同行平均
        CU = C - (C * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)   # 市场中性：减全体平均
        return s.heads(torch.cat([C, CI, CU], -1)) + s.skip(Fb)
def zs(x, w):
    cnt = w.sum(1, keepdim=True).clamp(min=1); mu = (x * w).sum(1, keepdim=True) / cnt; sd = (((x - mu) ** 2 * w).sum(1, keepdim=True) / cnt).sqrt() + 1e-6
    return (x - mu) / sd * w
def loss_fn(f, y, v):
    w = (v.unsqueeze(-1) & torch.isfinite(y)).float(); ic = (zs(f, w) * zs(torch.nan_to_num(y) * w, w)).sum(1) / w.sum(1).clamp(min=1)
    return -(ic.mean(0) * 10 + ic.mean(0) / (ic.std(0) + 1e-6)).mean(), ic.detach().mean(0)[-1].item()
def batches(days, bs):
    out = [days[i:i + bs] for i in range(0, len(days), bs)]
    if len(out) > 1 and len(out[-1]) < 8: out[-2] = np.concatenate([out[-2], out.pop()])
    return out
def run(model, days, opt=None, bs=64):
    tot, ics, k = 0., 0., 0
    for b in batches(days, bs):
        L, ic = loss_fn(model(X[b], valid[b], IND[b]), Yz[b], valid[b])
        if PRIOR: L = L + LAM * ((model.skip.weight - W0) ** 2).mean() + LAM * 0.1 * (model.heads.weight ** 2).mean() * 100
        if opt is not None and torch.isfinite(L): opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += L.item(); ics += ic; k += 1
    return tot / k, ics / k
di = lambda s: int(np.searchsorted(dates.values, np.datetime64(s)))
groups = []
for y in range(2015, 2027):
    for h in (1, 2):
        tr_end = f"{y}-07-01" if h == 1 else f"{y + 1}-01-01"; va_end = f"{y}-12-01" if h == 1 else f"{y + 1}-06-01"
        te0 = f"{y + 1}-01-01" if h == 1 else f"{y + 1}-07-01"; te1 = f"{y + 1}-07-01" if h == 1 else f"{y + 2}-01-01"
        if di(te0) < T: groups.append((tr_end, va_end, te0, te1, f"{y}S{h}"))
groups = groups[1:]
S = np.zeros((T, n), np.float32); t00 = time.time()
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed); model = Lite(); opt = torch.optim.Adam(model.parameters(), lr=3e-4 if PRIOR else 1e-3, weight_decay=0 if PRIOR else 1e-4); log = []
    for gi, (tr_end, va_end, te0, te1, tag) in enumerate(groups):
        tr = np.arange(0, di(tr_end) - 20); va = np.arange(di(tr_end), di(va_end) - 20); te = np.arange(di(te0), min(di(te1), T)); best, bs_ = 1e9, None
        for ep in range(12 if gi == 0 else 3):
            model.train(); run(model, np.random.permutation(tr), opt); model.eval()
            with torch.no_grad(): val, vic = run(model, va)
            if val < best or bs_ is None: best, bs_ = val, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(bs_); model.eval()
        with torch.no_grad():
            for b in batches(te, 64):
                f = model(X[b], valid[b], IND[b])[:, :, -1]; w = valid[b].float(); S[b] += zs(f, w).numpy() / len(SEEDS)
        log.append((tag, round(vic, 4)))
    print(U, TAG, "seed", seed, "val_ic", [l[1] for l in log][::4], round(time.time() - t00), "s", flush=True)
S[~valid.numpy()] = np.nan; S[:di(groups[0][2])] = np.nan
np.save(f"{D}/score_{U}_{TAG}.npy", S); print("DONE")
