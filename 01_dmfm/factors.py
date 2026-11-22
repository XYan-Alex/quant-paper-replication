"""63 个风格因子（反转 / 动量 / 规模 / 价值 / 质量 五组，仿论文 4. Dataset Construction），按指数成分股切成训练张量。
财报按法定披露截止日对齐（一季报 5-1、中报 9-1、三季报 11-1、年报次年 5-1 起可用），避免用到未公布的数据。"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import numpy as np, pandas as pd, glob, warnings
warnings.filterwarnings("ignore")
D = DATA
Z = np.load(f"{D}/prices.npz", allow_pickle=True)
dates = pd.to_datetime(Z["dates"]); codes = [str(c) for c in Z["codes"]]; T, N = len(dates), len(codes)
num = [c[2:] for c in codes]; col = {c: i for i, c in enumerate(num)}
close, high, low, amount, factor, vwap = (Z[k].astype(np.float32) for k in ("close", "high", "low", "amount", "factor", "vwap"))
amount = amount * 1e3                                           # 千元 → 元
traded = np.isfinite(Z["close"]) & (np.nan_to_num(Z["volume"]) > 0)
C = pd.DataFrame(close).ffill(limit=20).values.astype(np.float64)                  # 停牌：价格沿用，最多 20 天

# ---------- 财报：季度 × 股票 ----------
qs = sorted(f[-16:-8] for f in glob.glob(f"{D}/fund/yjbb_*.parquet")); Q = len(qs)
def qmat(kind, field):
    M = np.full((Q, N), np.nan)
    for i, q in enumerate(qs):
        try: d = pd.read_parquet(f"{D}/fund/{kind}_{q}.parquet")
        except Exception: continue
        d = d[d["股票代码"].isin(col)].drop_duplicates("股票代码")
        M[i, [col[c] for c in d["股票代码"]]] = pd.to_numeric(d[field], errors="coerce").values
    return M
eps, rev, npf, bps, cfps, gm = (qmat("yjbb", f) for f in ("每股收益", "营业总收入-营业总收入", "净利润-净利润", "每股净资产", "每股经营现金流量", "销售毛利率"))
assets, liab, equity, cash, recv, inv = (qmat("zcfz", f) for f in ("资产-总资产", "负债-总负债", "股东权益合计", "资产-货币资金", "资产-应收账款", "资产-存货"))
ind = {}
for q in qs:
    d = pd.read_parquet(f"{D}/fund/yjbb_{q}.parquet")
    for c, v in zip(d["股票代码"], d["所处行业"]):
        if v and v != "None" and v != "nan": ind[c] = v
shares = equity / np.where(bps > 0.05, bps, np.nan)             # 股本 ≈ 股东权益 ÷ 每股净资产
shares = pd.DataFrame(shares).ffill(limit=2).values
cfo = cfps * shares
qn = np.array([int(q[4:6]) // 3 for q in qs])                    # 1..4
def sq(y):                                                       # 累计值 → 单季
    o = y.copy(); o[1:] = np.where((qn[1:] > 1)[:, None], y[1:] - y[:-1], y[1:]); o[0] = np.nan if qn[0] > 1 else y[0]; return o
def ttm(y):                                                      # 累计值 → 最近四个季度合计
    o = np.full_like(y, np.nan)
    for i in range(4, Q):
        o[i] = y[i] if qn[i] == 4 else y[i] + y[i - qn[i]] - y[i - 4]
    return o
def lag(y, k): o = np.full_like(y, np.nan); o[k:] = y[:-k]; return o
np_t, rev_t, cfo_t = ttm(npf), ttm(rev), ttm(cfo); np_q, rev_q, cfo_q = sq(npf), sq(rev), sq(cfo)
roe_t = np_t / np.where(equity > 0, equity, np.nan); nm_t = np_t / np.where(rev_t > 0, rev_t, np.nan); at_t = rev_t / assets; lev = liab / assets
g = lambda a, b: (a - b) / np.where(np.abs(b) > 0, np.abs(b), np.nan)
# 每个交易日能看到的最新一季
avail = [pd.Timestamp(f"{int(q[:4]) + (1 if q[4:] == '1231' else 0)}-{ {'0331': '05', '0630': '09', '0930': '11', '1231': '05'}[q[4:]] }-01") for q in qs]
order = np.argsort(avail, kind="stable")                         # 年报和一季报同日可用：取后者（一季报）
qi = np.full(T, -1)
for j in order: qi[dates >= avail[j]] = np.maximum(qi[dates >= avail[j]], j)
ok = qi >= 0
def daily(M):
    o = np.full((T, N), np.nan); Mf = pd.DataFrame(M).ffill(limit=1).values; o[ok] = Mf[qi[ok]]; return o
# 市值：复权价 × (股本 ÷ 季末复权因子)，拆股不跳变
qend = [min(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(q)), "right") - 1, T - 1) for q in qs]
Ff = pd.DataFrame(factor).ffill().bfill().values
S = shares / np.stack([Ff[max(e, 0)] for e in qend])
cap = C * daily(S)

# ---------- 行情类 ----------
R = pd.DataFrame(C).pct_change(fill_method=None); Cd = pd.DataFrame(C)
ret = lambda k: (Cd / Cd.shift(k) - 1).values
roll = lambda X, k, f: getattr(pd.DataFrame(X).rolling(k, min_periods=int(k * .6)), f)().values
amt = np.where(traded, amount, np.nan)
vol20, vol60, vol120 = roll(R, 20, "std"), roll(R, 60, "std"), roll(R, 120, "std")
turn = amt / cap
lncap = np.log(cap)
MM = np.lib.format.open_memmap(f"{D}/F_all.npy", "w+", np.float32, (63, T, N)); F, G = {}, {}
def add(name, grp, X):
    j = len(F); MM[j] = np.asarray(X, np.float32); F[name] = j; G[name] = grp
class _F(dict):
    def __getitem__(s, n): return MM[dict.__getitem__(s, n)]
F = _F()
add("rev_5", "revs", -ret(5)); add("rev_10", "revs", -ret(10)); add("rev_20", "revs", -ret(20))
add("rev_vwap20", "revs", -(C / roll(np.where(traded, vwap, np.nan), 20, "mean") - 1))
add("lowmax_20", "revs", -roll(R, 20, "max")); add("lowvol_20", "revs", -vol20); add("lowvol_60", "revs", -vol60)
add("lowturn_20", "revs", -roll(turn, 20, "mean")); add("lowabturn", "revs", -roll(turn, 5, "mean") / roll(turn, 60, "mean"))
add("illiq_20", "revs", roll(np.abs(R.values) / amt * 1e8, 20, "mean")); add("lowskew_20", "revs", -roll(R, 20, "skew"))
add("lowrange_20", "revs", -roll((high - low) / close, 20, "mean"))
add("mom_60", "mon", ret(60)); add("mom_120", "mon", ret(120)); add("mom_250", "mon", ret(250))
add("mom_120_1", "mon", (Cd.shift(20) / Cd.shift(120) - 1).values); add("mom_250_1", "mon", (Cd.shift(20) / Cd.shift(250) - 1).values)
add("high_250", "mon", C / roll(C, 250, "max")); add("mom_120_adj", "mon", ret(120) / (vol120 * np.sqrt(120)))
add("up_120", "mon", roll((R > 0).astype(float).where(R.notna()), 120, "mean")); add("ma_20_120", "mon", roll(C, 20, "mean") / roll(C, 120, "mean") - 1)
add("ma_60_250", "mon", roll(C, 60, "mean") / roll(C, 250, "mean") - 1)
add("ln_cap", "size", lncap); add("ln_amt_20", "size", np.log(roll(amt, 20, "mean")))
add("ln_rev", "size", np.log(daily(np.where(rev_t > 0, rev_t, np.nan)))); add("ln_assets", "size", np.log(daily(assets)))
add("nl_size", "size", lncap ** 3)                               # 下面按截面对 ln_cap 正交
d = daily
add("ep_ttm", "value", d(np_t) / cap); add("bp", "value", d(equity) / cap); add("sp_ttm", "value", d(rev_t) / cap); add("cfp_ttm", "value", d(cfo_t) / cap)
add("ep_q", "value", 4 * d(np_q) / cap); add("sp_q", "value", 4 * d(rev_q) / cap); add("cfp_q", "value", 4 * d(cfo_q) / cap)
add("dq_ep", "value", d(np_t - lag(np_t, 1)) / cap); add("dy_ep", "value", d(np_t - lag(np_t, 4)) / cap)
add("dq_sp", "value", d(rev_t - lag(rev_t, 1)) / cap); add("dy_sp", "value", d(rev_t - lag(rev_t, 4)) / cap)
add("dq_bp", "value", d(equity - lag(equity, 1)) / cap); add("dy_bp", "value", d(equity - lag(equity, 4)) / cap)
add("dy_cfp", "value", d(cfo_t - lag(cfo_t, 4)) / cap); add("cash_p", "value", d(cash) / cap); add("ap", "value", d(assets) / cap)
add("roe_ttm", "qual", d(roe_t)); add("roe_q", "qual", d(np_q / np.where(equity > 0, equity, np.nan))); add("roa_ttm", "qual", d(np_t / assets))
add("gm", "qual", d(gm)); add("nm_ttm", "qual", d(nm_t)); add("at_ttm", "qual", d(at_t)); add("low_lev", "qual", -d(lev))
add("cash_quality", "qual", d((cfo_t - np_t) / assets)); add("rev_q_yoy", "qual", d(g(rev_q, lag(rev_q, 4)))); add("np_q_yoy", "qual", d(g(np_q, lag(np_q, 4))))
add("rev_ttm_yoy", "qual", d(g(rev_t, lag(rev_t, 4)))); add("np_ttm_yoy", "qual", d(g(np_t, lag(np_t, 4))))
add("dq_roe", "qual", d(roe_t - lag(roe_t, 1))); add("dy_roe", "qual", d(roe_t - lag(roe_t, 4)))
add("dq_gm", "qual", d(gm - lag(gm, 1))); add("dy_gm", "qual", d(gm - lag(gm, 4))); add("dy_nm", "qual", d(nm_t - lag(nm_t, 4)))
add("dy_at", "qual", d(at_t - lag(at_t, 4))); add("dy_lowlev", "qual", -d(lev - lag(lev, 4))); add("wc_eff", "qual", -d((np.nan_to_num(recv) + np.nan_to_num(inv)) / np.where(rev_t > 0, rev_t, np.nan)))
names = list(F); groups = [G[n] for n in names]; assert len(names) == 63, len(names)
print("factors", len(names), pd.Series(groups).value_counts().to_dict())

MM.flush(); import gc; del vol20, vol60, vol120, turn, R, amt; gc.collect()
# ---------- 行业（申万一级，按二级名称归并；现行分类，非历史时点） ----------
SW = {"农林牧渔": "种植业 渔业 林业Ⅱ 饲料 农产品加工 养殖业 动物保健Ⅱ 农业综合Ⅱ", "基础化工": "化学原料 化学制品 化学纤维 塑料 橡胶 农化制品 非金属材料Ⅱ", "钢铁": "冶钢原料 普钢 特钢Ⅱ",
 "有色金属": "金属新材料 工业金属 贵金属 小金属 能源金属", "电子": "半导体 元件 光学光电子 其他电子Ⅱ 消费电子 电子化学品Ⅱ", "汽车": "汽车零部件 汽车服务 摩托车及其他 乘用车 商用车",
 "家用电器": "白色家电 黑色家电 小家电 厨卫电器 照明设备Ⅱ 家电零部件Ⅱ 其他家电Ⅱ", "食品饮料": "食品加工 白酒Ⅱ 非白酒 饮料乳品 休闲食品 调味发酵品Ⅱ", "纺织服饰": "纺织制造 服装家纺 饰品",
 "轻工制造": "造纸 包装印刷 家居用品 文娱用品", "医药生物": "化学制药 中药Ⅱ 生物制品 医药商业 医疗器械 医疗服务", "公用事业": "电力 燃气Ⅱ", "交通运输": "物流 铁路公路 航空机场 航运港口",
 "房地产": "房地产开发 房地产服务", "商贸零售": "贸易Ⅱ 一般零售 专业连锁Ⅱ 互联网电商 旅游零售Ⅱ", "社会服务": "体育Ⅱ 专业服务 酒店餐饮 旅游及景区 教育", "银行": "银行Ⅱ 国有大型银行Ⅱ 股份制银行Ⅱ 城商行Ⅱ 农商行Ⅱ",
 "非银金融": "证券Ⅱ 保险Ⅱ 多元金融", "综合": "综合Ⅱ", "建筑材料": "水泥 玻璃玻纤 装修建材", "建筑装饰": "房屋建设Ⅱ 装修装饰Ⅱ 基础建设 专业工程 工程咨询服务Ⅱ",
 "电力设备": "电机Ⅱ 其他电源设备Ⅱ 光伏设备 风电设备 电池 电网设备", "机械设备": "通用设备 专用设备 轨交设备Ⅱ 工程机械 自动化设备", "国防军工": "航天装备Ⅱ 航空装备Ⅱ 地面兵装Ⅱ 航海装备Ⅱ 军工电子Ⅱ",
 "计算机": "计算机设备 IT服务Ⅱ 软件开发", "传媒": "游戏Ⅱ 广告营销 影视院线 数字媒体 出版 电视广播Ⅱ", "通信": "通信服务 通信设备", "煤炭": "煤炭开采 焦炭Ⅱ", "石油石化": "油气开采Ⅱ 油服工程 炼化及贸易",
 "环保": "环境治理 环保设备Ⅱ", "美容护理": "个护用品 化妆品 医疗美容"}
l2 = {s: i for i, (k, v) in enumerate(SW.items()) for s in v.split()}
miss = sorted({v for v in ind.values() if v not in l2}); print("unmapped L2:", miss)
indus = np.array([l2.get(ind.get(c, ""), 31) for c in num]); print("industry coverage", (indus < 31).mean().round(3), "n_ind", len(set(indus)))

# ---------- 按股票池切片 ----------
t0 = int(np.searchsorted(dates.values, np.datetime64("2010-01-04"))); KS = [3, 5, 10, 15, 20]
fwd = {k: (Cd.shift(-k) / Cd - 1).values for k in KS}
for u in ("csi300", "csi500", "csi1000"):
    Mb = Z["m_" + u].copy(); full = np.argmax(Mb.sum(1) >= 0.9 * Mb.sum(1).max()); Mb[:full] = Mb[full]
    nmax = int(Mb[t0:].sum(1).max()); Tn = T - t0
    idx = np.full((Tn, nmax), -1, np.int32); X = np.zeros((Tn, nmax, 63), np.float32); Y = np.full((Tn, nmax, 5), np.nan, np.float32); cov = np.zeros(63)
    for i, t in enumerate(range(t0, T)):
        m = np.where(Mb[t] & np.isfinite(C[t]))[0]; idx[i, :len(m)] = m
        for j, n in enumerate(names):
            x = np.asarray(F[n][t, m], np.float64); x[~np.isfinite(x)] = np.nan
            if n == "nl_size":
                a = np.asarray(F["ln_cap"][t, m], np.float64); v = np.isfinite(a) & np.isfinite(x)
                if v.sum() > 10: b = np.polyfit(a[v], x[v], 1); x = x - (b[0] * a + b[1])
            v = np.isfinite(x); cov[j] += v.mean() / Tn
            if v.sum() > 10:
                lo, hi = np.percentile(x[v], [1, 99]); x = np.clip(x, lo, hi); x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
            X[i, :len(m), j] = np.nan_to_num(x)
        for j, k in enumerate(KS): Y[i, :len(m), j] = fwd[k][t, m]
    np.savez(f"{D}/panel_{u}.npz", idx=idx, X=X, Y=Y, dates=dates[t0:].values, indus=indus, names=np.array(names), groups=np.array(groups), first_full=str(dates[full].date()))
    print(u, X.shape, "membership real from", dates[full].date(), "min coverage", names[int(cov.argmin())], cov.min().round(3), "mean cov", cov.mean().round(3), flush=True)
np.savez(f"{D}/aux.npz", cap=cap[t0:].astype(np.float32), traded=traded[t0:], C=C[t0:], open=Z["open"][t0:])
