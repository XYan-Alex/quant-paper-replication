"""公式化因子引擎 + 遗传搜索（Alpha-GPT / QuantaAlpha 复现共用）"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import numpy as np, bottleneck as bn, random, warnings
warnings.filterwarnings("ignore")
Z=np.load(os.path.join(DATA,'prices.npz'),allow_pickle=True)
D=Z['dates'].astype('datetime64[D]'); t0=np.searchsorted(D,np.datetime64('2014-01-01'))
D=D[t0:]
mem=(Z['m_csi300'][t0:]>0)|(Z['m_csi500'][t0:]>0)|(Z['m_csi1000'][t0:]>0)
cols=np.where(mem.any(0))[0]; mem=mem[:,cols]
def L(k): return Z[k][t0:][:,cols].astype(np.float32)
fac=L('factor'); 
F={k:L(k)*fac for k in ['open','high','low','close','vwap']}
F['volume']=L('volume')/np.where(fac>0,fac,np.nan); F['amount']=L('amount')
for k in F: F[k][~np.isfinite(F[k])|(F[k]<=0)]=np.nan
F['ret']=F['close']/np.roll(F['close'],1,0)-1; F['ret'][0]=np.nan
T,N=F['close'].shape
import os
c=F['close']; H=int(os.environ.get('AG_H','5'))
fwd=np.full((T,N),np.nan,np.float32); fwd[:T-H-1]=c[H+1:]/c[1:T-H]-1   # t+1收盘 → t+6收盘
fwd[~mem]=np.nan
# 收益先对“成交额规模”(60日均成交额的截面排名)做中性化：避免搜索退化成小盘/低流动性因子
_sz=bn.move_mean(np.log(F['amount']),60,axis=0,min_count=20)
for _i in range(T):
    m=np.isfinite(fwd[_i])&np.isfinite(_sz[_i])
    if m.sum()<200: fwd[_i]=np.nan; continue
    r=bn.rankdata(_sz[_i][m]); r=(r-r.mean())/r.std(); y=fwd[_i][m]; b=(r*(y-y.mean())).mean()
    fwd[_i][m]=y-y.mean()-b*r; fwd[_i][~m]=np.nan
IS=(D>=np.datetime64('2016-01-01'))&(D<np.datetime64('2022-01-01')); OS=(D>=np.datetime64('2022-01-01'))&(D<np.datetime64('2026-11-01'))
# ---------- operators
def _w(d): return max(2,int(d))
def ts_mean(x,d): return bn.move_mean(x,_w(d),axis=0,min_count=max(2,_w(d)//2))
def ts_std(x,d): return bn.move_std(x,_w(d),axis=0,min_count=max(2,_w(d)//2))
def ts_sum(x,d): return bn.move_sum(x,_w(d),axis=0,min_count=max(2,_w(d)//2))
def ts_min(x,d): return bn.move_min(x,_w(d),axis=0,min_count=max(2,_w(d)//2))
def ts_max(x,d): return bn.move_max(x,_w(d),axis=0,min_count=max(2,_w(d)//2))
def ts_rank(x,d): return bn.move_rank(x,_w(d),axis=0,min_count=max(2,_w(d)//2))
def shift(x,d):
    d=_w(d) if d>=2 else 1; y=np.full_like(x,np.nan); y[d:]=x[:-d]; return y
def ts_delta(x,d): return x-shift(x,d)
def ts_delta_ratio(x,d): return x/shift(x,d)-1
def ts_zscore(x,d): return (x-ts_mean(x,d))/(ts_std(x,d)+1e-9)
def ts_ir(x,d): return ts_mean(x,d)/(ts_std(x,d)+1e-9)
def ts_maxmin_scale(x,d):
    a=ts_min(x,d); b=ts_max(x,d); return (x-a)/(b-a+1e-9)
def ts_corr(x,y,d):
    d=_w(d); mx=ts_mean(x,d); my=ts_mean(y,d)
    return (ts_mean(x*y,d)-mx*my)/(ts_std(x,d)*ts_std(y,d)+1e-9)
def ts_decay(x,d):
    d=_w(d); out=np.zeros_like(x); 
    for i in range(d): out+=shift(x,i)*(d-i) if i>0 else x*d
    return out/(d*(d+1)/2)
def cs_rank(x):
    r=bn.nanrankdata(np.where(mem,x,np.nan),axis=1); n=np.sum(np.isfinite(r),1,keepdims=True); return (r-0.5)/np.maximum(n,1)
def cs_zscore(x):
    x=np.where(mem,x,np.nan); return (x-np.nanmean(x,1,keepdims=True))/(np.nanstd(x,1,keepdims=True)+1e-9)
def add(x,y): return x+y
def sub(x,y): return x-y
def mul(x,y): return x*y
def div(x,y): return x/np.where(np.abs(y)>1e-9,y,np.nan)
def neg(x): return -x
def absv(x): return np.abs(x)
def log(x): return np.log(np.where(x>0,x,np.nan))
def sign(x): return np.sign(x)
OPS1T={'ts_mean':ts_mean,'ts_std':ts_std,'ts_sum':ts_sum,'ts_min':ts_min,'ts_max':ts_max,'ts_rank':ts_rank,'shift':shift,'ts_delta':ts_delta,
       'ts_delta_ratio':ts_delta_ratio,'ts_zscore':ts_zscore,'ts_ir':ts_ir,'ts_maxmin_scale':ts_maxmin_scale,'ts_decay':ts_decay}
OPS2T={'ts_corr':ts_corr}
OPS1={'cs_rank':cs_rank,'cs_zscore':cs_zscore,'neg':neg,'absv':absv,'log':log,'sign':sign}
OPS2={'add':add,'sub':sub,'mul':mul,'div':div}
ENV=dict(F); ENV.update(OPS1T); ENV.update(OPS2T); ENV.update(OPS1); ENV.update(OPS2)
WINS=[3,5,10,20,40,60]
# ---------- tree <-> str.  tree = ('op',args...) | 'field' | int
import ast
def parse(s):
    def cv(n):
        if isinstance(n,ast.Call): return (n.func.id,)+tuple(cv(a) for a in n.args)
        if isinstance(n,ast.Name): return n.id
        if isinstance(n,ast.Constant): return n.value
        if isinstance(n,ast.UnaryOp): return ('neg',cv(n.operand))
        raise ValueError(ast.dump(n))
    return cv(ast.parse(s,mode='eval').body)
def tostr(t):
    if isinstance(t,tuple): return t[0]+'('+','.join(tostr(a) for a in t[1:])+')'
    return str(t)
def ev(t):
    if isinstance(t,tuple): return ENV[t[0]](*[ev(a) for a in t[1:]])
    if isinstance(t,str): return F[t]
    return t
def size(t): return 1+sum(size(a) for a in t[1:]) if isinstance(t,tuple) else 1
STEP=int(os.environ.get('AG_STEP','3'))
def ic_series(x,idx):
    """每 STEP 天取一天的截面 Rank IC"""
    out=[]
    for i in np.where(idx)[0][::STEP]:
        a=x[i]; b=fwd[i]; m=np.isfinite(a)&np.isfinite(b)
        if m.sum()<200: continue
        ra=bn.rankdata(a[m]); rb=bn.rankdata(b[m])
        if ra.std()<1e-6: continue
        out.append(np.corrcoef(ra,rb)[0,1])
    return np.array(out)
CACHE={}
def fitness(t):
    s=tostr(t)
    if s in CACHE: return CACHE[s]
    try:
        x=ev(t); x=np.where(mem,x,np.nan).astype(np.float32)
        cov=np.isfinite(x[IS]).sum()/max(1,mem[IS].sum())
        a=ic_series(x,IS); b=ic_series(x,OS)
        r=(float(np.mean(a)),float(np.mean(a)/(np.std(a)+1e-9)),float(np.mean(b)),float(np.mean(b)/(np.std(b)+1e-9))) if cov>0.6 and len(a)>50 else (0,0,0,0)
    except Exception as e: r=(0,0,0,0)
    CACHE[s]=r; return r
# ---------- GP
def subtrees(t,path=()):
    yield path,t
    if isinstance(t,tuple):
        for i,a in enumerate(t[1:],1): yield from subtrees(a,path+(i,))
def replace(t,path,new):
    if not path: return new
    l=list(t); l[path[0]]=replace(t[path[0]],path[1:],new); return tuple(l)
def is_panel(t): return isinstance(t,str) or (isinstance(t,tuple))
def rnd_leaf(): return random.choice(list(F))
def rnd_tree(depth=2):
    if depth==0 or random.random()<0.25: return rnd_leaf()
    k=random.random()
    if k<0.45: return (random.choice(list(OPS1T)),rnd_tree(depth-1),random.choice(WINS))
    if k<0.55: return ('ts_corr',rnd_tree(depth-1),rnd_tree(depth-1),random.choice(WINS))
    if k<0.75: return (random.choice(list(OPS1)),rnd_tree(depth-1))
    return (random.choice(list(OPS2)),rnd_tree(depth-1),rnd_tree(depth-1))
def mutate(t):
    nodes=list(subtrees(t)); path,n=random.choice(nodes); r=random.random()
    if isinstance(n,int): return replace(t,path,random.choice(WINS))
    if r<0.3: return replace(t,path,rnd_tree(2))                     # 子树替换
    if r<0.55 and isinstance(n,tuple):                                # 换同类算子
        for grp in (OPS1T,OPS1,OPS2):
            if n[0] in grp: return replace(t,path,(random.choice(list(grp)),)+n[1:])
    if r<0.8: return replace(t,path,(random.choice(['cs_rank','neg','ts_mean','ts_decay','ts_zscore']),n)+((random.choice(WINS),) if False else ()) ) if random.random()<0.5 else replace(t,path,(random.choice(['ts_mean','ts_decay','ts_zscore','ts_rank']),n,random.choice(WINS)))
    if isinstance(n,tuple):                                           # 提升子树
        ch=[a for a in n[1:] if not isinstance(a,int)]
        if ch: return replace(t,path,random.choice(ch))
    return replace(t,path,rnd_leaf())
def crossover(a,b):
    pa=[(p,n) for p,n in subtrees(a) if not isinstance(n,int)]; pb=[n for p,n in subtrees(b) if not isinstance(n,int)]
    p,_=random.choice(pa); return replace(a,p,random.choice(pb))
def search(seeds,gens=10,pop=24,log=None,seed=0):
    """Alpha-GPT 的搜索增强：从种子出发，每代变异/交叉，按样本内 |IC| 保留"""
    random.seed(seed); P=[parse(s) if isinstance(s,str) else s for s in seeds]
    hist=[]
    def key(t):
        f=fitness(t); return abs(f[0])-0.0004*size(t)
    for g in range(gens+1):
        P=sorted({tostr(t):t for t in P}.values(),key=key,reverse=True)[:pop]
        best=P[0]; f=fitness(best); top=[fitness(t) for t in P[:5]]
        hist.append(dict(gen=g,best=tostr(best),is_ic=abs(f[0]),os_ic=f[2]*np.sign(f[0]),top5_is=float(np.mean([abs(x[0]) for x in top])),top5_os=float(np.mean([x[2]*np.sign(x[0]) for x in top]))))
        if log: log(hist[-1])
        if g==gens: break
        kids=[]
        while len(kids)<pop:
            a=random.choice(P[:max(4,pop//2)])
            k=mutate(a) if random.random()<0.6 else crossover(a,random.choice(P[:max(4,pop//2)]))
            if size(k)<=22: kids.append(k)
        P=P+kids
    return P,hist
