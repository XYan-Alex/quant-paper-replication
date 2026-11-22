import numpy as np, itertools
from scipy.stats import norm, skew, kurtosis
def sr(x): x=np.asarray(x); return x.mean()/x.std(ddof=1)
def pbo(M, S=16):
    """CSCV：M 为 T×N 的每期收益，按 Sharpe 选优。返回 PBO 和 logit 列表"""
    T,N=M.shape; T=T-T%S; M=M[-T:]; blocks=np.array_split(np.arange(T),S); L=[]
    for c in itertools.combinations(range(S),S//2):
        i=np.concatenate([blocks[k] for k in c]); o=np.concatenate([blocks[k] for k in range(S) if k not in c])
        si=np.array([sr(M[i,n]) for n in range(N)]); so=np.array([sr(M[o,n]) for n in range(N)])
        b=np.argmax(si); w=(np.argsort(np.argsort(so))[b]+1)/(N+1); L.append(np.log(w/(1-w)))
    L=np.array(L); return float((L<=0).mean()), L
def dsr(x, srs_all, N):
    """Bailey & López de Prado (2014)。x=被选方案的每期收益；srs_all=所有试过方案的每期 Sharpe；N=试验次数"""
    x=np.asarray(x); T=len(x); s=sr(x); g3=skew(x); g4=kurtosis(x,fisher=False); v=np.var(srs_all,ddof=1); e=0.5772156649
    sr0=np.sqrt(v)*((1-e)*norm.ppf(1-1/N)+e*norm.ppf(1-1/(N*np.e))) if N>1 else 0.0
    z=(s-sr0)*np.sqrt(T-1)/np.sqrt(1-g3*s+(g4-1)/4*s*s); return float(norm.cdf(z)), float(s), float(sr0), T
def monthly(nav, bench, dates):
    import pandas as pd
    a=pd.Series(nav,index=dates).resample('ME').last(); b=pd.Series(bench,index=dates).resample('ME').last()
    ra=a.pct_change(); rb=b.pct_change(); return (ra-rb).dropna()
