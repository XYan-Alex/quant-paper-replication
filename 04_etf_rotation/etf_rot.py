"""论文02：动量 × R² 打分 + 相关性收缩 + 均值方差，13只ETF月度轮动"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import pandas as pd, numpy as np, json, sys
from scipy.optimize import minimize
C=['518880','159985','513100','510300','159915','159992','515700','510150','515790','515880','512720','512660','159740']
def load(c):
    df=pd.read_parquet(os.path.join(DATA,'etf',f'{c}.parquet')).set_index('date')
    r=df['close']/df['close'].shift(1); adj=pd.Series(1.0,index=df.index)
    for d in r.index[(r<0.7)|(r>1.4)]: adj[adj.index<d]*=r[d]      # 拆分日当天按零收益处理
    return df['open']*adj, df['close']*adj
O={};CL={}
for c in C: O[c],CL[c]=load(c)
O=pd.DataFrame(O);CL=pd.DataFrame(CL)
def score(p,mode):
    y=np.log(p.values) if mode=='price' else np.diff(np.log(p.values))
    x=np.arange(len(y)); b,a=np.polyfit(x,y,1); res=y-(a+b*x); r2=1-res.var()/max(y.var(),1e-12)
    return (np.exp(b*252)-1)*r2, np.exp(b*252)-1
def weights(win,mode,gamma=2,delta=0.5,W=25):
    sc={};mu={}
    for c in win.columns:
        p=win[c].dropna()
        if len(p)<W: continue
        s,m=score(p.iloc[-W:],mode); 
        if s>0: sc[c]=s; mu[c]=m
    if not sc: return {}
    cs=list(sc); R=win[cs].iloc[-W:].pct_change().dropna(); S=R.cov().values*252; sd=np.sqrt(np.diag(S)); Rc=S/np.outer(sd,sd)
    Rs=delta*Rc+(1-delta)*np.eye(len(cs)); Ss=np.outer(sd,sd)*Rs; m=np.array([mu[c] for c in cs]); n=len(cs)
    f=lambda w:-(w@m-gamma/2*w@Ss@w)
    r=minimize(f,np.ones(n)/n,bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
    return {c:float(x) for c,x in zip(cs,r.x) if x>1e-4}
def run(start,end,mode='price',buy=0.0002,sell=0.0012,thr=0.01,**kw):
    days=CL.index[(CL.index>=start)&(CL.index<=end)]; nav=1.0; w={}; out=[]; hold=[]; tov=0
    prev_m=None
    for i,d in enumerate(days):
        pc=CL.loc[:d].iloc[-2] if len(CL.loc[:d])>1 else None
        if prev_m!=d.month:                      # 每月第一个交易日开盘调仓
            prev_m=d.month
            tgt=weights(CL.loc[:d].iloc[:-1],mode,**kw)
            # 隔夜：旧仓位 昨收→今开
            gap=sum(x*(O.loc[d,c]/pc[c]-1) for c,x in w.items()) if w else 0
            nav*=1+gap
            allc=set(w)|set(tgt); new={}
            for c in allc:
                a=w.get(c,0); b=tgt.get(c,0); new[c]=b if abs(b-a)>thr else a
            s=sum(new.values()); 
            if s>1: new={c:x/s for c,x in new.items()}
            cost=sum(max(new.get(c,0)-w.get(c,0),0)*buy+max(w.get(c,0)-new.get(c,0),0)*sell for c in allc); tov+=sum(abs(new.get(c,0)-w.get(c,0)) for c in allc)/2
            nav*=1-cost; w={c:x for c,x in new.items() if x>0}; hold.append((str(d)[:10],{c:round(x,2) for c,x in sorted(w.items(),key=lambda z:-z[1])}))
            nav*=1+sum(x*(CL.loc[d,c]/O.loc[d,c]-1) for c,x in w.items())
        else:
            nav*=1+sum(x*(CL.loc[d,c]/pc[c]-1) for c,x in w.items())
        # 权重漂移
        if w:
            base=O.loc[d] if hold and hold[-1][0]==str(d)[:10] else pc
            g={c:x*(CL.loc[d,c]/base[c]) for c,x in w.items()}; tot=sum(g.values())+(1-sum(w.values())); w={c:x/tot for c,x in g.items()}
        out.append(nav)
    s=pd.Series(out,index=days); return s,hold,tov
def stats(s):
    r=s.pct_change().dropna(); yrs=len(s)/252
    return dict(total=round((s.iloc[-1]/s.iloc[0]-1)*100,2),ann=round(((s.iloc[-1]/s.iloc[0])**(1/yrs)-1)*100,2),vol=round(r.std()*np.sqrt(252)*100,2),sharpe=round((r.mean()*252-0.0)/(r.std()*np.sqrt(252)),2),mdd=round((s/s.cummax()-1).min()*100,2))
if __name__=='__main__':
    res={}
    for tag,(a,b) in {'paper':('2021-06-01','2025-12-31'),'ext':('2021-06-01','2026-11-18'),'after':('2026-01-01','2026-11-18')}.items():
        bm=CL['510300'][(CL.index>=a)&(CL.index<=b)]; res[tag]={'bench300':stats(bm)}
        ew=CL[C][(CL.index>=a)&(CL.index<=b)].pct_change().mean(axis=1).add(1).cumprod(); res[tag]['ew13']=stats(ew)
        for mode in ['price','ret']:
            s,h,t=run(a,b,mode); res[tag][mode]=stats(s); res[tag][mode]['tov_per_yr']=round(t/(len(s)/252),2)
            if tag=='ext' and mode=='price': s.to_csv(os.path.join(DATA,'etf_rot_nav.csv')); json.dump(h,open(os.path.join(DATA,'etf_rot_hold.json'),'w'),ensure_ascii=False)
        print(tag,json.dumps(res[tag],ensure_ascii=False))
    json.dump(res,open(os.path.join(DATA,'etf_rot_res.json'),'w'),ensure_ascii=False)
