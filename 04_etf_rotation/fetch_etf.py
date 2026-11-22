import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import akshare as ak, pandas as pd, numpy as np, time
codes={'518880':'黄金','159985':'豆粕','513100':'纳指','510300':'沪深300','159915':'创业板','159992':'创新药','515700':'新能车','510150':'消费','515790':'光伏','515880':'通信','512720':'计算机','512660':'军工','159740':'恒生科技','511010':'国债','510500':'中证500','512880':'证券','512480':'半导体','512010':'医药','510880':'红利','512800':'银行','512690':'酒','516160':'新能源','512170':'医疗'}
for c,n in codes.items():
    try:
        df=ak.fund_etf_hist_sina(symbol=('sh' if c[0]=='5' else 'sz')+c); df['date']=pd.to_datetime(df['date'])
        r=df['close']/df['close'].shift(1); j=df.index[(r<0.7)|(r>1.4)]
        adj=np.ones(len(df))
        for k in j:   # 拆分/合并：之前的价格统一缩放
            f=round(1/r[k]) if r[k]<1 else 1/round(r[k]); 
            if f not in (1,): adj[:k]/=f if r[k]<1 else 1; 
            if r[k]>1: adj[:k]*=round(r[k])
        for col in ['open','high','low','close']: df[col]=df[col]*adj
        df.to_parquet(os.path.join(DATA,'etf',f'{c}.parquet')); print(c,n,len(df),str(df['date'].iloc[0])[:10],round(df['close'].iloc[-1],3),'splits',[(str(df['date'][k])[:10],round(r[k],3)) for k in j],flush=True)
    except Exception as e: print(c,'ERR',repr(e)[:80],flush=True)
    time.sleep(0.8)
