import window_exp_c2 as w
import numpy as np, pandas as pd, dolphindb as ddb
sess=ddb.session(w.HOST,w.PORT); sess.login(w.USER,w.PW)
J=120; K=600
parts=[]
for yr,mo in w.MONTHS:
    df=w.fetch(sess,'IC0000',yr,mo)
    if not len(df): continue
    df=w.prep(df)
    if df.empty: continue
    df=w.add_voi(df)
    f=w.rollsum(df,'voi',J)
    df['last_tk']=df['px']/w.TICK
    g=df.groupby('gid',sort=False)['last_tk']
    dy=g.shift(-K)-df['last_tk']; dy[dy.abs()>100]=np.nan
    df['f']=f.values; df['dy']=dy.values; df['yr']=df['ts'].dt.year
    parts.append(df[['yr','f','dy']].dropna())
    print(f"{yr}-{mo:02d} ok",flush=True)
sess.close()
a=pd.concat(parts,ignore_index=True)
q001=a['f'].quantile(0.001); q01=a['f'].quantile(0.01)
out=[f"全样本: 最负0.1% 阈值={q001:.0f}, 最负1% 阈值={q01:.0f}, 总行数={len(a):,}"]
for name,thr in [("最负0.1%",q001),("最负1%",q01)]:
    b=a[a['f']<=thr]
    out.append(f"\n=== VOI IC {name}  回看60s/持有300s  (全样本={b['dy'].mean():+.2f}, n={len(b):,}) ===")
    out.append(f"{'年':>6}{'均值dy':>10}{'样本n':>9}{'占桶%':>8}")
    for yr in sorted(a['yr'].unique()):
        by=b[b['yr']==yr]
        if len(by): out.append(f"{yr:>6}{by['dy'].mean():>+10.2f}{len(by):>9,}{100*len(by)/len(b):>7.1f}%")
open("_ic7.out","w").write("\n".join(out)); print("DONE")
