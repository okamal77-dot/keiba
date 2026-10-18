"""1.5倍以下の1番人気を 1着に固定する買い方 と 1・2着どちらでもよい買い方 の比較 (2026年9月, 確定単勝オッズのある期間).

相手 a,b,c,d = 1番人気以外でモデル勝率 (交差検証の予想 pred2_underrated_rows_*.csv) の高い順.
出力: analysis/out/favfix_rows.csv
"""
import glob, itertools as it, pandas as pd, numpy as np
o=pd.concat([pd.read_csv(f,encoding='utf-8-sig',dtype={'競走年月日':int}) for f in glob.glob('data/odds/202609_*_odds.csv')])
o=o[o['賭式']=='単勝'][['競馬場','競走年月日','レース番号','番号1','オッズ']].rename(columns={'番号1':'馬番','オッズ':'単オッズ'})
o['馬番']=o['馬番'].astype(int)
print('odds dates',o['競走年月日'].min(),o['競走年月日'].max())
r=pd.concat([pd.read_csv(f) for f in glob.glob('analysis/out/pred2_underrated_rows_*.csv')])
print('oof dates',r['競走年月日'].min(),r['競走年月日'].max(), r['競馬場'].unique())
K=['競馬場','競走年月日','レース番号']
m=r.merge(o,on=K+['馬番'],how='inner')
p=pd.read_csv('data/202609_payback.csv',encoding='utf-8-sig')
print('races with odds',m.groupby(K).ngroups)
rows=[]
for k,g in m.groupby(K):
    g=g.sort_values('単オッズ')
    f=g.iloc[0]
    if not (f['単オッズ']<=1.5): continue
    pp=p[(p['競馬場']==k[0])&(p['競走年月日']==k[1])&(p['レース番号']==k[2])]
    if pp.empty or g['着順'].isna().all(): continue
    pp=pp.iloc[0]
    oth=g[g['馬番']!=f['馬番']].sort_values('勝率',ascending=False)['馬番'].astype(int).tolist()
    F=int(f['馬番']); a,b=oth[:2]; X4=oth[:4]
    fin=g.set_index('馬番')['着順']
    order=g.sort_values('着順')['馬番'].astype(int).tolist()[:3]
    umatan=(int(pp['馬単組番1']),int(pp['馬単組番2'])); um=float(pp['馬単払戻金（円）'])
    t3=(int(pp['３連単組番馬番1']),int(pp['３連単組番馬番2']),int(pp['３連単組番馬番3'])); tp=float(pp['３連単払戻金（円）'])
    def cost_ret(combos,kind):
        combos=set(combos)
        hit = (umatan if kind==2 else t3) in combos
        return len(combos)*100, (um if kind==2 else tp)*hit
    bets={}
    bets['馬単 F→a,b(2点)']=cost_ret([(F,a),(F,b)],2)
    bets['馬単 F⇄a,b(4点)']=cost_ret([(F,a),(F,b),(a,F),(b,F)],2)
    bets['3連単 F→ab→abcd(6点)']=cost_ret([(F,x,y) for x in (a,b) for y in X4 if y!=x],3)
    bets['3連単 F1-2着(12点)']=cost_ret([(F,x,y) for x in (a,b) for y in X4 if y!=x]+[(x,F,y) for x in (a,b) for y in X4 if y!=x],3)
    bets['3連単 ab→F→abcd(6点)']=cost_ret([(x,F,y) for x in (a,b) for y in X4 if y!=x],3)
    win_pay=float(pp['単勝払戻金（円）']) if int(pp['単勝組番'])==F else 0
    bets['単勝F']=(100,win_pay)
    d=dict(場=k[0],日=k[1],R=k[2],Fodds=f['単オッズ'],F勝率=f['勝率'],F着=fin[F],a=a,b=b)
    for n,(c,rt) in bets.items(): d[n+'_c']=c; d[n+'_r']=rt
    rows.append(d)
x=pd.DataFrame(rows)
x.to_csv('analysis/out/favfix_rows.csv',index=False,encoding='utf-8-sig')
names=[c[:-2] for c in x.columns if c.endswith('_c')]
def summ(y,label):
    print(f"\n== {label}: {len(y)}R  F勝率(実) {(y.F着==1).mean():.0%}  F2着 {(y.F着==2).mean():.0%}  F3着外 {(y.F着>3).mean():.0%}  モデルF勝率平均 {y.F勝率.mean():.0%}")
    for n in names:
        c=y[n+'_c'].sum(); r=y[n+'_r'].sum(); h=(y[n+'_r']>0).sum()
        print(f"  {n:22s} 的中 {h:3d}/{len(y)}  回収率 {r/c*100:5.0f}%")
summ(x,'1.5倍以下の1番人気 全体')
summ(x[x.F勝率<0.45],'うちモデル勝率45%未満 (市場より大幅に低い)')
summ(x[x.F勝率>=0.45],'うちモデル勝率45%以上')
