"""2026年11月 (11/1〜17) の締切オッズで, オッズの形のルールと1,000円の配分を確かめる (stake_alloc の関数を使う).
使い方: python3 analysis/nov_rule_check.py   (data/ に 202611_* と data/odds/202611_0x_odds.csv が要る)
"""
import sys; sys.argv=["x","/home/user/keiba/data"]
sys.path.insert(0,"/home/user/keiba/analysis")
import numpy as np, pandas as pd
import stake_alloc as SA
od,pay,jkn,names=SA.load()
rs=SA.races(od,pay,jkn,names)
AD={"1強":0.0,"2強":0.5,"3強":0.5,"4強":0.5}
ad=SA.simulate(rs,"1/確率比例",AD); eq=SA.simulate(rs,"均等",AD); fl=SA.simulate(rs,"1/確率比例(最低100円)",AD)
rows=[]
for r in rs:
    rows.append({"競走年月日":r["key"][1],"競馬場":r["key"][0],"形":r["形"],
      "投資3":len(r["tri"])*100,"払戻3":sum(r["t_pay"] for h in r["t_hit"] if h),
      "投資2":len(r["ex"])*100,"払戻2":sum(r["e_pay"] for h in r["e_hit"] if h)})
ru=pd.DataFrame(rows); ru["投資"]=ru["投資3"]+ru["投資2"]; ru["払戻"]=ru["払戻3"]+ru["払戻2"]
def roi(g,i="投資",p="払戻"): return g[p].sum()/max(g[i].sum(),1)*100
rng=np.random.default_rng(0)
def ci(d):
    i=d["投資"].to_numpy(); p=d["払戻"].to_numpy(); idx=rng.integers(0,len(d),(2000,len(d)))
    r=p[idx].sum(1)/np.maximum(i[idx].sum(1),1)*100; return "%.0f〜%.0f"%tuple(np.percentile(r,[5,95]))
print("対象", len(ru), ru["競走年月日"].min(), ru["競走年月日"].max())
print(pd.DataFrame([
 {"買い方":"決まりどおり 各100円","回収率":roi(ru),"90%区間":ci(ru),"３連単":roi(ru,"投資3","払戻3"),"馬単":roi(ru,"投資2","払戻2")},
 {"買い方":"1,000円 配分(今)","回収率":roi(ad),"90%区間":ci(ad),"３連単":roi(ad,"投資3","払戻3"),"馬単":roi(ad,"投資2","払戻2")},
 {"買い方":"1,000円 均等","回収率":roi(eq),"90%区間":ci(eq)},
 {"買い方":"1,000円 最低100円","回収率":roi(fl),"90%区間":ci(fl)}]).round(1).to_string(index=False))
print("\n== 形ごと (決まり各100円 / 配分)")
t=[]
for s in ["1強","2強","3強","4強"]:
    x=ru[ru["形"]==s]; y=ad[ad["形"]==s]
    t.append({"形":s,"レース":len(x),"３連単%":roi(x,"投資3","払戻3"),"３連単R":int((x["投資3"]>0).sum()),"馬単%":roi(x,"投資2","払戻2"),"馬単R":int((x["投資2"]>0).sum()),"配分%":roi(y)})
print(pd.DataFrame(t).round(0).to_string(index=False))
print("\n== 前半/後半")
for nm,mk in [("11/1-9",ru["競走年月日"]<=20261109),("11/10-17",ru["競走年月日"]>20261109)]:
    print(nm, len(ru[mk]), round(roi(ru[mk])), round(roi(ad[mk.values])))
print("\n== 競馬場")
print(pd.DataFrame([{"場":c,"R":len(x),"決まり%":roi(x),"配分%":roi(ad[ad["競馬場"]==c])} for c,x in ru.groupby("競馬場")]).round(0).to_string(index=False))
ru.to_csv("/home/user/keiba/analysis/out/nov_rule_races.csv",index=False,encoding="utf-8-sig")
