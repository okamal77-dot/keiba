"""３連単を当てるための条件分析とフォーメーションのバックテスト（2024〜2026年10月・ばんえい除く14場）.

レース前に分かる条件でレースを分け, ３連単の決まり方と買い方ごとの的中率・回収率を比べる.
  1番人気の位置取り : 近5走の4角位置 ÷ 頭数 の平均 (前 ~0.35 / 中 / 後 0.55~)
  1番人気が前の組か  : 位置取りのレース内順位 (前に行けそうな順) が 3位以内か
  1番人気の時計      : 近3走ベスト指数のレース内順位 (1位 / 2-3位 / 4位以下)
買い方 (1点100円, 人気は確定人気):
  A 基本       : 人気1・2 → 1〜4 → 1〜5 (18点)
  B 1人気3着   : 人気2・3 → 2〜4 → 1〜5 (1番人気を3着候補に回す, 18点)
  C 前の馬中心 : 1〜6番人気のうち前に行けそうな3頭 F. F → F+1番人気 → F+1番人気+人気上位2頭 (点数は可変)
  D 1人気消し  : 人気2〜5 の BOX (24点)
  D2/D3      : 1人気消し 人気2〜4 BOX (6点) / 人気2〜6 BOX (60点)
  E ３連複     : 人気1〜5 の BOX (10点, ３連複の払戻)
  F/F2 ３連複  : 1人気消し 人気2〜5 BOX (4点) / 人気2〜6 BOX (10点)
出力: analysis/out/tt_*.csv
"""
import numpy as np
import pandas as pd

import ability_index as A
import trifecta_factors as T

KEY = T.KEY


def pays():
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True)
    t = raw[KEY + ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]].dropna()
    f = raw[KEY + ["３連複組番馬番1", "３連複組番馬番2", "３連複組番馬番3", "３連複払戻金（円）"]].dropna()
    tri, fuku = {}, {}
    for r in t.itertuples(index=False):
        tri.setdefault(tuple(r[:3]), {})[tuple(int(x) for x in r[3:6])] = r[6]
    for r in f.itertuples(index=False):
        fuku.setdefault(tuple(r[:3]), {})[tuple(sorted(int(x) for x in r[3:6]))] = r[6]
    return tri, fuku


def combos(A_, B_, C_):
    return [(a, b, c) for a in A_ for b in B_ for c in C_ if len({a, b, c}) == 3]


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    d = h[(h["年"] >= 2024) & h["人気"].notna() & (h["近走数"] >= 1)].copy()
    d["前順"] = d.groupby(KEY)["位置取り"].rank(method="first")
    tri, fuku = pays()

    rows = []
    for k, g in d.groupby(KEY, sort=False):
        if k not in tri or len(g) < 7 or g["人気"].min() != 1:
            continue
        g = g.sort_values("人気")
        pop = g["馬番"].astype(int).tolist()
        fav = g.iloc[0]
        top6 = g.head(6)
        F = top6.sort_values("位置取り")["馬番"].astype(int).tolist()[:3]
        fv = int(fav["馬番"])
        res = sorted(g.dropna(subset=["着順"]).sort_values("着順")["馬番"].astype(int).tolist()[:3])
        win = list(tri[k].keys())[0]
        pos = dict(zip(g["馬番"].astype(int), g["着順"]))
        bets = {
            "A 基本 (人気1・2→1〜4→1〜5)": combos(pop[:2], pop[:4], pop[:5]),
            "B 1人気を3着に (2・3→2〜4→1〜5)": combos(pop[1:3], pop[1:4], pop[:5]),
            "C 前の馬中心": combos(F, list(dict.fromkeys(F + [fv])), list(dict.fromkeys(F + [fv] + pop[:2] + pop[2:3]))),
            "D 1人気消し (2〜5 BOX)": combos(pop[1:5], pop[1:5], pop[1:5]),
            "D2 1人気消し (2〜4 BOX)": combos(pop[1:4], pop[1:4], pop[1:4]),
            "D3 1人気消し (2〜6 BOX)": combos(pop[1:6], pop[1:6], pop[1:6]),
        }
        rec = {
            "競馬場": k[0], "日付": k[1], "R": k[2], "頭数": len(g),
            "1人気の位置取り": fav["位置取り"], "1人気の前順": fav["前順"], "1人気の指数順": fav["指数順"],
            "1人気の着順": fav["着順"], "３連単": tri[k][win], "前組の3着内数": sum(pos.get(x, 99) <= 3 for x in F),
            "上位5人気で決着": all(x in pop[:5] for x in win),
        }
        for name, cs in bets.items():
            hit = win in cs
            rec[name + "|点数"] = len(cs)
            rec[name + "|払戻"] = tri[k][win] if hit else 0
        box = tuple(sorted(pop[:5]))
        hitf = tuple(sorted(win)) in [tuple(sorted(c)) for c in combos(box, box, box)]
        rec["E ３連複 (人気1〜5 BOX)|点数"] = 10
        rec["E ３連複 (人気1〜5 BOX)|払戻"] = sum(v for c, v in fuku.get(k, {}).items() if set(c) <= set(box)) if hitf else 0
        for name, b2 in [("F ３連複 1人気消し (2〜5 BOX)", pop[1:5]), ("F2 ３連複 1人気消し (2〜6 BOX)", pop[1:6])]:
            n = len(b2)
            rec[name + "|点数"] = n * (n - 1) * (n - 2) // 6
            rec[name + "|払戻"] = sum(v for c, v in fuku.get(k, {}).items() if set(c) <= set(b2))
        rows.append(rec)
    r = pd.DataFrame(rows)
    r.to_csv(T.OUT / "tt_rows.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    r["1人気タイプ"] = pd.cut(r["1人気の位置取り"], [-1, 0.35, 0.55, 2], labels=["前(逃げ・先行)", "中", "後(差し・追込)"])
    r["1人気が前の組"] = np.where(r["1人気の前順"] <= 3, "前の組(3位以内)", "前の組に入らない")
    r["1人気の時計"] = pd.cut(r["1人気の指数順"], [0, 1, 3, 99], labels=["1位", "2-3位", "4位以下"])
    r["区分"] = np.select(
        [(r["1人気の前順"] > 3) & (r["1人気の指数順"] >= 4), (r["1人気の前順"] > 3), (r["1人気の指数順"] >= 4)],
        ["危険: 前の組でなく時計4位以下", "前の組に入らない (時計3位以内)", "前の組だが時計4位以下"], "信頼: 前の組で時計3位以内")

    def seg(g):
        out = {"レース数": len(g), "1人気の勝率%": (g["1人気の着順"] == 1).mean() * 100,
               "1人気の3着内率%": (g["1人気の着順"] <= 3).mean() * 100,
               "３連単の中央値": g["３連単"].median(), "上位5人気で決着%": g["上位5人気で決着"].mean() * 100,
               "前の組が2頭以上3着内%": (g["前組の3着内数"] >= 2).mean() * 100}
        for b in [c[:-3] for c in g.columns if c.endswith("|点数")]:
            cost = g[b + "|点数"].sum() * 100
            tag = b.split(" ")[0]
            out[tag + " 的中率%"] = (g[b + "|払戻"] > 0).mean() * 100
            out[tag + " 回収率%"] = g[b + "|払戻"].sum() / cost * 100
        out["C 平均点数"] = g["C 前の馬中心|点数"].mean()
        return pd.Series(out)

    res = []
    for col in ["区分", "1人気タイプ", "1人気が前の組", "1人気の時計"]:
        t = r.groupby(col, observed=True).apply(seg)
        t.index = pd.MultiIndex.from_product([[col], t.index])
        res.append(t)
    allr = seg(r).to_frame("全レース").T
    allr.index = pd.MultiIndex.from_tuples([("全体", "全レース")])
    res = pd.concat([allr] + res)
    res.to_csv(T.OUT / "tt_segments.csv", encoding="utf-8-sig")
    # 年別 (再現性): 区分ごとの 1人気3着内率 と 各買い方の回収率
    r["年"] = r["日付"] // 10000
    yr = r.groupby(["区分", "年"]).apply(seg)
    yr = yr[["レース数", "1人気の3着内率%"] + [c for c in yr.columns if c.endswith("回収率%")]]
    yr.to_csv(T.OUT / "tt_segments_by_year.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(res.round(1).to_string())
    print(yr.round(1).to_string())


if __name__ == "__main__":
    main()
