"""2〜3着の傾向 (第20節) で ▲△ (相手) を選び直すと ３連単・３連複 が良くなるかのバックテスト.

対象: 改良版モデルの検証期間 (直近3か月) の6場 (pred2_underrated_rows_*.csv の勝率).
◎○ (勝率1・2位) はそのまま. 3位以下の馬を「勝率 × 2〜3着補正」の順に並べ替えて ▲△△ を選ぶ.
2〜3着補正 (第20節の 2〜3着A/E をそのまま掛ける):
  前走の着順×時計: 前走2〜3着 速い1.06 / 中位1.05 / 遅い1.02, 前走4〜5着 速い1.07 / 中位1.07 / 遅い1.03,
                    前走6着以下 速い1.05 / 遅い0.86, 前走1着 遅い0.91 / 中位0.95
  前走の脚質: 差し1.06 / 先行1.03 / 逃げ0.92 / 追込0.92
  前走の上がり3F: 2〜3位 1.06 / 4位以下 0.97
買い方 (1点100円): ３連単 ◎○ → 上位4頭 → 上位5頭 (18点), ３連複 上位5頭BOX (10点)
出力: analysis/out/pb_*.csv
"""
import glob

import numpy as np
import pandas as pd

import trifecta_factors as T
import trifecta_tactics as TT

KEY = T.KEY


def mult(d, q):
    m = pd.Series(1.0, index=d.index)
    f, t = d["前走着順"], d["前走指数"]
    slow, fast = t <= q[0], t >= q[1]
    mid = ~slow & ~fast & t.notna()
    for cond, v in [((f >= 2) & (f <= 3) & fast, 1.06), ((f >= 2) & (f <= 3) & mid, 1.05), ((f >= 2) & (f <= 3) & slow, 1.02),
                    ((f >= 4) & (f <= 5) & fast, 1.07), ((f >= 4) & (f <= 5) & mid, 1.07), ((f >= 4) & (f <= 5) & slow, 1.03),
                    ((f >= 6) & fast, 1.05), ((f >= 6) & slow, 0.86), ((f == 1) & slow, 0.91), ((f == 1) & mid, 0.95)]:
        m[cond.fillna(False)] *= v
    m *= d["前走脚質"].map({"差し": 1.06, "先行": 1.03, "逃げ": 0.92, "追込": 0.92}).fillna(1.0)
    a = d["前走上がり順"]
    m[(a >= 2) & (a <= 3)] *= 1.06
    m[a >= 4] *= 0.97
    return m


def combos(A_, B_, C_):
    return [(a, b, c) for a in A_ for b in B_ for c in C_ if len({a, b, c}) == 3]


def main():
    h, _, _ = T.load()
    h = T.features(h)
    q = h.loc[h["年"] >= 2024, "指数"].quantile([1 / 3, 2 / 3]).values
    m = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "pred2_underrated_rows_*.csv"))])
    m = m.merge(h[KEY + ["馬番", "前走着順", "前走指数", "前走脚質", "前走上がり順"]], on=KEY + ["馬番"], how="left")
    m["補正"] = mult(m, q).values
    m["補正勝率"] = m["勝率"] * m["補正"]
    tri, fuku = TT.pays()
    rec = []
    for k, g in m.groupby(KEY):
        if k not in tri or len(g) < 6:
            continue
        g = g.sort_values("勝率", ascending=False)
        nos = g["馬番"].astype(int).tolist()
        ax = nos[:2]
        rest_old = nos[2:]
        rest_new = g.iloc[2:].sort_values("補正勝率", ascending=False)["馬番"].astype(int).tolist()
        win = list(tri[k].keys())[0]
        r = {"競馬場": k[0], "日付": k[1], "R": k[2], "◎勝率": g["勝率"].iloc[0], "結果": "-".join(map(str, win)),
             "３連単配当": tri[k][win]}
        for label, rest in [("いま", rest_old), ("2〜3着補正", rest_new)]:
            top4, top5 = ax + rest[:2], ax + rest[:3]
            cs = combos(ax, top4, top5)
            r[f"{label}|３連単払戻"] = tri[k][win] if win in cs else 0
            r[f"{label}|３連単点数"] = len(cs)
            box = set(top5)
            r[f"{label}|３連複払戻"] = sum(v for c, v in fuku.get(k, {}).items() if set(c) <= box)
            r[f"{label}|印5頭の3着内"] = len(set(win) & box)
        rec.append(r)
    d = pd.DataFrame(rec)
    d["◎勝率帯"] = pd.cut(d["◎勝率"], [0, 0.25, 0.4, 1.01], labels=["25%未満(３連複の場面)", "25-40%(３連単の場面)", "40%以上"])
    d.to_csv(T.OUT / "pb_rows.csv", index=False, encoding="utf-8-sig")

    def s(x):
        o = {"レース数": len(x)}
        for lb in ["いま", "2〜3着補正"]:
            o[f"{lb} ３連単的中%"] = (x[f"{lb}|３連単払戻"] > 0).mean() * 100
            o[f"{lb} ３連単回収率%"] = x[f"{lb}|３連単払戻"].sum() / (x[f"{lb}|３連単点数"].sum() * 100) * 100
            o[f"{lb} ３連複的中%"] = (x[f"{lb}|３連複払戻"] > 0).mean() * 100
            o[f"{lb} ３連複回収率%"] = x[f"{lb}|３連複払戻"].sum() / (len(x) * 1000) * 100
            o[f"{lb} 印5頭の3着内(平均)"] = x[f"{lb}|印5頭の3着内"].mean()
        return pd.Series(o)

    t = pd.concat([s(d).to_frame("全レース").T, d.groupby("◎勝率帯", observed=True).apply(s),
                   d.groupby("競馬場").apply(s)])
    t.to_csv(T.OUT / "pb_summary.csv", encoding="utf-8-sig")
    d["月"] = d["日付"] // 100
    mo = d.groupby("月").apply(s)[[c for c in t.columns if "回収率" in c]]
    mo.to_csv(T.OUT / "pb_by_month.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(1).T.to_string())
    print(mo.round(0).to_string())
    for lb in ["いま", "2〜3着補正"]:
        for b, cost in [("３連単", None), ("３連複", 1000)]:
            p = d[f"{lb}|{b}払戻"]
            c = d[f"{lb}|３連単点数"].sum() * 100 if b == "３連単" else len(d) * cost
            print(lb, b, "上位3本を除く回収率%", round((p.sum() - p.nlargest(3).sum()) / c * 100, 1))


if __name__ == "__main__":
    main()
