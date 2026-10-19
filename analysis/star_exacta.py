"""本命が堅いレースで ◎→★ (過小評価候補) を2着に置く買い方のバックテスト.

対象: 改良版モデル (predict_v2.py) の検証期間 (直近3か月) の6場. 入力は pred2_underrated_rows_*.csv.
  ◎ = 勝率1位, ○▲ = 勝率2・3位, ★ = 消し以外で 勝率順5位以内 かつ 人気が勝率順より2つ以上下.
  ★の人気は 想定人気 (発売前) と 確定人気 の2通りで判定する.
買い方 (1点100円, 払戻は確定値):
  単勝◎ / 馬連 ◎-○ / 馬連 ◎-★ / 単勝◎＋馬連 ◎-○ / 単勝◎＋馬連 ◎-★ /
  単勝◎ / 馬単 ◎→○ / 馬単 ◎→★ / ３連単 ◎→○▲→○▲ / ３連単 ◎→○▲★→○▲★ / ３連単 ◎→★→全 / ３連単 ◎→全→★
出力: analysis/out/star_exacta_*.csv
"""
import glob

import pandas as pd

import trifecta_factors as T

KEY = T.KEY


def pays():
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    out = {}
    for d in raw.to_dict("records"):
        k = (d["競馬場"], d["競走年月日"], d["レース番号"])
        e = {}
        try:
            e["tan"] = (int(d["単勝組番"]), d["単勝払戻金（円）"])
            e["uma"] = ((int(d["馬単組番1"]), int(d["馬単組番2"])), d["馬単払戻金（円）"])
            e["umaren"] = (tuple(sorted((int(d["馬複組番1"]), int(d["馬複組番2"])))), d["馬複払戻金（円）"])
            e["tri"] = ((int(d["３連単組番馬番1"]), int(d["３連単組番馬番2"]), int(d["３連単組番馬番3"])),
                        d["３連単払戻金（円）"])
        except (ValueError, TypeError):
            continue
        out[k] = e
    return out


def main():
    rows = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "pred2_underrated_rows_*.csv"))])
    P = pays()
    rec = []
    for k, g in rows.groupby(KEY):
        if k not in P or g["人気"].isna().all():
            continue
        g = g.sort_values("勝率", ascending=False)
        h = g["馬番"].astype(int).tolist()
        n = len(h)
        hon, sub = h[0], h[1:3]
        p = P[k]
        win, ex, tri, ren = p["tan"], p["uma"], p["tri"], p["umaren"]
        for basis in ("想定人気", "人気"):
            star = g[(~g["消し"]) & (g["勝率順"] <= 5) & (g[basis] >= g["勝率順"] + 2) & (g["馬番"] != hon)]["馬番"].astype(int).tolist()
            r = {"競馬場": k[0], "日付": k[1], "R": k[2], "判定": basis, "◎勝率": g["勝率"].iloc[0], "頭数": n,
                 "★数": len(star), "◎の着順": g["着順"].iloc[0],
                 "◎が1番人気": g["人気"].iloc[0] == 1, "★が2着": ex[0][1] in star, "★が3着内": any(x in star for x in tri[0][1:])}
            bets = {
                "単勝◎": [("tan", hon)],
                "馬単 ◎→○": [("uma", (hon, sub[0]))],
                "馬単 ◎→★": [("uma", (hon, s)) for s in star],
                "馬連 ◎-○": [("ren", tuple(sorted((hon, sub[0]))))],
                "馬連 ◎-★": [("ren", tuple(sorted((hon, x)))) for x in star],
                "単勝◎＋馬連 ◎-○": [("tan", hon), ("ren", tuple(sorted((hon, sub[0]))))],
                "単勝◎＋馬連 ◎-★": [("tan", hon)] + [("ren", tuple(sorted((hon, x)))) for x in star],
                "３連単 ◎→○▲→○▲": [("tri", (hon, a, b)) for a in sub for b in sub if a != b],
                "３連単 ◎→○▲★→○▲★": [("tri", (hon, a, b)) for a in sub + star for b in sub + star if a != b],
                "３連単 ◎→★→全": [("tri", (hon, s, x)) for s in star for x in h if x not in (hon, s)],
                "３連単 ◎→全→★": [("tri", (hon, x, s)) for s in star for x in h if x not in (hon, s)],
            }
            for name, bl in bets.items():
                bl = list(dict.fromkeys(bl))
                ret = 0
                for typ, c in bl:
                    if typ == "tan" and c == win[0]:
                        ret += win[1]
                    if typ == "uma" and c == ex[0]:
                        ret += ex[1]
                    if typ == "ren" and c == ren[0]:
                        ret += ren[1]
                    if typ == "tri" and c == tri[0]:
                        ret += tri[1]
                r[name + "|点数"] = len(bl)
                r[name + "|払戻"] = ret
            rec.append(r)
    d = pd.DataFrame(rec)
    d.to_csv(T.OUT / "star_exacta_rows.csv", index=False, encoding="utf-8-sig")

    names = [c[:-3] for c in d.columns if c.endswith("|点数")]

    def summ(x):
        o = {"レース数": len(x), "◎勝率%": (x["◎の着順"] == 1).mean() * 100,
             "★ありのレース%": (x["★数"] > 0).mean() * 100, "★が2着%": x["★が2着"].mean() * 100}
        for b in names:
            used = x[x[b + "|点数"] > 0]
            cost = used[b + "|点数"].sum() * 100
            o[b + " 対象R"] = len(used)
            o[b + " 的中%"] = (used[b + "|払戻"] > 0).mean() * 100 if len(used) else float("nan")
            o[b + " 回収率%"] = used[b + "|払戻"].sum() / cost * 100 if cost else float("nan")
        return pd.Series(o)

    d["◎勝率帯"] = pd.cut(d["◎勝率"], [0, 0.25, 0.4, 0.5, 1.01], labels=["25%未満", "25-40%", "40-50%", "50%以上"])
    t = d.groupby(["判定", "◎勝率帯"], observed=True).apply(summ)
    t.to_csv(T.OUT / "star_exacta_summary.csv", encoding="utf-8-sig")
    strong = d[d["◎勝率"] >= 0.4]
    c = strong.groupby(["判定", "競馬場"]).apply(summ)
    c.to_csv(T.OUT / "star_exacta_by_course.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(t.round(1).T.to_string())
    print(c[[col for col in c.columns if "★" in col or col == "レース数"]].round(0).to_string())
    s2 = d[d["◎勝率"] >= 0.4].groupby(["判定", "◎勝率帯", "◎が1番人気"], observed=True).apply(summ)
    s2.to_csv(T.OUT / "star_exacta_by_fav.csv", encoding="utf-8-sig")
    cols = ["レース数", "◎勝率%"] + [c for c in s2.columns if any(k in c for k in ["単勝◎ 回収", "馬連", "単勝◎＋"])]
    print(s2[cols].round(1).to_string())
    # 大当たりを除いた頑健性
    for basis in ("想定人気", "人気"):
        x = d[(d["判定"] == basis) & (d["◎勝率"] >= 0.4)]
        for b in ["馬単 ◎→★", "３連単 ◎→★→全", "馬連 ◎-★", "単勝◎＋馬連 ◎-★", "単勝◎＋馬連 ◎-○"]:
            u = x[x[b + "|点数"] > 0]
            pay = u[b + "|払戻"]
            cost = u[b + "|点数"].sum() * 100
            print(basis, b, "上位3本を除く回収率%", round((pay.sum() - pay.nlargest(3).sum()) / cost * 100, 1),
                  "最大払戻", pay.max())


if __name__ == "__main__":
    main()
