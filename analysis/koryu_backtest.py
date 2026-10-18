"""交流戦 (他地区の馬が2頭以上出るレース) でのモデルの印の成績.

使い方: python3 analysis/koryu_backtest.py <data> <競馬場>   (各場の予想行を出力)
        python3 analysis/koryu_backtest.py <data> --summary  (全場をまとめて集計)
2025年1月〜2026年10月を半年ごとに区切り, その期間より前のデータだけで学習した改良版モデル (predict_v2.build) で
各レースの勝率と印 (◎○ は勝率順, ▲△ は 2〜3着補正の順) を出す. 交流戦と通常戦の成績を同じモデル・同じ期間で比べる.
交流戦の判定: 調教師所属が開催地の地区 (南関東4場は1地区) と違う馬が2頭以上.
出力: analysis/out/koryu_rows_<競馬場>.csv, koryu_summary.csv
"""
import sys

import numpy as np
import pandas as pd

import ability_index as A
import favorite_extra as X
import place_backtest as PB
import predict_race as P
import predict_v2 as V
import trifecta_factors as T

KEY = T.KEY
HOST = {"佐賀": "佐賀", "名古屋": "愛知", "園田": "兵庫", "姫路": "兵庫", "大井": "南関", "川崎": "南関", "船橋": "南関",
        "浦和": "南関", "水沢": "岩手", "盛岡": "岩手", "笠松": "笠松", "金沢": "金沢", "門別": "北海道", "高知": "高知"}
NANKAN = {"大井", "川崎", "船橋", "浦和"}
PERIODS = ["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01", "2027-01-01"]


def area(s):
    return s.map(lambda x: "南関" if x in NANKAN else x)


def run(course):
    h, _, _ = T.load()
    h["予想対象"] = False
    races = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][1]], ignore_index=True).drop_duplicates(KEY)
    h = h.drop(columns=["レース名"], errors="ignore").merge(races[KEY + ["レース名"]], on=KEY, how="left")
    h = T.features(h)
    h = A.add_runs(h)
    h = X.extra(h)
    h = V.class_features(h)
    jr_c = P.jockey_rate(h, course)
    jr_a = A.jockey_rates(h)
    q_idx = h.loc[h["年"] >= 2024, "指数"].quantile([1 / 3, 2 / 3]).values
    d = h[(h["競馬場"] == course) & (h["年"] >= 2024) & h["着順"].notna()].copy()
    d.index = range(len(d))
    d["他地区"] = area(d["調教師所属"]) != HOST[course]
    d["交流"] = d.groupby(KEY)["他地区"].transform("sum") >= 2
    f = V.build(d, jr_c, jr_a)
    out = []
    for a, b in zip(PERIODS[:-1], PERIODS[1:]):
        tr = d[d["日付"] < a]
        te = d[(d["日付"] >= a) & (d["日付"] < b)]
        if te.empty:
            continue
        Xt, Mt, Ot, _ = P.pad(tr, f.loc[tr.index])
        bw, mw, sw = P.fit(Xt, Mt, Ot)
        Xs, Ms, _, rix = P.pad(te, f.loc[te.index], with_order=False)
        pw = P.win_probs(Xs, Ms, bw, mw, sw)
        for i, ix in enumerate(rix):
            x = te.loc[ix]
            w = pw[i, :len(ix)]
            o = list((-w).argsort())
            adj = w * PB.mult(x.reset_index(drop=True), q_idx).values
            order = o[:2] + sorted(o[2:], key=lambda j: -adj[j])
            rank = np.empty(len(w), dtype=int)
            rank[order] = np.arange(len(w))
            r = x[KEY + ["馬番", "人気", "着順", "他地区", "交流", "調教師所属"]].copy()
            r["勝率"] = w
            r["印順"] = rank + 1
            r["消し"] = T.kesi(x).values
            out.append(r)
    pd.concat(out).to_csv(T.OUT / f"koryu_rows_{course}.csv", index=False, encoding="utf-8-sig")


def summary():
    import glob
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(str(T.OUT / "koryu_rows_*.csv"))])
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    pay = pay.set_index(KEY)
    res = []
    for k, g in rows.groupby(KEY):
        if k not in pay.index or g["着順"].isna().all():
            continue
        p = pay.loc[k]
        g = g.sort_values("印順")
        nos = g["馬番"].astype(int).tolist()
        w0 = g["勝率"].iloc[0]
        fin = dict(zip(g["馬番"].astype(int), g["着順"]))
        top3 = sorted([n for n in nos if fin[n] <= 3], key=lambda n: fin[n])
        win = float(p["単勝払戻金（円）"]) if int(p["単勝組番"]) == nos[0] else 0.0
        t3 = (int(p["３連単組番馬番1"]), int(p["３連単組番馬番2"]), int(p["３連単組番馬番3"]))
        f3 = set(t3)
        tp, fp = float(p["３連単払戻金（円）"]), float(p["３連複払戻金（円）"])
        # ルールの買い目 (predict_race.suggest と同じ)
        if w0 >= 0.40:
            rc, rr = 100, win
        elif w0 >= 0.25:
            pts = [(a, b, c) for a in nos[:2] for b in nos[:4] for c in nos[:5] if len({a, b, c}) == 3]
            rc, rr = 100 * len(pts), tp * (t3 in pts)
        else:
            rc, rr = 1000, fp * f3.issubset(nos[:5])
        fav = g[g["人気"] == 1]
        res.append({"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "交流": bool(g["交流"].iloc[0]),
                    "◎他地区": bool(g["他地区"].iloc[0]), "◎勝": fin[nos[0]] == 1, "◎3着内": fin[nos[0]] <= 3,
                    "◎人気": g["人気"].iloc[0], "1人気勝": bool((fav["着順"] == 1).any()),
                    "1人気3着内": bool((fav["着順"] <= 3).any()), "単勝◎": win,
                    "印5頭の3着内数": sum(fin[n] <= 3 for n in nos[:5]),
                    "3連複◎○▲": fp * f3.issubset(nos[:3]), "3連複5頭BOX": fp * f3.issubset(nos[:5]),
                    "ルール_c": rc, "ルール_r": rr,
                    "消しの3着内": int(((g["消し"]) & (g["着順"] <= 3)).sum()), "消し頭数": int(g["消し"].sum())})
    x = pd.DataFrame(res)
    x.to_csv(T.OUT / "koryu_races.csv", index=False, encoding="utf-8-sig")

    def agg(y):
        n = len(y)
        return pd.Series({"レース数": n, "◎勝率%": y["◎勝"].mean() * 100, "◎3着内率%": y["◎3着内"].mean() * 100,
                          "1番人気勝率%": y["1人気勝"].mean() * 100, "1番人気3着内率%": y["1人気3着内"].mean() * 100,
                          "◎の平均人気": y["◎人気"].mean(), "単勝◎回収率%": y["単勝◎"].sum() / n,
                          "3連複◎○▲的中%": (y["3連複◎○▲"] > 0).mean() * 100, "3連複◎○▲回収率%": y["3連複◎○▲"].sum() / n,
                          "3連複5頭BOX回収率%": y["3連複5頭BOX"].sum() / (10 * n),
                          "ルール買い目回収率%": y["ルール_r"].sum() / y["ルール_c"].sum() * 100,
                          "消しの3着内/頭数": f"{y['消しの3着内'].sum()}/{y['消し頭数'].sum()}"})
    s = x.groupby("交流").apply(agg).rename(index={True: "交流戦", False: "通常戦"})
    k = x[x["交流"]]
    s2 = k.groupby("◎他地区").apply(agg).rename(index={True: "交流戦: ◎が他地区の馬", False: "交流戦: ◎が地元の馬"})
    out = pd.concat([s, s2])
    out.to_csv(T.OUT / "koryu_summary.csv", encoding="utf-8-sig")
    print(out.round(1).T.to_string())
    print("\n交流戦の競馬場別\n", k.groupby("競馬場").apply(agg)[["レース数", "◎勝率%", "1番人気勝率%", "単勝◎回収率%", "3連複◎○▲回収率%", "ルール買い目回収率%"]].round(0).to_string())


if __name__ == "__main__":
    summary() if sys.argv[2] == "--summary" else run(sys.argv[2])
