"""改良版の予想モデル: これまでの分析結果をすべて取り込んだ事前情報モデル (人気・オッズは使わない).

使い方: python3 analysis/predict_v2.py <data> <horselist.csv> <racelist.csv> <競馬場>

特徴量 = predict_race.py の項目
       + 能力値 (近5走の平均走破タイム・平均上がり3F・位置取り・先行率)            ... 第13節
       + 前走の着差, 昇級・降級 (賞金比), 調教師の成績, 通算勝率                    ... 第14節
       + クラス変化 (前走から / 4走前から), 近4走平均着順, 割引・加点の組み合わせ   ... 第16節
対象場の 2024年1月〜検証開始前で学習し, 直近3か月で predict_race.py の旧モデルと比較 → 全期間で再学習して予想.
出力: analysis/out/pred2_<日付>_<競馬場>.csv
"""
import sys

import numpy as np
import pandas as pd

import ability_index as A
import class_change as C
import favorite_extra as X
import predict_race as P
import trifecta_factors as T

KEY = T.KEY


def class_features(h):
    """クラス (級・段・組) の変化. h は馬ID・日付順に並べ替える"""
    h = h.sort_values(["馬ID", "日付", "レース番号"]).copy()
    h["所属地"] = h["競馬場"].map(T.HOME).fillna(h["競馬場"])
    h["クラス"] = [C.parse_class(n, c) for n, c in zip(h["レース名"], h["競馬場"])]
    g = h.groupby("馬ID")
    for i in range(1, 5):
        h[f"{i}走前クラス"] = g["クラス"].shift(i)
        h[f"{i}走前着順"] = g["着順"].shift(i)
        h[f"{i}走前所属地"] = g["所属地"].shift(i)
    same = h["1走前所属地"] == h["所属地"]
    h["前走からの変化"] = [C.change_label(p, c) if s else "比較できない"
                       for p, c, s in zip(h["1走前クラス"], h["クラス"], same)]
    tiers = []
    for rec in h[["クラス", "所属地"] + [f"{i}走前クラス" for i in (2, 3, 4)] + [f"{i}走前所属地" for i in (2, 3, 4)]].itertuples(index=False):
        cur, home = rec[0], rec[1]
        val = np.nan
        if isinstance(cur, tuple):
            for p, ph in zip(rec[2:5][::-1], rec[5:8][::-1]):  # 4走前 → 2走前の順
                if isinstance(p, tuple) and ph == home:
                    val = (p[0] * 10 + p[1]) - (cur[0] * 10 + cur[1])
                    if val == 0 and p[2] is not None and cur[2] is not None:
                        val = (p[2] - cur[2]) / 100  # 同じ段なら組の差を小さな値で
                    break
        tiers.append(val)
    h["4走前からの段の変化"] = tiers
    h["近4走平均着順"] = h[[f"{i}走前着順" for i in range(1, 5)]].mean(axis=1)
    return h


def build(d, jr_course, jr_all):
    f = P.build_features(d, jr_course)
    a = A.features(d, jr_all)
    for c in ["平均走破タイム", "平均上がり3F", "先行率", "位置取り"]:
        f[c] = a[c].values
    f["前走勝ち馬との差(秒)"] = d["前走勝ち馬との差"].fillna(2.0).values
    f["昇級(賞金比,対数)"] = np.log(d["1着賞金(円)"] / d["前走賞金"]).fillna(0).clip(-1.5, 1.5).values
    f["調教師3着内率"] = d["調教師率"].values
    n, _ = T.rec_n(d["全成績"])
    w = d["全成績"].astype(str).str.extract(r"^(\d+)-")[0].astype(float)
    f["通算勝率"] = ((w.fillna(0) + 0.5) / (n.fillna(0) + 5)).values
    f["近4走平均着順"] = d["近4走平均着順"].fillna(6).values
    ch = d["前走からの変化"]
    for k in ["昇級(段・級が上がる)", "組上がり", "組下がり", "降級(段・級が下がる)", "同じ組"]:
        f["前走から:" + k] = (ch == k).astype(float).values
    up = ch.isin(["組上がり", "昇級(段・級が上がる)"])
    f["前走4着以下で組・段が上がった"] = (up & (d["前走着順"] >= 4)).astype(float).values
    f["前走1着で組・段が上がった"] = (up & (d["前走着順"] == 1)).astype(float).values
    t = d["4走前からの段の変化"]
    f["4走前から上がった"] = (t > 0).astype(float).values
    f["4走前から下がった"] = (t < 0).astype(float).values
    f["好成績で組・段が下がった"] = ((t < 0) & (d["近4走平均着順"] <= 2.5)).astype(float).values
    return f


def main():
    data, hfile, rfile, course = sys.argv[1:5]
    h, _, _ = T.load()
    e, r = P.load_entries(hfile, rfile, course)
    h = pd.concat([h, e], ignore_index=True)
    h["予想対象"] = h["予想対象"].astype("boolean").fillna(False).astype(bool)
    races = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][1]], ignore_index=True).drop_duplicates(KEY)
    names = pd.concat([races[KEY + ["レース名"]], r[KEY + ["レース名"]]]).drop_duplicates(KEY)
    h = h.drop(columns=["レース名"], errors="ignore").merge(names, on=KEY, how="left")
    h = T.features(h)
    h = A.add_runs(h)
    h = X.extra(h)
    prize = r.set_index(KEY)["1着賞金(円)"]
    m = h["予想対象"]
    h.loc[m, "1着賞金(円)"] = prize.reindex(pd.MultiIndex.from_frame(h.loc[m, KEY])).values
    h = class_features(h)
    jr_c = P.jockey_rate(h, course)
    jr_a = A.jockey_rates(h)
    d = h[(h["競馬場"] == course) & (h["年"] >= 2024)].copy()
    d.index = range(len(d))
    f_old = P.build_features(d, jr_c)
    f_new = build(d, jr_c, jr_a)

    hist = d[~d["予想対象"]]
    cut = hist["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = hist[hist["日付"] <= cut], hist[hist["日付"] > cut]
    rows = []
    for label, f in [("旧モデル (predict_race)", f_old), ("改良版", f_new)]:
        Xt, Mt, Ot, _ = P.pad(tr, f.loc[tr.index])
        b, mu, sd = P.fit(Xt, Mt, Ot)
        Xv, Mv, Ov, _ = P.pad(va, f.loc[va.index])
        Pw = P.win_probs(Xv, Mv, b, mu, sd)
        ok = Ov[:, 0] >= 0
        top = Pw.argmax(1)
        rows.append({"モデル": label, "検証レース数": int(ok.sum()),
                     "本命の勝率%": (top == Ov[:, 0])[ok].mean() * 100,
                     "本命の3着内率%": np.mean([t in o for t, o in zip(top[ok], Ov[ok])]) * 100,
                     "勝ち馬の対数尤度": np.log(Pw[np.arange(len(Pw)), Ov[:, 0]][ok]).mean()})
    fav = va[va["人気"] == 1]
    rows.append({"モデル": "参考: 1番人気", "本命の勝率%": (fav["着順"] == 1).mean() * 100,
                 "本命の3着内率%": (fav["着順"] <= 3).mean() * 100})
    vt = pd.DataFrame(rows).set_index("モデル")
    print(f"== 検証 {course} {va['日付'].min():%Y-%m}〜{va['日付'].max():%Y-%m}\n", vt.round(3).to_string())

    Xt, Mt, Ot, _ = P.pad(hist, f_new.loc[hist.index])
    b, mu, sd = P.fit(Xt, Mt, Ot)
    tg = d[d["予想対象"]].sort_values(KEY + ["馬番"])
    Xv, Mv, _, races_ix = P.pad(tg, f_new.loc[tg.index], with_order=False)
    Pw = P.win_probs(Xv, Mv, b, mu, sd)
    out = []
    for i, ix in enumerate(races_ix):
        x = tg.loc[ix]
        w = Pw[i, :len(ix)]
        top3, tri = P.pl_probs(w)
        nos = x["馬番"].astype(int).values
        kind, buy, npts, hit = P.suggest(w, tri, nos)
        rank = (-w).argsort().argsort()
        marks = np.array(["◎", "○", "▲", "△", "△"] + [""] * 30)[rank]
        ff = f_new.loc[ix]
        for j, (_, row) in enumerate(x.iterrows()):
            flags = []
            if ff.iloc[j]["前走4着以下で組・段が上がった"]:
                flags.append("前走4着以下で組上がり(割引)")
            if ff.iloc[j]["前走1着で組・段が上がった"]:
                flags.append("前走1着で昇級(加点)")
            if ff.iloc[j]["好成績で組・段が下がった"]:
                flags.append("好成績で降級(加点)")
            if row["指数順"] >= 6 and row["前走脚質"] == "追込":
                flags.append("消し条件")
            out.append({
                "レース": int(row["レース番号"]), "レース名": row["レース名"], "距離": row["距離"], "印": marks[j],
                "馬番": int(row["馬番"]), "馬名": row["馬名"], "騎手": row["騎手名"],
                "勝率%": round(w[j] * 100, 1), "3着内率%": round(top3[j] * 100, 1),
                "前走": f"{'' if pd.isna(row['前走着順']) else int(row['前走着順'])}着"
                        f"({'' if pd.isna(row['前走人気']) else int(row['前走人気'])}人気)",
                "クラス変化": row["前走からの変化"], "近4走平均着順": None if pd.isna(row["近4走平均着順"]) else round(row["近4走平均着順"], 1),
                "指数順": None if pd.isna(row["指数順"]) else int(row["指数順"]),
                "位置取り": None if pd.isna(row["位置取り"]) else round(row["位置取り"], 2),
                "注記": " / ".join(flags),
                "買い目": f"[{kind}] {buy} ({npts}点, モデル上の的中確率 {hit*100:.0f}%)" if j == 0 else "",
            })
    o = pd.DataFrame(out)
    date = int(tg["競走年月日"].iloc[0])
    o.to_csv(T.OUT / f"pred2_{date}_{course}.csv", index=False, encoding="utf-8-sig")
    vt.to_csv(T.OUT / f"pred2_{date}_{course}_valid.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for k, g in o.groupby("レース"):
        print(f"\n## {k}R {g['レース名'].iloc[0]} {g['距離'].iloc[0]}m")
        print(g.sort_values("勝率%", ascending=False).drop(columns=["レース", "レース名", "距離", "買い目"]).to_string(index=False))
        print("買い目:", g["買い目"].iloc[0])


if __name__ == "__main__":
    main()
