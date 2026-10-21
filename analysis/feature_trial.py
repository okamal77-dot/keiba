"""能力モデル (predict_v2.build) に新しい項目を足して, ◎の精度が上がるかを時系列で確かめる.

使い方: python3 analysis/feature_trial.py <data> [競馬場 ...]   (省略時はばんえいを除く14場)
足す項目 (どれも当該レースより前に分かる情報だけで作る):
  ペース   : レースの前半の速さ = (勝ちタイム − レースの上がり3F) の 競馬場×距離×馬場区分 の z (速いほど +).
             ラップ (ハロンタイム) は南関東4場と2026年の園田にしかないので, 全場で取れるこの値を使う.
             - 近5走平均「前半の速さ × 前にいた度合い」(速い流れを前で受けた馬ほど +, 遅い流れで前にいて恵まれた馬は −)
             - 近5走平均「前半の速さ」
             - 今回の先行争い: 自分の先行率 × 他の馬の先行率の合計 (先行馬が多いほど前の馬に不利)
  血統     : 父・母父の「その距離帯」「重・不良」での3着内 A/E (人気から見込む3着内数との比, 縮小).
             半年ごとに, その期間より前のデータだけで表を作る. 経験の少ない馬 (近走2走以下 or 距離が200m以上変わる) との掛け算も足す.
  当日の馬場: 同じ日・同じ競馬場の, それより前のレースの結果から
             - 前有利度 (3着内の馬の4角の前目度 − 全馬の平均) × 自分の前目度 (近5走の位置取りから)
             - 内有利度 (3着内の馬の内枠度 − 全馬の平均) × 自分の内枠度
             前のレースが少ないうちは 0 に寄せる (n/(n+3) を掛ける).
検証: 2025年1月〜2026年10月を半年ごと (最後は 2026年7月〜) に区切り, その期間より前 (2024年1月〜) だけで学習して予想.
指標: ◎の勝率・3着内率, 勝ち馬に付けた確率の対数 (大きいほどよい), 単勝◎の回収率, ◎の平均人気, ◎の勝ち A/E (人気から見込む勝ち数との比).
出力: analysis/out/ft_rows.csv (レースごと), ft_summary.csv, ft_coef.csv
"""
import sys

import numpy as np
import pandas as pd

import ability_index as A
import favorite_extra as X
import predict_race as P
import predict_v2 as V
import extra_features as E
import trifecta_factors as T

KEY = T.KEY
PERIODS = E.PERIODS
COURSES = ["大井", "川崎", "船橋", "浦和", "名古屋", "笠松", "金沢", "園田", "姫路", "高知", "佐賀", "門別", "盛岡", "水沢"]
PACE = ["ペース×前目(近5走)", "前半の速さ(近5走)", "先行争い"]
BLOOD, BIAS = E.BLOOD, E.BIAS
VARIANTS = {"今のモデル": [], "+ペース": PACE, "+血統": BLOOD, "+当日の馬場": BIAS, "+3つすべて": PACE + BLOOD + BIAS,
            "+血統+当日の馬場 (採用)": BLOOD + BIAS}


def add_pace(h, r):
    """レースの前半の速さ (z) と, 1走ごとの「前半の速さ × 前目度」, その近5走平均"""
    win = h[h["着順"] == 1].groupby(KEY)["秒"].min().rename("勝ち秒")
    rr = r[KEY + ["上がり3F"]].rename(columns={"上がり3F": "レース上がり3F"})
    h = h.merge(rr, on=KEY, how="left").merge(win, on=KEY, how="left")
    h["レース上がり3F"] = pd.to_numeric(h["レース上がり3F"], errors="coerce")
    front = h["勝ち秒"] - h["レース上がり3F"]
    front = front.where((h["レース上がり3F"] > 30) & (h["レース上がり3F"] < 50))
    g = front.groupby([h["競馬場"], h["距離"], h["馬場G"]])
    z = -(front - g.transform("mean")) / g.transform("std")
    h["前半の速さ"] = z.where(g.transform("size") >= 30).clip(-3, 3)
    h["前目度"] = 1 - (h["4角位置"] - 1) / (h["頭数"] - 1).clip(lower=1)
    h["ペース×前目"] = h["前半の速さ"] * (h["前目度"] - 0.5)
    h = h.sort_values(["馬ID", "日付", "レース番号"])
    for c, name in [("ペース×前目", "ペース×前目(近5走)"), ("前半の速さ", "前半の速さ(近5走)")]:
        h[name] = h[c].groupby(h["馬ID"]).transform(lambda s: s.shift(1).rolling(A.N_PAST, min_periods=1).mean())
    return h


def extra(d):
    f = pd.DataFrame(index=d.index)
    for c in ["ペース×前目(近5走)", "前半の速さ(近5走)"]:
        f[c] = d[c].fillna(0)
    sen = d["先行率"].fillna(d["先行率"].mean())
    tot = sen.groupby([d[k] for k in KEY]).transform("sum")
    f["先行争い"] = sen * (tot - sen)
    for c in BLOOD + BIAS:
        f[c] = d[c].fillna(0)
    return f


def run(course, h, pay):
    jr_c = P.jockey_rate(h, course)
    jr_a = A.jockey_rates(h)
    d = h[(h["競馬場"] == course) & (h["年"] >= 2024) & h["着順"].notna()].copy()
    d.index = range(len(d))
    base = V.build(d, jr_c, jr_a)
    ex = extra(d)
    rows, coefs = [], []
    for a, b in zip(PERIODS[:-1], PERIODS[1:]):
        tr, te = d[d["日付"] < a], d[(d["日付"] >= a) & (d["日付"] < b)]
        if te.empty or tr.empty:
            continue
        for name, cols in VARIANTS.items():
            f = pd.concat([base, ex[cols]], axis=1)
            Xt, Mt, Ot, _ = P.pad(tr, f.loc[tr.index])
            bw, mw, sw = P.fit(Xt, Mt, Ot)
            if cols:
                coefs.append({"競馬場": course, "期間": a, "モデル": name,
                              **{c: bw[list(f.columns).index(c)] for c in cols}})
            Xs, Ms, _, rix = P.pad(te, f.loc[te.index], with_order=False)
            pw = P.win_probs(Xs, Ms, bw, mw, sw)
            for i, ix in enumerate(rix):
                x = te.loc[ix]
                w = pw[i, :len(ix)]
                j = int(w.argmax())
                k = tuple(x[KEY].iloc[0])
                fin = x["着順"].values
                win_pay = 0.0
                if k in pay.index and fin[j] == 1:
                    win_pay = float(pay.loc[k, "単勝払戻金（円）"])
                wi = np.where(fin == 1)[0]
                rows.append({"競馬場": course, "競走年月日": k[1], "レース番号": k[2], "期間": a, "モデル": name,
                             "◎勝": fin[j] == 1, "◎3着内": fin[j] <= 3, "◎人気": x["人気"].values[j],
                             "◎期待勝": x["期待勝"].values[j], "単勝◎": win_pay,
                             "勝ち馬の対数確率": np.log(w[wi[0]]) if len(wi) else np.nan})
    return rows, coefs


def agg(g):
    n = len(g)
    return pd.Series({"レース数": n, "◎勝率%": g["◎勝"].mean() * 100, "◎3着内率%": g["◎3着内"].mean() * 100,
                      "対数確率": g["勝ち馬の対数確率"].mean(), "単勝◎回収率%": g["単勝◎"].sum() / n,
                      "◎平均人気": g["◎人気"].mean(), "◎勝ちA/E": g["◎勝"].sum() / g["◎期待勝"].sum()})


def main():
    courses = sys.argv[2:] or COURSES
    h, r, p = T.load()
    h["予想対象"] = False
    races = r.drop_duplicates(KEY)
    h = h.drop(columns=["レース名"], errors="ignore").merge(races[KEY + ["レース名"]], on=KEY, how="left")
    h = T.features(h)
    h = A.add_runs(h)
    h = X.extra(h)
    h = V.class_features(h)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待勝"] = h.groupby(["年", "人気c", "頭数帯"], observed=True)["着順"].transform(lambda s: (s == 1).mean())
    h = add_pace(h, races)
    h = E.add_blood(h)
    h = E.add_bias(h)
    pay = p.drop_duplicates(KEY).set_index(KEY)
    rows, coefs = [], []
    for c in courses:
        rr, cc = run(c, h, pay)
        rows += rr
        coefs += cc
        print(c, "done", file=sys.stderr)
    x = pd.DataFrame(rows)
    x.to_csv(T.OUT / "ft_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefs).to_csv(T.OUT / "ft_coef.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    s = x.groupby("モデル", sort=False).apply(agg)
    print("== 全場 (2025年1月〜2026年10月, 半年ごとに学習し直し)\n", s.round(3).to_string())
    by = x.groupby(["期間", "モデル"], sort=False).apply(agg)[["◎勝率%", "対数確率", "単勝◎回収率%"]].unstack("モデル")
    print("\n== 期間ごと\n", by.round(3).to_string())
    bc = x.groupby(["競馬場", "モデル"], sort=False).apply(agg)["対数確率"].unstack("モデル")
    print("\n== 競馬場ごとの対数確率\n", bc.round(4).to_string())
    s.round(4).to_csv(T.OUT / "ft_summary.csv", encoding="utf-8-sig")
    # 今のモデルとの差 (レースごとの対応のある比較, ブートストラップ)
    base = x[x["モデル"] == "今のモデル"].set_index(KEY)
    rng = np.random.default_rng(0)
    print("\n== 今のモデルとの差 (90%区間)")
    for name in list(VARIANTS)[1:]:
        o = x[x["モデル"] == name].set_index(KEY).reindex(base.index)
        for col, scale in [("勝ち馬の対数確率", 1), ("◎勝", 100), ("単勝◎", 1)]:
            dv = ((o[col].astype(float) - base[col].astype(float)) * scale).dropna().to_numpy()
            bs = dv[rng.integers(0, len(dv), (1000, len(dv)))].mean(1)
            print(f"  {name} {col}: {dv.mean():+.4f} [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]")


if __name__ == "__main__":
    main()
