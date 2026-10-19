"""市場の癖 (データから見込まれる人気と実際の人気のずれ) を補正値として使えるかの検証.

 1) 2024年のデータで, 事前情報 (favorite_extra.py の項目) から人気の順番を予測するモデルを作る
 2) 2025年1月以降の各出走で 上振れ = log(予測した人気順位) − log(実際の人気順位)
    (+ ならデータから見込まれるより人気になった)
 3) 馬・調教師・騎手・馬主ごとに, そのレースの前日までの上振れを縮約平均して「癖」とする
 4) 2025年〜検証開始前で学習し, 直近3か月で
      (a) 1番人気の予測が当たるようになるか
      (b) 着順 (人気を使わない) の予測が良くなるか
      (c) 人気を差し引いた後, 癖が大きい馬は人気以上に走るか (= 癖は隠れた情報) / 人気ほど走らないか (= 単なる偏り)
    を比べる

出力: analysis/out/habit_*.csv
"""
import numpy as np
import pandas as pd

import ability_index as A
import favorite_drivers as F
import favorite_extra as X
import predict_race as P
import trifecta_factors as T

KEY = T.KEY
GROUPS = {"馬": ("馬ID", 3), "調教師": ("調教師", 20), "騎手": ("騎手名", 30), "馬主": ("馬主氏名", 20)}


def past_mean(d, col, val, k):
    """col ごとに, 当日より前の val の縮約平均 (平均0へ k 件分寄せる) と件数"""
    day = d.groupby([col, "日付"])[val].agg(["sum", "count"]).reset_index().sort_values([col, "日付"])
    g = day.groupby(col)
    day["cs"] = g["sum"].cumsum() - day["sum"]
    day["cn"] = g["count"].cumsum() - day["count"]
    m = d[[col, "日付"]].merge(day[[col, "日付", "cs", "cn"]], on=[col, "日付"], how="left")
    return (m["cs"] / (m["cn"] + k)).fillna(0).values, m["cn"].fillna(0).values


def rank_in_race(d, score):
    return pd.Series(score, index=d.index).groupby([d[k] for k in KEY]).rank(ascending=False, method="average")


def fit_eval(tr, va, ftr, fva, order_col):
    if order_col == "着順":
        Xt, Mt, Ot, _ = P.pad(tr, ftr)
        Xv, Mv, Ov, _ = P.pad(va, fva)
    else:
        Xt, Mt, Ot, _ = P.pad(F.order_by(tr, order_col), ftr)
        Xv, Mv, Ov, _ = P.pad(F.order_by(va, order_col), fva)
    b, mu, sd = P.fit(Xt, Mt, Ot)
    Pw = P.win_probs(Xv, Mv, b, mu, sd)
    ok = Ov[:, 0] >= 0
    top = Pw.argmax(1)
    hit = (top == Ov[:, 0])[ok].mean() * 100
    in3 = np.mean([t in o for t, o in zip(top[ok], Ov[ok])]) * 100
    ll = np.log(Pw[np.arange(len(Pw)), Ov[:, 0]][ok]).mean()
    return hit, in3, ll, b


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    h = X.extra(h)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待"] = h.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)["top3"].transform("mean")
    jr = A.jockey_rates(h)
    d = h[(h["年"] >= 2024) & h["人気"].notna()].copy()
    d.index = range(len(d))
    f = X.build(d, jr)

    # 1) 人気モデル (2024年)
    base = d["年"] == 2024
    Xb, Mb, Ob, _ = P.pad(F.order_by(d[base], "人気"), f[base])
    b, mu, sd = P.fit(Xb, Mb, Ob)
    score = ((f - mu) / sd).values @ b
    d["予測人気順位"] = rank_in_race(d, score)
    # 2) 上振れ
    d["上振れ"] = np.log(d["予測人気順位"]) - np.log(d["人気"])
    d = d[d["年"] >= 2025].copy()
    f = f.loc[d.index]
    # 3) 癖 (前日までの縮約平均)
    hab = pd.DataFrame(index=d.index)
    for name, (col, k) in GROUPS.items():
        hab[f"{name}の癖"], hab[f"{name}の件数"] = past_mean(d, col, "上振れ", k)
    feats_h = [f"{n}の癖" for n in GROUPS]
    print("癖の分布\n", hab[feats_h].describe().round(3).to_string())

    # 4) 検証
    cut = d["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = d[d["日付"] <= cut], d[d["日付"] > cut]
    fh = pd.concat([f, hab[feats_h]], axis=1)
    rows = []
    for label, ff in [("データの項目のみ", f), ("＋市場の癖", fh)]:
        hp, ip, lp, bp = fit_eval(tr, va, ff.loc[tr.index], ff.loc[va.index], "人気")
        hr, ir, lr, br = fit_eval(tr, va, ff.loc[tr.index], ff.loc[va.index], "着順")
        rows.append({"モデル": label,
                     "1番人気の的中%": hp, "1番人気が予想3位以内%": ip, "人気の対数尤度": lp,
                     "本命の勝率%": hr, "本命の3着内率%": ir, "勝ち馬の対数尤度": lr})
        if label == "＋市場の癖":
            coef = pd.DataFrame({"人気への影響": bp[-len(feats_h):], "結果への影響(人気なし)": br[-len(feats_h):]}, index=feats_h)
    fav = va[va["人気"] == 1]
    rows.append({"モデル": "参考: 実際の1番人気", "本命の勝率%": (fav["着順"] == 1).mean() * 100,
                 "本命の3着内率%": (fav["着順"] <= 3).mean() * 100})
    t = pd.DataFrame(rows).set_index("モデル")
    t.to_csv(T.OUT / "habit_validation.csv", encoding="utf-8-sig")
    print(f"\n== 検証 {va['日付'].min():%Y-%m}〜{va['日付'].max():%Y-%m} {va.groupby(KEY).ngroups}レース\n", t.round(3).to_string())

    # (c) 人気を差し引いた後の癖の影響 (全期間)
    f2 = fh.copy()
    f2["人気(対数)"] = np.log(d["人気"])
    Xa, Ma, Oa, _ = P.pad(d, f2)
    ba, _, _ = P.fit(Xa, Ma, Oa)
    coef["人気を差し引いた後の影響"] = ba[len(f.columns):len(f.columns) + len(feats_h)]
    coef["値が大きい(人気になりやすい)と"] = np.select(
        [coef["人気を差し引いた後の影響"] <= -0.03, coef["人気を差し引いた後の影響"] >= 0.03],
        ["人気ほど走らない(単なる偏り)", "人気以上に走る(隠れた情報)"], "ほぼ人気どおり")
    coef.to_csv(T.OUT / "habit_coef.csv", encoding="utf-8-sig")
    print("\n", coef.round(3).to_string())

    # 癖の大きさ別: 人気補正後の成績 (検証期間)
    for name in GROUPS:
        q = pd.qcut(hab.loc[va.index, f"{name}の癖"].rank(method="first"), 5, labels=["下位20%", "2", "3", "4", "上位20%"])
        g = va.groupby(q, observed=True)
        s = pd.DataFrame({"平均人気": g["人気"].mean(), "勝率%": g["着順"].apply(lambda x: (x == 1).mean() * 100),
                          "3着内率%": g["top3"].mean() * 100, "A/E(人気補正)": g["top3"].sum() / g["期待"].sum()})
        print(f"\n{name}の癖 (検証期間, 5分位)\n", s.round(2).to_string())
        s.to_csv(T.OUT / f"habit_quintile_{name}.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
