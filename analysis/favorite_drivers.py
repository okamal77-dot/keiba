"""1番人気（市場の評価）を決めている要因の分析（2024〜2026年9月・ばんえい除く14場）.

 (a) 人気の順番 (1〜3番人気) を事前情報で説明する順位ロジット → 市場が何を重く見ているか
 (b) 着順 (1〜3着) を「人気 + 事前情報」で説明する順位ロジット → 人気を差し引いた後の係数が
     マイナスなら市場が過大評価, プラスなら過小評価している要因
 (c) 事前情報だけで1番人気をどこまで当てられるか (当たらない分 = データの外にある情報)
 (d) 1番人気の特徴別の勝率・3着内率

出力: analysis/out/fav_*.csv
"""
import numpy as np
import pandas as pd

import ability_index as A
import predict_race as P
import trifecta_factors as T

KEY = T.KEY


def build(d, jr):
    f = A.features(d, jr)[["平均走破タイム", "平均上がり3F", "位置取り", "コンビ3着内率", "騎手3着内率", "距離適性", "近走なし"]].copy()
    f["近3走ベスト指数"] = d["近3走ベスト指数"].clip(-3, 3).fillna(d["近3走ベスト指数"].mean())
    f["前走着順(対数)"] = np.log(d["前走着順"].fillna(6))
    f["前走人気(対数)"] = np.log(d["前走人気"].fillna(6))
    f["前走逃げ"] = (d["前走脚質"] == "逃げ").astype(float)
    f["前走追込"] = (d["前走脚質"] == "追込").astype(float)
    f["体重+10kg以上"] = (d["馬体重増減"] >= 10).astype(float)
    f["体重-10kg以下"] = (d["馬体重増減"] <= -10).astype(float)
    f["転入初戦"] = (d["前走調教師所属"].notna() & (d["前走調教師所属"] != d["調教師所属"])).astype(float)
    f["遠征"] = (d["調教師所属"] != d["所属地"]).astype(float)
    f["間隔(対数)"] = np.log1p(d["間隔日"].fillna(60).clip(0, 365))
    f["年齢"] = d["齢"].clip(2, 9)
    f["牝馬"] = (d["性"] == "牝").astype(float)
    f["枠番"] = d["枠番"]
    return f


def order_by(d, col):
    """col の小さい順に 1〜3 を着順として扱った DataFrame"""
    x = d.copy()
    x["着順"] = x.groupby(KEY)[col].rank(method="first")
    return x


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待"] = h.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)["top3"].transform("mean")
    jr = A.jockey_rates(h)
    d = h[(h["年"] >= 2024) & h["人気"].notna()].copy()
    f = build(d, jr)
    cols = list(f.columns)

    # (a) 人気の順番
    X, M, O, _ = P.pad(order_by(d, "人気"), f)
    b_pop, mu, sd = P.fit(X, M, O)
    # 結果 (人気なし)
    X, M, O, _ = P.pad(d, f)
    b_res, _, _ = P.fit(X, M, O)
    # (b) 結果 (人気あり)
    f2 = f.copy()
    f2["人気(対数)"] = np.log(d["人気"])
    X, M, O, _ = P.pad(d, f2)
    b_adj, _, _ = P.fit(X, M, O)

    # 係数はいずれも「その値が大きい馬ほど上位 (人気上位 / 着順上位) になりやすい」ときに +
    t = pd.DataFrame({"人気への影響": b_pop, "結果への影響(人気なし)": b_res,
                      "人気を差し引いた後の影響": b_adj[:-1]}, index=cols)
    t["値が大きい馬は"] = np.select([t["人気を差し引いた後の影響"] <= -0.05, t["人気を差し引いた後の影響"] >= 0.05],
                              ["人気ほど走らない", "人気以上に走る"], "ほぼ人気どおり")
    t = t.sort_values("人気への影響", key=abs, ascending=False)
    t.to_csv(T.OUT / "fav_drivers.csv", encoding="utf-8-sig")
    print(t.round(3).to_string())
    print("人気(対数) の係数", round(b_adj[-1], 3))

    # (c) 事前情報だけで1番人気を当てられるか (直近3か月で検証)
    cut = d["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = d[d["日付"] <= cut], d[d["日付"] > cut]
    X, M, O, _ = P.pad(order_by(tr, "人気"), f.loc[tr.index])
    b, mu, sd = P.fit(X, M, O)
    X, M, O, races = P.pad(order_by(va, "人気"), f.loc[va.index])
    Pw = P.win_probs(X, M, b, mu, sd)
    top = Pw.argmax(1)
    hit1 = (top == O[:, 0]).mean()
    hit3 = np.mean([t_ in o for t_, o in zip(top, O)])
    rank_of_fav = np.array([(-Pw[i]).argsort().tolist().index(O[i, 0]) + 1 for i in range(len(Pw))])
    c = pd.Series({"レース数": len(Pw), "モデル予想の1番人気が実際の1番人気": hit1 * 100,
                   "モデル予想の1番人気が実際の3番人気以内": hit3 * 100,
                   "実際の1番人気がモデル予想で2位以内": (rank_of_fav <= 2).mean() * 100})
    c.to_csv(T.OUT / "fav_predictability.csv", encoding="utf-8-sig")
    print(c.round(1).to_string())

    # (d) 1番人気の特徴別成績
    fav = d[d["人気"] == 1].copy()
    fav["指数順"] = fav["指数順"]
    fav["勝"] = (fav["着順"] == 1)
    seg = {
        "前走1着": fav["前走着順"] == 1, "前走2-3着": fav["前走着順"].between(2, 3), "前走4着以下": fav["前走着順"] >= 4,
        "前走なし": fav["前走着順"].isna(),
        "近3走指数 レース内1位": fav["指数順"] == 1, "近3走指数 2-3位": fav["指数順"].between(2, 3),
        "近3走指数 4位以下": fav["指数順"] >= 4,
        "騎手3着内率 上位25%": f.loc[fav.index, "騎手3着内率"] >= f["騎手3着内率"].quantile(0.75),
        "騎手3着内率 下位50%": f.loc[fav.index, "騎手3着内率"] <= f["騎手3着内率"].quantile(0.5),
        "前走逃げ": fav["前走脚質"] == "逃げ", "前走追込": fav["前走脚質"] == "追込",
        "体重+10kg以上": fav["馬体重増減"] >= 10, "転入初戦": f.loc[fav.index, "転入初戦"] == 1,
        "遠征": f.loc[fav.index, "遠征"] == 1,
    }
    rows = []
    for k, m in seg.items():
        x = fav[m]
        rows.append((k, len(x), len(x) / len(fav) * 100, x["勝"].mean() * 100, x["top3"].mean() * 100))
    t = pd.DataFrame(rows, columns=["1番人気の特徴", "頭数", "1番人気に占める割合%", "勝率%", "3着内率%"]).set_index("1番人気の特徴")
    t.loc["1番人気 全体"] = [len(fav), 100, fav["勝"].mean() * 100, fav["top3"].mean() * 100]
    t.to_csv(T.OUT / "fav_segments.csv", encoding="utf-8-sig")
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()
