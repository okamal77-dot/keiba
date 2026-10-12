"""前に行く馬の目印 と 中央からの転入馬 の検証.

 1) 前に行く馬: 直近の4角位置 (前走 / 2走前) が3番手以内か
      「前走3番手以内」「直近2走のどちらか」「直近2走とも」
 2) 中央からの転入馬: 通算出走数 (全成績) に比べて, データ内 (地方) の出走が少ない馬
      地方での出走が0〜2走 かつ 通算出走が地方の出走より3走以上多い
指標 (2024〜2026年10月, ばんえい除く):
  市場に対する A/E = 実際の勝ち(3着内) ÷ 同じ年・場・人気・頭数帯の平均  … 1より大きければ人気より走る
  モデルに対する A/E = 実際の勝ち ÷ 改良版モデルの勝率の合計 (直近3か月, pred2_underrated_rows_*.csv の6場)
出力: analysis/out/ft_*.csv
"""
import glob

import numpy as np
import pandas as pd

import ability_index as A
import trifecta_factors as T

KEY = T.KEY


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    h = h.sort_values(["馬ID", "日付", "レース番号"])
    g = h.groupby("馬ID")
    p1, p2 = g["4角位置"].shift(1), g["4角位置"].shift(2)
    h["前走3番手内"] = p1 <= 3
    h["直近2走どちらか3番手内"] = (p1 <= 3) | (p2 <= 3)
    h["直近2走とも3番手内"] = (p1 <= 3) & (p2 <= 3)
    h["地方出走数"] = g.cumcount()
    n, _ = T.rec_n(h["全成績"])
    h["通算出走"] = n.values
    h["中央からの転入"] = (h["地方出走数"] <= 2) & (h["通算出走"] >= h["地方出走数"] + 3) & (h["日付"] >= "2024-01-01")
    h["勝"] = (h["着順"] == 1).astype(float)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    d = h[(h["年"] >= 2024) & h["人気"].notna() & h["着順"].notna()].copy()
    grp = d.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)
    d["期待勝"] = grp["勝"].transform("mean")
    d["期待"] = grp["top3"].transform("mean")

    def ae(x):
        return pd.Series({"頭数": len(x), "勝率%": x["勝"].mean() * 100, "平均人気": x["人気"].mean(),
                          "勝ちA/E(市場)": x["勝"].sum() / x["期待勝"].sum(),
                          "3着内A/E(市場)": x["top3"].sum() / x["期待"].sum()})

    flags = ["前走3番手内", "直近2走どちらか3番手内", "直近2走とも3番手内", "中央からの転入"]
    rows = []
    for f in flags:
        for v in (True, False):
            x = d[d[f] == v]
            r = ae(x)
            r["目印"], r["該当"] = f, v
            for y in (2024, 2025, 2026):
                r[f"{y}年 勝ちA/E"] = ae(x[x["年"] == y])["勝ちA/E(市場)"]
            rows.append(r)
    t1 = pd.DataFrame(rows).set_index(["目印", "該当"])
    t1.to_csv(T.OUT / "ft_market.csv", encoding="utf-8-sig")
    # 人気別 (人気薄ほど効くか)
    d["人気帯"] = pd.cut(d["人気"], [0, 1, 3, 6, 99], labels=["1番人気", "2-3番人気", "4-6番人気", "7番人気以下"])
    t2 = d.groupby(["人気帯", "直近2走どちらか3番手内"], observed=True).apply(ae)
    t2.to_csv(T.OUT / "ft_market_by_pop.csv", encoding="utf-8-sig")
    t3 = d[d["中央からの転入"]].groupby("人気帯", observed=True).apply(ae)
    t3.to_csv(T.OUT / "ft_transfer_by_pop.csv", encoding="utf-8-sig")

    # モデルに対する A/E (直近3か月)
    m = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "pred2_underrated_rows_*.csv"))])
    m = m.merge(d[KEY + ["馬番", "勝"] + flags], on=KEY + ["馬番"], how="inner")
    out = []
    for f in flags:
        for v in (True, False):
            x = m[m[f] == v]
            out.append({"目印": f, "該当": v, "頭数": len(x), "実際の勝率%": x["勝"].mean() * 100,
                        "モデル勝率の平均%": x["勝率"].mean() * 100, "市場の平均%": x["期待勝"].mean() * 100,
                        "勝ちA/E(モデル)": x["勝"].sum() / x["勝率"].sum(),
                        "勝ちA/E(市場)": x["勝"].sum() / x["期待勝"].sum()})
    t4 = pd.DataFrame(out).set_index(["目印", "該当"])
    t4.to_csv(T.OUT / "ft_model.csv", encoding="utf-8-sig")
    # 転入馬: モデルと市場を混ぜたら良くなるか (対数尤度)
    pd.set_option("display.width", 250)
    for name, t in [("市場に対して", t1), ("人気帯 × 直近2走どちらか3番手内", t2), ("転入馬 × 人気帯", t3), ("モデルに対して (直近3か月)", t4)]:
        print(f"\n== {name}\n", t.round(2).to_string())


if __name__ == "__main__":
    main()
