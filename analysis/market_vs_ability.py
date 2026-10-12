"""市場人気と能力 (走破タイム指数) のずれ.

 指数 = 競馬場×距離×馬場状態ごとに標準化した走破タイム (大きいほど速い, trifecta_factors.features)
 1) 市場は何に反応して人気を上げ下げするか: 前走の着順 と 前走の指数 (タイム) で 今回の人気がどう動くか
 2) 前々走→前走で指数が上がった/下がった馬の 人気の動きと 実際の成績
 3) 前走人気→今回人気の変化と成績 (人気が急に上がった馬は買われすぎか)
 4) 今回のメンバー内での 能力順位 (近3走ベスト指数) と 人気順位の差
指標: 勝ちA/E・2〜3着A/E (1着を除く 2着か3着に入る率)・3着内A/E (同じ年・場・人気・頭数帯の平均に対する比. 1より大きい = 人気より走る), 単勝回収率
対象: 2024〜2026年10月, ばんえい除く. 出力: analysis/out/mva_*.csv
"""
import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY


def main():
    h, _, _ = T.load()
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"].sort_values(["馬ID", "日付", "レース番号"])
    g = h.groupby("馬ID")
    h["前々走指数"] = g["指数"].shift(2)
    h["指数変化"] = h["前走指数"] - h["前々走指数"]
    h["人気変化"] = h["人気"] - h["前走人気"]  # マイナス = 人気が上がった
    h["能力順"] = h["指数順"]
    h["人気と能力順の差"] = h["人気"] - h["能力順"]  # プラス = 能力順位より人気が低い
    h["勝"] = (h["着順"] == 1).astype(float)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    d = h[(h["年"] >= 2024) & h["人気"].notna() & h["着順"].notna()].copy()
    gr = d.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)
    d["期待勝"] = gr["勝"].transform("mean")
    d["期待"] = gr["top3"].transform("mean")
    d["2-3着"] = d["着順"].isin([2, 3]).astype(float)
    d["期待23"] = gr["2-3着"].transform("mean")
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True)
    tan = raw[KEY + ["単勝組番", "単勝払戻金（円）"]].dropna()
    tan = tan.assign(馬番=tan["単勝組番"].astype(int)).groupby(KEY + ["馬番"])["単勝払戻金（円）"].sum()
    d = d.merge(tan.rename("単勝払戻").reset_index(), on=KEY + ["馬番"], how="left")
    d["単勝払戻"] = d["単勝払戻"].fillna(0)
    fk = pd.concat([raw[KEY + [f"複勝組番{i}", f"複勝払戻金{i}（円）"]].dropna().set_axis(KEY + ["馬番", "複勝払戻"], axis=1)
                    for i in (1, 2, 3)])
    fk["馬番"] = fk["馬番"].astype(int)
    fk = fk.groupby(KEY + ["馬番"])["複勝払戻"].sum()
    d = d.merge(fk.reset_index(), on=KEY + ["馬番"], how="left")
    d["複勝払戻"] = d["複勝払戻"].fillna(0)

    def s(x):
        return pd.Series({"頭数": len(x), "今回の平均人気": x["人気"].mean(), "前走の平均人気": x["前走人気"].mean(),
                          "人気の変化(平均)": x["人気変化"].mean(), "勝率%": x["勝"].mean() * 100,
                          "勝ちA/E": x["勝"].sum() / x["期待勝"].sum(), "3着内A/E": x["top3"].sum() / x["期待"].sum(),
                          "単勝回収率%": x["単勝払戻"].mean(), "2-3着率%": x["2-3着"].mean() * 100,
                          "2-3着A/E": x["2-3着"].sum() / x["期待23"].sum(), "複勝回収率%": x["複勝払戻"].mean()})

    def by_year(x, cols, a="勝", e="期待勝"):
        return x.groupby(cols + ["年"], observed=True).apply(lambda z: z[a].sum() / z[e].sum()).unstack()

    d["前走着順帯"] = pd.cut(d["前走着順"], [0, 1, 3, 5, 99], labels=["1着", "2-3着", "4-5着", "6着以下"])
    q = d["前走指数"].quantile([1 / 3, 2 / 3]).values
    d["前走タイム"] = pd.cut(d["前走指数"], [-99, q[0], q[1], 99], labels=["遅い(下位1/3)", "中位", "速い(上位1/3)"])
    d["指数変化帯"] = pd.cut(d["指数変化"], [-99, -1, -0.3, 0.3, 1, 99],
                         labels=["大きく下がった(-1未満)", "下がった", "ほぼ同じ(±0.3)", "上がった", "大きく上がった(+1超)"])
    d["人気変化帯"] = pd.cut(d["人気変化"], [-99, -4, -2, -1, 0, 1, 3, 99],
                         labels=["4つ以上上がった", "2-3上がった", "1上がった", "同じ", "1下がった", "2-3下がった", "4つ以上下がった"])
    d["人気と能力の差帯"] = pd.cut(d["人気と能力順の差"], [-99, -4, -2, 1, 3, 99],
                           labels=["能力順より4つ以上人気(買われすぎ?)", "2-3人気", "ほぼ同じ(±1)", "2-3人気がない", "4つ以上人気がない"])

    d["前走上がり帯"] = pd.cut(d["前走上がり順"], [0, 1, 3, 99], labels=["上がり1位", "2-3位", "4位以下"])
    out = {
        "mva23_style": d.groupby("前走脚質", observed=True).apply(s),
        "mva23_agari": d.groupby("前走上がり帯", observed=True).apply(s),
        "mva23_style_x_pop": d.groupby(["前走脚質", pd.cut(d["人気"], [0, 1, 3, 6, 99], labels=["1番人気", "2-3番人気", "4-6番人気", "7番人気以下"])], observed=True).apply(s),
        "mva_finish_x_time": d.groupby(["前走着順帯", "前走タイム"], observed=True).apply(s),
        "mva_index_change": d.groupby("指数変化帯", observed=True).apply(s),
        "mva_index_change_x_finish": d.groupby(["指数変化帯", "前走着順帯"], observed=True).apply(s),
        "mva_pop_change": d.groupby("人気変化帯", observed=True).apply(s),
        "mva_pop_change_x_finish": d.groupby(["前走着順帯", "人気変化帯"], observed=True).apply(s),
        "mva_pop_vs_ability": d.groupby("人気と能力の差帯", observed=True).apply(s),
    }
    yr = {
        "mva_finish_x_time_by_year": by_year(d, ["前走着順帯", "前走タイム"]),
        "mva_index_change_by_year": by_year(d, ["指数変化帯"]),
        "mva_pop_change_by_year": by_year(d, ["人気変化帯"]),
        "mva_pop_vs_ability_by_year": by_year(d, ["人気と能力の差帯"]),
        "mva23_finish_x_time_by_year": by_year(d, ["前走着順帯", "前走タイム"], "2-3着", "期待23"),
        "mva23_pop_change_by_year": by_year(d, ["人気変化帯"], "2-3着", "期待23"),
        "mva23_pop_vs_ability_by_year": by_year(d, ["人気と能力の差帯"], "2-3着", "期待23"),
        "mva23_style_by_year": by_year(d, ["前走脚質"], "2-3着", "期待23"),
        "mva23_agari_by_year": by_year(d, ["前走上がり帯"], "2-3着", "期待23"),
    }
    # 市場が何に反応するか: 今回人気(対数) を 前走着順・前走指数・前走人気 で回帰 (同じレース内の相対値)
    x = d.dropna(subset=["前走着順", "前走指数", "前走人気"]).copy()
    for c in ["前走着順", "前走指数", "前走人気", "近3走ベスト指数"]:
        x[c + "(レース内標準化)"] = x.groupby(KEY)[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    cols = [c for c in x.columns if c.endswith("(レース内標準化)")]
    X = np.c_[np.ones(len(x)), x[cols].fillna(0).values]
    y = np.log(x["人気"].values)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    coef = pd.Series(beta[1:], index=cols, name="人気(対数)への影響 (+は人気を下げる)")
    out["mva_market_coef"] = coef.to_frame()
    for k, t in {**out, **yr}.items():
        t.to_csv(T.OUT / f"{k}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for k, t in {**out, **yr}.items():
        print(f"\n== {k}\n", t.round(2).to_string())


if __name__ == "__main__":
    main()
