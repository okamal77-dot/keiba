"""消しの条件の見直し.

いま: 指数順(近3走ベスト指数のレース内順位) 6位以下 かつ 前走の脚質が追込
前走だけで判定すると, 大敗や距離が大きく違うレースで後ろにいただけの馬まで消してしまう (10/12 川崎12R, 10/16 川崎10R・園田10R).
比べる条件 (どれも 指数順6位以下 が前提):
  A いま                 : 前走 追込
  B 2走とも追込           : 前走 追込 かつ 前々走 追込
  C 前走の距離が近い      : 前走 追込 かつ 前走の距離が今回の±200m以内
  D B または C            : 2走とも追込, または 距離の近い前走で追込
  E B かつ C
指標: 消した頭数, 3着内率, 3着内A/E (同じ年・場・人気・頭数帯の平均3着内率に対する比. 小さいほど正しく消せている),
      1着の数, 年別の3着内A/E. 対象 2024〜2026年10月, ばんえい除く. 出力: analysis/out/kesi_*.csv
"""
import pandas as pd

import trifecta_factors as T

KEY = T.KEY


def main():
    h, _, _ = T.load()
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"].sort_values(["馬ID", "日付", "レース番号"])
    g = h.groupby("馬ID")
    h["前々走脚質"] = g["脚質"].shift(2)
    h["前走距離"] = g["距離"].shift(1)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    d = h[(h["年"] >= 2024) & h["人気"].notna() & h["着順"].notna()].copy()
    gr = d.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)
    d["期待"] = gr["top3"].transform("mean")
    low = d["指数順"] >= 6
    a = low & (d["前走脚質"] == "追込")
    b = a & (d["前々走脚質"] == "追込")
    c = a & ((d["前走距離"] - d["距離"]).abs() <= 200)
    rules = {"A いま (前走追込)": a, "B 2走とも追込": b, "C 前走の距離が近い(±200m)": c,
             "D B または C": b | c, "E B かつ C": b & c, "参考: A から外れる馬 (A かつ Dでない)": a & ~(b | c)}
    rows = []
    for name, m in rules.items():
        x = d[m]
        r = {"条件": name, "消した頭数": len(x), "1レースあたり": len(x) / d.groupby(KEY).ngroups,
             "3着内率%": x["top3"].mean() * 100, "3着内A/E": x["top3"].sum() / x["期待"].sum(),
             "1着の数": int((x["着順"] == 1).sum()), "勝率%": (x["着順"] == 1).mean() * 100, "平均人気": x["人気"].mean()}
        for y in (2024, 2025, 2026):
            xy = x[x["年"] == y]
            r[f"{y}年 3着内A/E"] = xy["top3"].sum() / xy["期待"].sum()
        rows.append(r)
    t = pd.DataFrame(rows).set_index("条件")
    t.to_csv(T.OUT / "kesi_rules.csv", encoding="utf-8-sig")
    by_c = pd.DataFrame({name: d[m].groupby("競馬場").apply(lambda x: x["top3"].sum() / x["期待"].sum())
                         for name, m in rules.items() if not name.startswith("参考")})
    by_c.to_csv(T.OUT / "kesi_rules_by_course.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    print(by_c.round(2).to_string())


if __name__ == "__main__":
    main()
