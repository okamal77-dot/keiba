"""期待値の高い馬の選定.

発売前はオッズが分からないため, 想定の単勝オッズ = その場の単勝払戻率 ÷ 同じ人気順位 (想定人気) の過去の勝率 とし,
  期待値 = モデルの勝率 × 想定オッズ   (1を超えると, 人気から見てモデルの評価が高い)
とする. 実際のオッズが分かれば 期待値 = 勝率 × オッズ.
人気順位ごとの勝率と払戻率は, 2024年以降の各場 (人気 1〜10以上 × 頭数帯) から求める.

このファイルを実行すると, 直近3か月 (pred2_underrated_rows_*.csv の6場) で
「想定の期待値が高い馬の単勝」を買った場合の成績を確かめる. 出力: analysis/out/ev_*.csv
"""
import glob

import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY
BANDS = [0, 8, 10, 12, 99]


def market_table(h):
    """(競馬場, 人気c, 頭数帯) → 勝率, 競馬場 → 単勝払戻率"""
    d = h[(h["年"] >= 2024) & h["人気"].notna() & h["着順"].notna()].copy()
    d["人気c"] = d["人気"].clip(upper=10)
    d["頭数帯"] = pd.cut(d["頭数"], BANDS)
    q = d.groupby(["競馬場", "人気c", "頭数帯"], observed=True)["着順"].apply(lambda s: (s == 1).mean()).rename("人気の勝率")
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    raw = raw[pd.to_numeric(raw["競走年月日"], errors="coerce") >= 20240101]
    pay = raw.merge(d.drop_duplicates(KEY)[KEY], on=KEY).groupby("競馬場")["単勝払戻金（円）"].sum()
    n = d.groupby("競馬場").size()
    rate = (pay.reindex(n.index) / (n * 100)).clip(0.6, 0.9).fillna(0.75).rename("払戻率")
    return q, rate


def expected_value(sub, win, pop, q, rate):
    """sub: 競馬場・頭数 を持つ行, win: モデル勝率, pop: 想定人気 (または実際の人気)"""
    k = pd.MultiIndex.from_arrays([sub["競馬場"].values, np.clip(np.asarray(pop, float), 1, 10),
                                   pd.cut(sub["頭数"], BANDS).values])
    qq = q.reindex(k).values
    rr = rate.reindex(sub["競馬場"].values).values
    odds = rr / np.where(np.isnan(qq) | (qq <= 0), np.nan, qq)
    return np.asarray(win, float) * odds, odds


def main():
    h, _, _ = T.load()
    h = T.features(h)
    q, rate = market_table(h)
    print("単勝払戻率", rate.round(3).to_dict())
    m = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "pred2_underrated_rows_*.csv"))])
    m = m.merge(h[KEY + ["馬番", "頭数"]], on=KEY + ["馬番"], how="left")
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    tan = raw[KEY + ["単勝組番", "単勝払戻金（円）"]].dropna()
    tan = tan.assign(馬番=tan["単勝組番"].astype(int)).groupby(KEY + ["馬番"])["単勝払戻金（円）"].sum().rename("単勝")
    m = m.merge(tan.reset_index(), on=KEY + ["馬番"], how="left").fillna({"単勝": 0})
    m = m[m["人気"].notna()]
    m["期待値(想定人気)"], _ = expected_value(m, m["勝率"], m["想定人気"], q, rate)
    m["期待値(実際の人気)"], _ = expected_value(m, m["勝率"], m["人気"], q, rate)
    m["期待値順"] = m.groupby(KEY)["期待値(想定人気)"].rank(ascending=False, method="first")
    rows = []
    for basis in ["期待値(想定人気)", "期待値(実際の人気)"]:
        for lo in [0.8, 1.0, 1.2, 1.5, 2.0]:
            for kesi_ok in [True]:
                x = m[(m[basis] >= lo) & ~m["消し"]]
                rows.append({"基準": basis, "期待値の下限": lo, "頭数": len(x), "1レースあたり": len(x) / m.groupby(KEY).ngroups,
                             "勝率%": (x["着順"] == 1).mean() * 100, "平均人気": x["人気"].mean(),
                             "単勝回収率%": x["単勝"].mean(), "上位3本を除く回収率%": (x["単勝"].sum() - x["単勝"].nlargest(3).sum()) / max(len(x), 1)})
    for r in [1, 2, 3]:
        x = m[m["期待値順"] == r]
        rows.append({"基準": f"各レースの期待値{r}位", "頭数": len(x), "勝率%": (x["着順"] == 1).mean() * 100, "平均人気": x["人気"].mean(),
                     "単勝回収率%": x["単勝"].mean(), "上位3本を除く回収率%": (x["単勝"].sum() - x["単勝"].nlargest(3).sum()) / len(x)})
    x = m[m["勝率順"] == 1]
    rows.append({"基準": "参考: ◎ (勝率1位)", "頭数": len(x), "勝率%": (x["着順"] == 1).mean() * 100, "平均人気": x["人気"].mean(),
                 "単勝回収率%": x["単勝"].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(T.OUT / "ev_backtest.csv", index=False, encoding="utf-8-sig")
    m["月"] = m["競走年月日"] // 100
    mo = m[(m["期待値(想定人気)"] >= 1.2) & ~m["消し"]].groupby("月")["単勝"].mean()
    c = m[(m["期待値(想定人気)"] >= 1.2) & ~m["消し"]].groupby("競馬場")["単勝"].agg(["size", "mean"])
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    print("期待値1.2以上 月別回収率", mo.round(0).to_dict())
    print(c.round(0).to_string())


if __name__ == "__main__":
    main()
