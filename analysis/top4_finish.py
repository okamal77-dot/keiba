"""上位人気4頭で1-3着が決まるレースの割合と, その時の1番人気単勝オッズ.

オッズは配当データからしか分からないため, 1番人気が勝った場合の単勝払戻/100 を使う.
"""
import sys
from pathlib import Path

import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
YEARS = {2023: "2023_horselist_全月まとめ.csv", 2024: "2024_horselist_全月まとめ.csv",
         2025: "2025_horselist_全月まとめ.csv", 2026: "2026_horselist_1-8月まとめ.csv"}
ORDER = ["帯広ば", "大井", "川崎", "浦和", "船橋", "名古屋", "笠松", "園田", "姫路", "金沢",
         "高知", "佐賀", "門別", "盛岡", "水沢"]
BANDS = [1.0, 1.5, 2.0, 3.0, 4.0, 100]
BLAB = ["1.0-1.4倍", "1.5-1.9倍", "2.0-2.9倍", "3.0-3.9倍", "4.0倍~"]


def main():
    races = []
    for y, hf in YEARS.items():
        p = pd.read_csv(DATA / f"pay{y}.csv", encoding="utf-8-sig", low_memory=False).drop_duplicates(KEY)
        h = pd.read_csv(DATA / hf, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["人気", "着順"])
        h["人気"] = pd.to_numeric(h["人気"], errors="coerce")
        h["着順"] = pd.to_numeric(h["着順"], errors="coerce")
        top = h[h["着順"] <= 3].sort_values(KEY + ["着順"])
        g = top.groupby(KEY)["人気"].agg(list).rename("pops")
        n = h.groupby(KEY).size().rename("頭数")
        p = p.merge(g, left_on=KEY, right_index=True).merge(n, left_on=KEY, right_index=True)
        races.append(p[KEY + ["pops", "頭数", "単勝人気", "単勝払戻金（円）", "３連単払戻金（円）", "馬単払戻金（円）"]])
    r = pd.concat(races, ignore_index=True)
    r = r[r["pops"].str.len() >= 3]
    r["top4_3"] = r["pops"].apply(lambda x: max(x[:3]) <= 4)        # 1-3着すべて4番人気以内
    r["top4_2"] = r["pops"].apply(lambda x: max(x[:2]) <= 4)        # 1-2着が4番人気以内
    r["top3_3"] = r["pops"].apply(lambda x: max(x[:3]) <= 3)        # 1-3着すべて3番人気以内
    r["fav_in"] = r["pops"].apply(lambda x: 1 in x[:3])
    r["fav_win"] = r["単勝人気"] == 1
    r["fav_odds"] = r["単勝払戻金（円）"].where(r["fav_win"]) / 100

    g = r.groupby("競馬場")
    t4 = r[r["top4_3"]]
    g4 = t4.groupby("競馬場")
    tab = pd.DataFrame({
        "レース数": g.size(),
        "1-3着が上位4人気": g["top4_3"].mean() * 100,
        "うち上位3人気": g["top3_3"].mean() * 100,
        "1-2着が上位4人気": g["top4_2"].mean() * 100,
        "4人気決着時_1人気1着": g4["fav_win"].mean() * 100,
        "4人気決着時_1人気3着内": g4["fav_in"].mean() * 100,
        "4人気決着時_1人気勝ちオッズ中央値": g4["fav_odds"].median(),
        "4人気決着時_1人気勝ちオッズ平均": g4["fav_odds"].mean(),
        "全体_1人気勝ちオッズ中央値": g["fav_odds"].median(),
        "4人気決着時_3連単中央値": g4["３連単払戻金（円）"].median(),
    })
    # 頭数別
    r["頭数帯"] = pd.cut(r["頭数"], [0, 8, 10, 12, 99], labels=["~8頭", "9-10頭", "11-12頭", "13頭~"])
    byn = r.pivot_table(index="競馬場", columns="頭数帯", values="top4_3", aggfunc="mean", observed=False) * 100
    # 1番人気が勝ったレースで, そのオッズ帯ごとに 4人気決着になった率
    w = r[r["fav_win"]].copy()
    w["帯"] = pd.cut(w["fav_odds"], BANDS, right=False, labels=BLAB)
    band = w.pivot_table(index="競馬場", columns="帯", values="top4_3", aggfunc="mean", observed=False) * 100
    band_n = w.pivot_table(index="競馬場", columns="帯", values="top4_3", aggfunc="size", observed=False)
    band_share = pd.crosstab(t4[t4["fav_win"]]["競馬場"], pd.cut(t4[t4["fav_win"]]["fav_odds"], BANDS, right=False, labels=BLAB), normalize="index") * 100
    for name, df in [("top4_summary", tab), ("top4_by_fieldsize", byn), ("top4_rate_by_fav_odds", band),
                     ("top4_n_by_fav_odds", band_n), ("top4_fav_odds_share", band_share)]:
        tot = None
        df = df.reindex(ORDER)
        df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")
        print(f"\n== {name}\n", df.round(1).to_string())
    # 全体
    print("\n== 全場計")
    print({"1-3着上位4人気": r["top4_3"].mean() * 100, "上位3人気": r["top3_3"].mean() * 100,
           "1-2着上位4人気": r["top4_2"].mean() * 100, "4決着時1人気勝": t4["fav_win"].mean() * 100,
           "4決着時オッズ中央値": t4["fav_odds"].median(), "全体オッズ中央値": r["fav_odds"].median(),
           "4決着時3連単中央値": t4["３連単払戻金（円）"].median()})
    print(w.groupby("帯", observed=False)["top4_3"].agg(["mean", "size"]))
    print(pd.cut(t4.loc[t4["fav_win"], "fav_odds"], BANDS, right=False, labels=BLAB).value_counts(normalize=True).sort_index() * 100)


if __name__ == "__main__":
    main()
