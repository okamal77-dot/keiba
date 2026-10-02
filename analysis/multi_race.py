"""5重勝 (後半5Rの1着) / トリプル馬単 (後半3Rの1-2着) の的中率シミュレーション.

各開催日の最終 N レースを対象とみなし, 人気上位から買う戦略の的中率を実績で検証する.
- 5重勝: 単勝人気 k 位以内に勝ち馬がいるか
- トリプル馬単: 馬単人気 k 位以内に結果があるか
点数配分は「頭数が少ないレースほど少なく」割り当てる.
"""
import itertools
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
NANKAN = ["大井", "川崎", "浦和", "船橋"]


def load():
    ps = []
    for y, hf in YEARS.items():
        p = pd.read_csv(DATA / f"pay{y}.csv", encoding="utf-8-sig", low_memory=False).drop_duplicates(KEY)
        n = pd.read_csv(DATA / hf, encoding="utf-8-sig", usecols=KEY, low_memory=False).groupby(KEY).size().rename("頭数")
        ps.append(p.merge(n, left_on=KEY, right_index=True))
    p = pd.concat(ps)
    for c in ["単勝人気", "馬単人気1", "単勝払戻金（円）", "馬単払戻金（円）"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    return p.dropna(subset=["単勝人気", "馬単人気1"])


def last_n(p, n):
    p = p.sort_values(KEY)
    p["残り"] = p.groupby(["競馬場", "競走年月日"]).cumcount(ascending=False)
    d = p[p["残り"] < n].copy()
    ok = d.groupby(["競馬場", "競走年月日"]).size() == n
    d = d.set_index(["競馬場", "競走年月日"]).loc[ok[ok].index].reset_index()
    d["順"] = d.groupby(["競馬場", "競走年月日"]).cumcount()
    return d


def patterns(n, budget):
    """各レース点数(昇順)の組で 積<=budget のうち極大なもの"""
    cands = [1, 2, 3, 4, 5, 6, 8]
    res = set()
    for c in itertools.combinations_with_replacement(cands, n):
        prod = 1
        for x in c:
            prod *= x
        if prod <= budget and prod * 2 > budget * 0.66:
            res.add(c)
    return sorted(res, key=lambda c: -eval("*".join(map(str, c))))


def simulate(d, col, pats, paycol):
    """pats: 昇順の点数タプル. 頭数の少ない順に割り当て. 日単位の的中率と配当目安."""
    rows = []
    for pat in pats:
        hits, pars = [], []
        for (jo, day), g in d.groupby(["競馬場", "競走年月日"]):
            g = g.sort_values(["頭数", "順"])
            ok = (g[col].values <= pd.Series(pat).values).all()
            hits.append((jo, ok))
            if ok:
                pars.append((jo, (g[paycol] / 100).prod() * 100))
        h = pd.DataFrame(hits, columns=["競馬場", "hit"])
        pr = pd.DataFrame(pars, columns=["競馬場", "par"])
        r = h.groupby("競馬場")["hit"].agg(["mean", "size"])
        r["配当目安中央値"] = pr.groupby("競馬場")["par"].median()
        r["pattern"] = "×".join(map(str, pat))
        r["点数"] = eval("*".join(map(str, pat)))
        rows.append(r.reset_index())
    return pd.concat(rows)


def main():
    p = load()
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)

    # ---- 5重勝 ----
    d5 = last_n(p, 5)
    per = pd.DataFrame({f"単勝{k}人気内": d5.groupby("競馬場")["単勝人気"].apply(lambda x: (x <= k).mean() * 100) for k in (1, 2, 3, 4)})
    per["平均頭数"] = d5.groupby("競馬場")["頭数"].mean()
    per["日数"] = d5.groupby("競馬場")["競走年月日"].nunique()
    # 5Rのうち勝ち馬が1-2人気だったレース数の分布
    cnt = d5.assign(t=d5["単勝人気"] <= 2).groupby(["競馬場", "競走年月日"])["t"].sum()
    dist = pd.crosstab(cnt.index.get_level_values(0), cnt.values, normalize="index") * 100
    print("== 5重勝: 後半5R 各レースの的中率(%)\n", per.reindex(ORDER).round(1))
    print("\n== 5レース中 1-2人気が勝った数の分布(%)\n", dist.reindex(ORDER).round(1))
    pats5 = [(2, 2, 2, 2, 2), (1, 2, 2, 2, 4), (1, 1, 2, 4, 4), (1, 2, 2, 3, 3), (1, 1, 3, 3, 3), (1, 1, 2, 3, 5),
             (3, 3, 3, 3, 3), (1, 1, 1, 1, 1), (2, 2, 2, 3, 3), (1, 2, 3, 4, 5)]
    s5 = simulate(d5, "単勝人気", pats5, "単勝払戻金（円）")
    t5 = s5.pivot_table(index="競馬場", columns=["点数", "pattern"], values="mean").reindex(ORDER) * 100
    print("\n== 5重勝 戦略別 的中率(%) (頭数の少ないレースに少ない点数)\n", t5.round(1).to_string())
    m5 = s5.pivot_table(index="競馬場", columns="pattern", values="配当目安中央値").reindex(ORDER)
    print("\n== 的中時の単勝転がし配当 中央値(円)\n", m5.round(-1).to_string())

    # ---- トリプル馬単 ----
    d3 = last_n(p[p["競馬場"].isin(NANKAN)], 3)
    per3 = pd.DataFrame({f"馬単{k}人気内": d3.groupby("競馬場")["馬単人気1"].apply(lambda x: (x <= k).mean() * 100) for k in (1, 2, 3, 4, 6, 10)})
    per3["平均頭数"] = d3.groupby("競馬場")["頭数"].mean()
    per3["日数"] = d3.groupby("競馬場")["競走年月日"].nunique()
    print("\n== トリプル馬単: 後半3R 各レースの的中率(%)\n", per3.round(1))
    pats3 = [(1, 1, 6), (1, 2, 3), (1, 1, 4), (1, 2, 2), (2, 2, 2), (1, 3, 4), (2, 2, 3), (2, 3, 4), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (10, 10, 10)]
    s3 = simulate(d3, "馬単人気1", pats3, "馬単払戻金（円）")
    t3 = s3.pivot_table(index="競馬場", columns=["点数", "pattern"], values="mean") * 100
    print("\n== トリプル馬単 戦略別 的中率(%)\n", t3.round(2).to_string())
    m3 = s3.pivot_table(index="競馬場", columns="pattern", values="配当目安中央値")
    print("\n== 的中時の馬単転がし配当 中央値(円)\n", m3.round(-2).to_string())
    allp = s3.groupby(["点数", "pattern"]).apply(lambda x: (x["mean"] * x["size"]).sum() / x["size"].sum() * 100, include_groups=False)
    print("\n== 南関東4場計\n", allp.round(2))

    per.to_csv(OUT / "win5_per_race.csv", encoding="utf-8-sig")
    t5.to_csv(OUT / "win5_strategy.csv", encoding="utf-8-sig")
    per3.to_csv(OUT / "triple_per_race.csv", encoding="utf-8-sig")
    t3.to_csv(OUT / "triple_strategy.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
