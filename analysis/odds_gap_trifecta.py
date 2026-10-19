"""1〜4番人気の単勝オッズの差 (オッズの崖の位置) ごとに, 上位人気の３連単の組み合わせの回収率を比べる.

使い方: python3 analysis/odds_gap_trifecta.py <data>
  確定オッズ (data/odds/2026MM_*_odds.csv) がある 2026年9月〜10月のレースが対象. 人気は確定単勝オッズの順.
  9月で傾向を見つけ, 10月で同じ傾向が出るかを確かめる (組み合わせを多く比べるので, 片方だけの好成績は偶然の可能性が高い).
オッズの崖: 隣り合う人気のオッズ比 (2番人気/1番人気, 3番/2番, 4番/3番, 5番/4番) が最も大きい位置.
  1強 = 1番人気と2番人気の間が最も開く, 2強 = 2・3番人気の間, 3強 = 3・4番人気の間, 4強 = 4・5番人気の間.
※ 確定オッズは締切後の値. 買う時点のオッズとは少し違う.
出力: analysis/out/ogt_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
MONTHS = {"9月": "202609", "10月": "202610"}
SHAPES = ["1強", "2強", "3強", "4強"]


def perm(A, B, C):
    return [(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3]


# 人気番号で書いた買い方
BETS = {
    "1-2-3 (1点)": [(1, 2, 3)],
    "1→2,3→2,3 (2点)": perm([1], [2, 3], [2, 3]),
    "上位3頭BOX (6点)": perm([1, 2, 3], [1, 2, 3], [1, 2, 3]),
    "1→2,3→2,3,4 (4点)": perm([1], [2, 3], [2, 3, 4]),
    "1→2,3,4→2,3,4 (6点)": perm([1], [2, 3, 4], [2, 3, 4]),
    "1,2→1,2→3,4 (4点)": perm([1, 2], [1, 2], [3, 4]),
    "1,2→1,2,3→1,2,3,4 (8点)": perm([1, 2], [1, 2, 3], [1, 2, 3, 4]),
    "1,2→1,2,3,4→1,2,3,4 (12点)": perm([1, 2], [1, 2, 3, 4], [1, 2, 3, 4]),
    "上位4頭BOX (24点)": perm([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]),
    "2,3,4→1→2,3,4 (6点)": perm([2, 3, 4], [1], [2, 3, 4]),
    "2,3,4→2,3,4→1 (6点)": perm([2, 3, 4], [2, 3, 4], [1]),
    "2,3,4BOX 1番人気消し (6点)": perm([2, 3, 4], [2, 3, 4], [2, 3, 4]),
    "1→2,3,4→2〜7 (15点)": perm([1], [2, 3, 4], [2, 3, 4, 5, 6, 7]),
}


def load():
    rows = []
    for mname, ym in MONTHS.items():
        files = sorted((DATA / "odds").glob(f"{ym}_*_odds.csv"))
        if not files:
            continue
        o = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False,
                                   usecols=KEY + ["賭式", "番号1", "オッズ"]) for f in files])
        w = o[(o["賭式"] == "単勝")].copy()
        w["オッズ"] = pd.to_numeric(w["オッズ"], errors="coerce")
        w = w[w["オッズ"] > 0]
        pay = pd.read_csv(DATA / f"{ym}_payback.csv", encoding="utf-8-sig", low_memory=False).drop_duplicates(KEY)
        cols = ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]
        pay = pay.dropna(subset=cols).set_index(KEY)[cols]
        for k, g in w.groupby(KEY):
            if k not in pay.index or len(g) < 5:
                continue
            g = g.sort_values(["オッズ", "番号1"])
            nos = g["番号1"].astype(int).tolist()
            od = g["オッズ"].tolist()
            pr = pay.loc[k]
            res = tuple(int(pr[c]) for c in cols[:3])
            pop = {n: i + 1 for i, n in enumerate(nos)}           # 馬番 → 人気 (オッズ順)
            rp = tuple(pop.get(n, 99) for n in res)               # 結果を人気で表す
            ratios = [od[i + 1] / od[i] for i in range(4)]
            rows.append({"月": mname, "競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "頭数": len(g),
                         "o1": od[0], "o2": od[1], "o3": od[2], "o4": od[3], "o5": od[4],
                         "r12": ratios[0], "r23": ratios[1], "r34": ratios[2], "r45": ratios[3],
                         "形": SHAPES[int(np.argmax(ratios))], "結果人気": rp, "配当": float(pr[cols[3]])})
    return pd.DataFrame(rows)


def evaluate(g):
    """買い方ごとの 的中率・回収率"""
    out = {"レース数": len(g)}
    for name, combos in BETS.items():
        s = set(combos)
        hit = g["結果人気"].map(lambda r: r in s)
        out[name + " 的中%"] = hit.mean() * 100
        out[name + " 回収率%"] = g.loc[hit, "配当"].sum() / (len(g) * len(combos) * 100) * 100
    return pd.Series(out)


def boot(g, name, n=2000):
    s = set(BETS[name])
    ret = np.where(g["結果人気"].map(lambda r: r in s), g["配当"], 0.0) / (len(BETS[name]) * 100)
    idx = np.random.default_rng(0).integers(0, len(ret), (n, len(ret)))
    return np.percentile(ret[idx].mean(1) * 100, [5, 95])


def main():
    d = load()
    d["1人気オッズ帯"] = pd.cut(d["o1"], [0, 1.5, 2.0, 3.0, 99], labels=["~1.4倍", "1.5~1.9倍", "2.0~2.9倍", "3.0倍~"], right=False)
    d.drop(columns=["結果人気"]).assign(結果人気=d["結果人気"].map(lambda r: "-".join(map(str, r)))) \
        .to_csv(OUT / "ogt_races.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
    print(f"対象: {len(d)}レース (9月 {(d['月'] == '9月').sum()}, 10月 {(d['月'] == '10月').sum()})")
    print("\n== オッズの崖の位置ごとの件数と, 各人気のオッズの中央値")
    print(d.groupby("形")[["o1", "o2", "o3", "o4", "o5"]].median().assign(レース数=d.groupby("形").size()).round(1).to_string())

    # 形 × 買い方 (9月・10月・計)
    res = []
    for shape in SHAPES + ["全体"]:
        g0 = d if shape == "全体" else d[d["形"] == shape]
        for m in ["9月", "10月", "計"]:
            g = g0 if m == "計" else g0[g0["月"] == m]
            if len(g):
                res.append(evaluate(g).rename((shape, m)))
    t = pd.DataFrame(res)
    t.index = pd.MultiIndex.from_tuples(t.index, names=["形", "月"])
    t.round(1).to_csv(OUT / "ogt_shape_bets.csv", encoding="utf-8-sig")
    for shape in SHAPES + ["全体"]:
        x = t.loc[shape]
        rows = []
        for name in BETS:
            rows.append({"買い方": name,
                         **{f"{m} 的中%": x.loc[m, name + " 的中%"] for m in x.index},
                         **{f"{m} 回収率%": x.loc[m, name + " 回収率%"] for m in x.index}})
        tab = pd.DataFrame(rows).set_index("買い方")
        g = d if shape == "全体" else d[d["形"] == shape]
        tab["計 90%区間"] = [f"{lo:.0f}〜{hi:.0f}" for lo, hi in (boot(g, n) for n in BETS)]
        n9, n10 = int(x.loc["9月", "レース数"]) if "9月" in x.index else 0, int(x.loc["10月", "レース数"]) if "10月" in x.index else 0
        print(f"\n== {shape} (9月 {n9}・10月 {n10} レース)")
        print(tab.round(1).to_string())

    # 1番人気オッズ帯 × 形 の主な買い方
    main_bets = ["1→2,3→2,3,4 (4点)", "1→2,3,4→2,3,4 (6点)", "1,2→1,2→3,4 (4点)", "上位3頭BOX (6点)",
                 "上位4頭BOX (24点)", "2,3,4→1→2,3,4 (6点)"]
    rows = []
    for (band, shape), g in d.groupby(["1人気オッズ帯", "形"], observed=True):
        if len(g) < 40:
            continue
        e = evaluate(g)
        rows.append({"1人気オッズ帯": band, "形": shape, "レース数": len(g),
                     **{b: e[b + " 回収率%"] for b in main_bets}})
    t2 = pd.DataFrame(rows)
    print("\n== 1番人気のオッズ帯 × 形 の回収率% (40レース以上)\n", t2.round(1).to_string(index=False))
    t2.round(2).to_csv(OUT / "ogt_band_shape.csv", index=False, encoding="utf-8-sig")

    # 上位4人気の24通りの並び: 形ごとの的中率と回収率 (9月・10月)
    rows = []
    for shape in SHAPES:
        g = d[d["形"] == shape]
        for c in perm([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]):
            r = {"形": shape, "並び": "-".join(map(str, c))}
            for m in ["9月", "10月"]:
                gm = g[g["月"] == m]
                hit = gm["結果人気"] == c
                r[f"{m} 的中"] = int(hit.sum())
                r[f"{m} 回収率%"] = gm.loc[hit, "配当"].sum() / max(len(gm) * 100, 1) * 100
            rows.append(r)
    t3 = pd.DataFrame(rows)
    t3["両月とも100%超"] = (t3["9月 回収率%"] > 100) & (t3["10月 回収率%"] > 100)
    t3.round(1).to_csv(OUT / "ogt_top4_perms.csv", index=False, encoding="utf-8-sig")
    both = t3[t3["両月とも100%超"]]
    print(f"\n== 上位4人気の並び (4形×24通り=96) のうち, 9月・10月とも回収率100%超: {len(both)}通り")
    print(both.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
