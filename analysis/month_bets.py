"""確定人気だけで決まる単純な買い方を, 2026年11月 (11/1〜) と他の期間で比べる.

使い方: python3 analysis/month_bets.py <data>
  2026年11月で「うまくいった買い方」が, 9〜10月・過去の11月 (2021〜2025年) でも通用するかを確かめる.
  オッズの形のルール (odds_bets) は締切オッズが要るので, 11月の締切オッズがそろうまでは比べられない.
対象: ばんえいを除く14場, 5頭以上. 人気は確定人気 (同じ人気は馬番の小さい方を先).
買い方 (各100円): 単勝・複勝・馬連・ワイド・馬単・３連複・３連単 の人気順の組み合わせ (下の PLANS).
回収率の90%区間はレース単位のブートストラップ.
出力: analysis/out/mb_summary.csv, mb_course.csv
"""
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
FILES = [(f"{y}_horselist_全月まとめ.csv", f"pay{y}.csv") for y in range(2021, 2026)] + [
    ("2026_horselist_1-8月まとめ.csv", "pay2026.csv"),
    ("202609_horselist.csv", "202609_payback.csv"),
    ("202610_horselist.csv", "202610_payback.csv"),
    ("202611_horselist.csv", "202611_payback.csv")]


def P(*xs):  # 人気の番号 (1始まり) → 組
    return xs


PLANS = {
    "単勝 1番人気": ("単勝", [P(1)]),
    "単勝 2番人気": ("単勝", [P(2)]),
    "単勝 3番人気": ("単勝", [P(3)]),
    "単勝 4〜6番人気": ("単勝", [P(4), P(5), P(6)]),
    "複勝 1番人気": ("複勝", [P(1)]),
    "複勝 2番人気": ("複勝", [P(2)]),
    "複勝 3番人気": ("複勝", [P(3)]),
    "ワイド 1-2": ("ワイド", [P(1, 2)]),
    "ワイド 1-3": ("ワイド", [P(1, 3)]),
    "ワイド 2-3": ("ワイド", [P(2, 3)]),
    "ワイド 1軸 2〜5": ("ワイド", [P(1, x) for x in range(2, 6)]),
    "馬連 1-2": ("馬連", [P(1, 2)]),
    "馬連 1軸 2〜5": ("馬連", [P(1, x) for x in range(2, 6)]),
    "馬連 2-3-4 BOX": ("馬連", [P(a, b) for a, b in combinations([2, 3, 4], 2)]),
    "馬単 1→2〜5 (今のルール)": ("馬単", [P(1, x) for x in range(2, 6)]),
    "馬単 2〜5→1 (裏)": ("馬単", [P(x, 1) for x in range(2, 6)]),
    "馬単 2→1,3,4,5": ("馬単", [P(2, x) for x in (1, 3, 4, 5)]),
    "３連複 1-2-3": ("３連複", [P(1, 2, 3)]),
    "３連複 1軸 2〜5": ("３連複", [P(1, a, b) for a, b in combinations(range(2, 6), 2)]),
    "３連複 1-2軸 3〜7": ("３連複", [P(1, 2, x) for x in range(3, 8)]),
    "３連複 1〜5 BOX": ("３連複", [P(*c) for c in combinations(range(1, 6), 3)]),
    "３連複 2〜6 BOX (1番人気消し)": ("３連複", [P(*c) for c in combinations(range(2, 7), 3)]),
    "３連単 1→2,3,4→2,3,4 (今のルール)": ("３連単", [P(1, a, b) for a, b in permutations([2, 3, 4], 2)]),
    "３連単 1,2→1,2→3〜6": ("３連単", [P(a, b, c) for a, b in [(1, 2), (2, 1)] for c in range(3, 7)]),
    "３連単 2,3→1→2〜5": ("３連単", [P(a, 1, c) for a in (2, 3) for c in range(2, 6) if c != a]),
}


def read(f, **kw):
    return pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, **kw)


def load():
    hs, ps = [], []
    for hf, pf in FILES:
        if not ((DATA / hf).exists() and (DATA / pf).exists()):
            continue
        h = read(hf, usecols=KEY + ["馬番", "人気", "着順"])
        for c in ["馬番", "人気", "着順"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")
        hs.append(h[(h["競馬場"] != "帯広ば") & h["人気"].notna()])
        ps.append(read(pf))
    h = pd.concat(hs).drop_duplicates(KEY + ["馬番"], keep="last")
    p = pd.concat(ps).drop_duplicates(KEY, keep="last")
    p = p[p["競馬場"] != "帯広ば"].set_index(KEY)
    return h, p


def iv(x):
    return int(x) if pd.notna(x) else None


def hits(pr):
    """券種ごとの 当たり組 → 払戻 (100円あたり)"""
    d = {"単勝": {}, "複勝": {}, "ワイド": {}, "馬連": {}, "馬単": {}, "３連複": {}, "３連単": {}}
    if iv(pr["単勝組番"]):
        d["単勝"][(iv(pr["単勝組番"]),)] = pr["単勝払戻金（円）"]
    for i in (1, 2, 3):
        n = iv(pr.get(f"複勝組番{i}"))
        if n:
            d["複勝"][(n,)] = pr[f"複勝払戻金{i}（円）"]
        a, b = iv(pr.get(f"ワイド組番{i}馬番1")), iv(pr.get(f"ワイド組番{i}馬番2"))
        if a and b:
            d["ワイド"][frozenset((a, b))] = pr[f"ワイド払戻金{i}（円）"]
    a, b = iv(pr["馬複組番1"]), iv(pr["馬複組番2"])
    if a and b:
        d["馬連"][frozenset((a, b))] = pr["馬複払戻金（円）"]
    a, b = iv(pr["馬単組番1"]), iv(pr["馬単組番2"])
    if a and b:
        d["馬単"][(a, b)] = pr["馬単払戻金（円）"]
    t = tuple(iv(pr[f"３連複組番馬番{i}"]) for i in (1, 2, 3))
    if all(t):
        d["３連複"][frozenset(t)] = pr["３連複払戻金（円）"]
    t = tuple(iv(pr[f"３連単組番馬番{i}"]) for i in (1, 2, 3))
    if all(t):
        d["３連単"][t] = pr["３連単払戻金（円）"]
    return d


def races(h, p):
    rows = []
    for k, g in h.groupby(KEY):
        if k not in p.index or len(g) < 5:
            continue
        pr = p.loc[k]
        if pd.isna(pr["単勝組番"]) or pd.isna(pr["３連単組番馬番1"]):
            continue
        nos = g.sort_values(["人気", "馬番"])["馬番"].astype(int).tolist()
        hd = hits(pr)
        row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "頭数": len(nos)}
        for name, (kind, combos) in PLANS.items():
            pts, ret = 0, 0.0
            for c in combos:
                if max(c) > len(nos):
                    continue
                t = tuple(nos[i - 1] for i in c)
                key = frozenset(t) if kind in ("ワイド", "馬連", "３連複") else t
                pts += 1
                ret += hd[kind].get(key, 0)
            row[name + "_点"], row[name + "_払"] = pts, ret
        rows.append(row)
    return pd.DataFrame(rows)


def roi(g, n):
    return g[n + "_払"].sum() / max(g[n + "_点"].sum() * 100, 1) * 100


def ci(g, n, rng, B=1000):
    inv, ret = (g[n + "_点"] * 100).to_numpy(), g[n + "_払"].to_numpy()
    idx = rng.integers(0, len(g), (B, len(g)))
    r = ret[idx].sum(1) / np.maximum(inv[idx].sum(1), 1) * 100
    return np.percentile(r, [5, 95])


def main():
    h, p = load()
    d = races(h, p)
    m = d["競走年月日"] // 100
    periods = {"2026年11月": d[m == 202611], "2026年9〜10月": d[(m >= 202609) & (m <= 202610)],
               "2026年1〜8月": d[(m >= 202601) & (m <= 202608)],
               "過去の11月 (2021〜25)": d[(m % 100 == 11) & (m < 202600)], "2021〜2025年 全体": d[m < 202600]}
    print("レース数:", {k: len(v) for k, v in periods.items()})
    rng = np.random.default_rng(0)
    rows = []
    for name in PLANS:
        r = {"買い方": name}
        for pk, g in periods.items():
            r[pk] = roi(g, name)
        lo, hi = ci(periods["2026年11月"], name, rng)
        r["11月の90%区間"] = f"{lo:.0f}〜{hi:.0f}"
        g = periods["2026年11月"]
        r["11月の的中率%"] = (g[name + "_払"] > 0).mean() * 100
        rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    print(t.round(1).to_string(index=False))
    t.round(1).to_csv(OUT / "mb_summary.csv", index=False, encoding="utf-8-sig")

    # 11月の週ごと (日付の前半・後半) と 競馬場ごと
    g = periods["2026年11月"]
    half = np.where(g["競走年月日"] <= 20261109, "11/1〜9", "11/10〜18")
    rows = []
    for name in PLANS:
        r = {"買い方": name}
        for hk in ["11/1〜9", "11/10〜18"]:
            r[hk] = roi(g[half == hk], name)
        rows.append(r)
    print("\n== 11月の前半・後半\n", pd.DataFrame(rows).round(0).to_string(index=False))
    rows = []
    for c, gc in g.groupby("競馬場"):
        r = {"競馬場": c, "レース数": len(gc)}
        for name in ["単勝 1番人気", "複勝 1番人気", "ワイド 1-2", "馬単 1→2〜5 (今のルール)", "３連単 1→2,3,4→2,3,4 (今のルール)",
                     "３連複 1〜5 BOX", "３連複 2〜6 BOX (1番人気消し)"]:
            r[name] = roi(gc, name)
        rows.append(r)
    tc = pd.DataFrame(rows)
    print("\n== 2026年11月 競馬場ごと\n", tc.round(0).to_string(index=False))
    tc.round(1).to_csv(OUT / "mb_course.csv", index=False, encoding="utf-8-sig")
    # 1番人気の勝率の推移
    for pk, gg in periods.items():
        w = (gg["単勝 1番人気_払"] > 0).mean() * 100
        print(f"1番人気の勝率 {pk}: {w:.1f}%")


if __name__ == "__main__":
    main()
