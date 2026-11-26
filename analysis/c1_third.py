"""３連単 1番人気→2,3番人気→全部 (tri_flow.py の C1) で, 3着に来る馬の特徴 (人気・脚質など).

使い方: python3 analysis/c1_third.py
対象: 2023年1月〜2026年11月, ばんえいを除く, 6頭以上 (trifecta_factors.load で読めるレース). 人気は確定人気.
脚質はレース前に分かるもの (近5走で最も多い脚質. ashi_1kyo.pre_style). 近走がない馬は「不明」.
(1) 1着=1番人気・2着=2or3番人気 のレース (C1が当たるレース) で, 3着馬の人気・脚質・前走着順の分布と,
    その特徴を持つ馬が「3着候補 (1〜3番人気以外)」のうち3着に来る率 (1頭あたり).
(2) 3着を絞った C1 の回収率: 3着を 人気4〜6 / 4〜8 / 前に行く馬 (逃げ・先行) / 人気4〜8かつ前 など.
出力: analysis/out/c1_third.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], str(Path(__file__).parent.parent / "data")]
import trifecta_factors as T  # noqa: E402
from ashi_1kyo import pre_style  # noqa: E402

ROOT = Path(__file__).parent
KEY = ["競馬場", "競走年月日", "レース番号"]


def main():
    h, _, p = T.load()
    h = h[h["競馬場"] != "帯広ば"]
    h = pre_style(h)
    h = T.features(h) if "前走着順" not in h.columns else h
    p = p.drop_duplicates(KEY).set_index(KEY)
    h["人気順"] = h.sort_values(["人気", "馬番"]).groupby(KEY).cumcount() + 1
    h = h[h["頭数"] >= 6]
    tri = p[["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]].dropna()
    tri.columns = ["a", "b", "c", "pay"]
    tri[["a", "b", "c"]] = tri[["a", "b", "c"]].astype(int)
    h = h.join(tri, on=KEY, how="inner")
    rk = h.set_index(KEY + ["馬番"])["人気順"]
    # レースごとに 1着・2着 の人気
    first = h[h["馬番"] == h["a"]][KEY + ["人気順"]].rename(columns={"人気順": "1着人気"})
    second = h[h["馬番"] == h["b"]][KEY + ["人気順"]].rename(columns={"人気順": "2着人気"})
    h = h.merge(first, on=KEY).merge(second, on=KEY)
    h["C1対象"] = (h["1着人気"] == 1) & h["2着人気"].isin([2, 3])
    h["3着"] = h["馬番"] == h["c"]
    h["前走着順c"] = pd.cut(pd.to_numeric(h["前走着順"], errors="coerce"), [0, 1, 3, 5, 9, 99], labels=["1着", "2〜3着", "4〜5着", "6〜9着", "10着〜"])
    h["人気帯"] = pd.cut(h["人気順"], [0, 3, 4, 5, 6, 8, 99], labels=["1〜3", "4", "5", "6", "7〜8", "9〜"])
    rows = []
    # (1) C1対象レースでの3着馬の特徴 (3着候補 = 1着・2着以外の馬)
    cand = h[h["C1対象"] & (h["馬番"] != h["a"]) & (h["馬番"] != h["b"])]
    n_r = cand.groupby(KEY).ngroups
    for col in ["人気帯", "事前脚質", "前走着順c"]:
        for v, g in cand.groupby(col, observed=True):
            rows.append({"分析": "(1)3着馬の特徴", "項目": col, "値": str(v), "3着の割合%": g["3着"].sum() / n_r * 100,
                         "候補1頭あたり3着率%": g["3着"].mean() * 100, "候補数": len(g)})
    # 人気帯×脚質 (人気4〜8)
    for (a, b), g in cand[cand["人気順"].between(4, 8)].groupby(["人気帯", "事前脚質"], observed=True):
        rows.append({"分析": "(1)3着馬の特徴", "項目": "人気帯×脚質", "値": f"{a}番人気・{b}", "3着の割合%": g["3着"].sum() / n_r * 100,
                     "候補1頭あたり3着率%": g["3着"].mean() * 100, "候補数": len(g)})
    # (2) 3着を絞った C1 の回収率 (全レース)
    races = h.groupby(KEY).agg(頭数=("馬番", "size"), pay=("pay", "first"), C1=("C1対象", "first")).reset_index()
    third = h[h["3着"]][KEY + ["人気順", "事前脚質"]].rename(columns={"人気順": "3着人気", "事前脚質": "3着脚質"})
    races = races.merge(third, on=KEY, how="left")
    # 点数: 2着候補2頭 × 3着候補 (2着になった方以外の2,3番人気も含む)
    st_by = h.set_index(KEY + ["人気順"])["事前脚質"]
    plans = {
        "C1 3着全部": lambda x: x["人気順"] >= 2,
        "3着 2〜6番人気": lambda x: x["人気順"].between(2, 6),
        "3着 2〜8番人気": lambda x: x["人気順"].between(2, 8),
        "3着 逃げ・先行 (+2,3番人気)": lambda x: (x["人気順"].isin([2, 3])) | x["事前脚質"].isin(["逃げ", "先行"]),
        "3着 2〜8番人気のうち逃げ・先行 (+2,3番人気)": lambda x: x["人気順"].isin([2, 3]) | (x["人気順"].between(4, 8) & x["事前脚質"].isin(["逃げ", "先行"])),
        "3着 差し・追込 (+2,3番人気)": lambda x: (x["人気順"].isin([2, 3])) | x["事前脚質"].isin(["差し", "追込"]),
        "3着 7番人気以下を除く前走1〜5着 (+2,3番人気)": lambda x: x["人気順"].isin([2, 3]) | (x["人気順"].between(4, 6) & x["前走着順c"].isin(["1着", "2〜3着", "4〜5着"])),
    }
    hh = h[h["人気順"] >= 2]
    for name, f in plans.items():
        sel = hh[f(hh)]
        n3 = sel.groupby(KEY).size().rename("n3")
        r = races.merge(n3, on=KEY, how="left").fillna({"n3": 0})
        # 2着候補 (2,3番人気) それぞれについて, 3着候補は自分以外
        pts = 2 * r["n3"] - 2  # 2,3番人気は両方とも3着候補に含まれる前提 (含まれない場合も2頭とも含む設計)
        hit_third = sel.set_index(KEY + ["馬番"]).index
        r = r.merge(h[h["3着"]][KEY + ["馬番"]], on=KEY, how="left")
        r["3着が候補"] = [ (k1, k2, k3, b) in hit_third for k1, k2, k3, b in zip(r["競馬場"], r["競走年月日"], r["レース番号"], r["馬番"])]
        hit = r["C1"] & r["3着が候補"]
        inv = pts.clip(lower=0) * 100
        rows.append({"分析": "(2)3着を絞ったC1", "項目": name, "値": "", "レース": len(r), "平均点数": pts.mean(),
                     "的中率%": hit.mean() * 100, "回収率%": (r["pay"] * hit).sum() / inv.sum() * 100,
                     "前半(2023-24)回収率%": (r["pay"] * hit)[r["競走年月日"] < 20250101].sum() / inv[r["競走年月日"] < 20250101].sum() * 100,
                     "後半(2025-26)回収率%": (r["pay"] * hit)[r["競走年月日"] >= 20250101].sum() / inv[r["競走年月日"] >= 20250101].sum() * 100})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.max_columns", 20)
    print("C1が当たるレース (1着=1番人気, 2着=2or3番人気):", n_r, "/ 全", races.shape[0])
    print(t.round(1).to_string(index=False))
    t.round(2).to_csv(ROOT / "out" / "c1_third.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
