"""1番人気が 1着・2着・3着・4着以下 になるレースで, 1番人気の脚質とオッズに特徴があるか.

使い方: python3 analysis/fav_finish.py
脚質: レース前に分かるもの (近5走で最も多い脚質. ashi_1kyo.pre_style).
(1) 脚質: 2023年1月〜2026年11月 (確定人気の1番人気), ばんえいを除く, 6頭以上.
(2) オッズ・脚質×オッズ: 確定オッズのある 2026年3月・6〜11月 (確定単勝オッズの1番人気).
各区分で 1着率・2着率・3着率・4着以下率 と, 「1着だったときに比べて2着・3着が多いか」(2・3着の合計に対する2着の割合).
出力: analysis/out/fav_finish.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], str(Path(__file__).parent.parent / "data")]
import trifecta_factors as T  # noqa: E402
from ashi_1kyo import pre_style  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
KEY = ["競馬場", "競走年月日", "レース番号"]


def rates(x, col="着順"):
    f = x[col]
    return {"レース": len(x), "1着%": (f == 1).mean() * 100, "2着%": (f == 2).mean() * 100, "3着%": (f == 3).mean() * 100,
            "4着以下%": (f >= 4).mean() * 100, "3着内%": (f <= 3).mean() * 100,
            "2,3着のうち2着%": (f == 2).sum() / max(((f == 2) | (f == 3)).sum(), 1) * 100}


def main():
    h, _, _ = T.load()
    h = h[(h["競馬場"] != "帯広ば") & (h["頭数"] >= 6)]
    h = pre_style(h)
    fav = h.sort_values(["人気", "馬番"]).groupby(KEY).head(1)
    fav = fav[fav["人気"] == 1]
    rows = []
    for v, g in fav.groupby("事前脚質"):
        rows.append({"分析": "(1)脚質 2023〜26年", "区分": v, **rates(g)})
    for (c, v), g in fav.groupby(["競馬場", "事前脚質"]):
        if len(g) >= 150:
            rows.append({"分析": "(1)競馬場×脚質", "区分": f"{c}・{v}", **rates(g)})
    # (2) オッズ (確定単勝オッズ)
    fs = sorted((DATA / "odds").glob("2026*_odds.csv"), key=lambda f: "final" in f.name)
    od = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["賭式", "番号1", "オッズ"]).query("賭式 == '単勝'")
                    for f in fs]).drop_duplicates(KEY + ["番号1"], keep="last")
    od["o"] = pd.to_numeric(od["オッズ"], errors="coerce")
    od = od[od["o"] > 0].sort_values(KEY + ["o", "番号1"])
    top = od.groupby(KEY).head(2).copy()
    top["順"] = top.groupby(KEY).cumcount()
    f1 = top[top["順"] == 0][KEY + ["番号1", "o"]].rename(columns={"番号1": "馬番", "o": "1人気オッズ"})
    f2 = top[top["順"] == 1][KEY + ["o"]].rename(columns={"o": "2人気オッズ"})
    st = h[KEY + ["馬番", "着順", "事前脚質"]]
    o = f1.merge(f2, on=KEY).merge(st, on=KEY + ["馬番"], how="inner")
    o["オッズ帯"] = pd.cut(o["1人気オッズ"], [0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 99],
                       labels=["〜1.3", "1.4〜1.5", "1.6〜1.8", "1.9〜2.0", "2.1〜2.5", "2.6〜3.0", "3.1〜4.0", "4.1〜"])
    o["2番人気との差"] = pd.cut(o["2人気オッズ"] / o["1人気オッズ"], [0, 1.2, 1.5, 2.0, 3.0, 99],
                           labels=["〜1.2倍", "1.2〜1.5倍", "1.5〜2倍", "2〜3倍", "3倍〜"])
    for v, g in o.groupby("オッズ帯", observed=True):
        rows.append({"分析": "(2)オッズ 2026年", "区分": str(v), **rates(g)})
    for v, g in o.groupby("2番人気との差", observed=True):
        rows.append({"分析": "(2)2番人気オッズ÷1番人気", "区分": str(v), **rates(g)})
    o["オッズ3帯"] = pd.cut(o["1人気オッズ"], [0, 1.5, 2.5, 99], labels=["1.5倍以下", "1.6〜2.5倍", "2.6倍以上"])
    for (b, v), g in o.groupby(["オッズ3帯", "事前脚質"], observed=True):
        if len(g) >= 80:
            rows.append({"分析": "(2)オッズ×脚質", "区分": f"{b}・{v}", **rates(g)})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_rows", 300)
    print("1番人気 (2023〜26年):", len(fav), "レース / オッズのある2026年:", len(o), "レース")
    print(t.round(1).to_string(index=False))
    t.round(2).to_csv(ROOT / "out" / "fav_finish.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
