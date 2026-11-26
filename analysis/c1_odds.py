"""３連単 1番人気→2,3番人気→全部 (C1) と 3着を2〜8番人気に絞った形 (C1-8) を, 単勝オッズのばらつき具合で分ける.

使い方: python3 analysis/c1_odds.py
対象: 確定オッズのある 2026年3月・6〜11月, ばんえいを除く, 6頭以上 (b1_features.py と同じ). 人気は確定単勝オッズの順.
特徴: オッズの形, 1番人気のオッズ, 2番人気のオッズ, 2番人気÷1番人気, 3番人気÷2番人気, 4番人気÷3番人気,
      10倍以下の頭数, 上位3頭の支持の集中度 (1/オッズを合計1にした確率の上位3頭の和), 支持のばらつき (確率の標準偏差の代わりに 実効頭数=1/Σp²).
回収率は各100円 (C1: 2×(頭数-2) 点, C1-8: 2×(2〜8番人気の頭数-1) 点). 前半 (3・6〜8月) と後半 (9〜11月) も出す.
出力: analysis/out/c1_odds.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import odds_bets as OB

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
KEY = ["競馬場", "競走年月日", "レース番号"]


def main():
    fs = sorted((DATA / "odds").glob("2026*_odds.csv"), key=lambda f: "final" in f.name)
    od = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["賭式", "番号1", "オッズ"]).query("賭式 == '単勝'")
                    for f in fs]).drop_duplicates(KEY + ["番号1"], keep="last")
    od["o"] = pd.to_numeric(od["オッズ"], errors="coerce")
    od = od[(od["o"] > 0) & (od["競馬場"] != "帯広ば")]
    pf = [DATA / "pay2026.csv"] + [DATA / f"2026{m:02d}_payback.csv" for m in (9, 10, 11)]
    pay = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in pf if f.exists()]).drop_duplicates(KEY, keep="last")
    pay = pay.dropna(subset=["３連単組番馬番1"]).set_index(KEY)
    rows = []
    for k, g in od.groupby(KEY):
        if k not in pay.index or len(g) < 6:
            continue
        g = g.sort_values(["o", "番号1"])
        nos, oo = g["番号1"].astype(int).tolist(), g["o"].tolist()
        n = len(nos)
        pr = pay.loc[k]
        r3 = [int(pr[f"３連単組番馬番{i}"]) for i in (1, 2, 3)]
        c1 = r3[0] == nos[0] and r3[1] in nos[1:3]
        c8 = c1 and r3[2] in nos[1:8]
        q = 1 / np.array(oo)
        p = q / q.sum()
        rows.append({"競馬場": k[0], "競走年月日": k[1], "形": OB.shape_of(oo)[0], "頭数": n,
                     "1人気": oo[0], "2人気": oo[1], "2÷1": oo[1] / oo[0], "3÷2": oo[2] / oo[1], "4÷3": oo[3] / oo[2],
                     "10倍以下": int(sum(o <= 10 for o in oo)), "上位3頭の支持": p[:3].sum(), "実効頭数": 1 / (p ** 2).sum(),
                     "C1的中": c1, "C1払": pr["３連単払戻金（円）"] if c1 else 0, "C1点": 2 * (n - 2),
                     "C8的中": c8, "C8払": pr["３連単払戻金（円）"] if c8 else 0, "C8点": 2 * (min(n, 8) - 2)})
    d = pd.DataFrame(rows)
    d["期"] = np.where(d["競走年月日"] < 20260901, "前半", "後半")
    bands = {
        "形": None,
        "1人気": [0, 1.5, 2.0, 2.5, 3.0, 4.0, 99],
        "2人気": [0, 3, 4, 5, 6, 8, 999],
        "2÷1": [0, 1.3, 1.7, 2.2, 3.0, 5.0, 99],
        "3÷2": [0, 1.1, 1.25, 1.5, 2.0, 99],
        "4÷3": [0, 1.3, 1.7, 2.2, 3.0, 99],
        "10倍以下": [0, 2, 3, 4, 5, 99],
        "上位3頭の支持": [0, 0.5, 0.6, 0.7, 0.8, 1.01],
        "実効頭数": [0, 3, 4, 5, 6, 8, 99],
    }
    rng = np.random.default_rng(0)

    def summ(x):
        r = {"レース": len(x)}
        for c in ["C1", "C8"]:
            inv, ret = x[c + "点"].to_numpy() * 100, x[c + "払"].to_numpy()
            r[c + "的中率%"] = x[c + "的中"].mean() * 100
            r[c + "回収率%"] = ret.sum() / inv.sum() * 100
            for pp, gp in x.groupby("期"):
                r[f"{c}回収率%({pp})"] = gp[c + "払"].sum() / (gp[c + "点"].sum() * 100) * 100
        if len(x) >= 60:
            inv, ret = x["C8点"].to_numpy() * 100, x["C8払"].to_numpy()
            idx = rng.integers(0, len(x), (1000, len(x)))
            r["C8の90%区間"] = "%.0f〜%.0f" % tuple(np.percentile(ret[idx].sum(1) / inv[idx].sum(1) * 100, [5, 95]))
        return r

    out = [{"特徴": "全体", "値": "全体", **summ(d)}]
    for col, b in bands.items():
        key = d[col] if b is None else pd.cut(d[col], b, right=True)
        for v, g in d.groupby(key, observed=True):
            if len(g) >= 100:
                out.append({"特徴": col, "値": str(v), **summ(g)})
    t = pd.DataFrame(out)
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.max_columns", 20)
    print("対象:", len(d), "レース")
    print(t.round(1).to_string(index=False))
    t.round(2).to_csv(ROOT / "out" / "c1_odds.csv", index=False, encoding="utf-8-sig")
    d.to_csv(ROOT / "out" / "c1_odds_races.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
