"""月ごとの回収率は馬場 (良・稍重・重・不良) で変わるか.
使い方: python3 analysis/baba_roi.py   (先に contrib.py を実行して out/contrib_races.csv を作る)
ルール各100円 (全部) と 1,000円配分 (配分), 土台の 市場だけ を, 馬場・月・オッズの形 で分ける. 90%区間はレース単位のブートストラップ.
出力: analysis/out/baba_roi.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
KEY = ["競馬場", "競走年月日", "レース番号"]
BABA = ["良", "稍重", "重", "不良"]


def main():
    d = pd.read_csv(ROOT / "out" / "contrib_races.csv", encoding="utf-8-sig")
    rf = [DATA / "race2026.csv"] + [DATA / f"2026{m:02d}_racelist.csv" for m in (9, 10, 11)] + sorted((DATA / "entries").glob("*_racelist.csv"))
    rl = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["馬場", "天候"]) for f in rf if f.exists()])
    d = d.merge(rl.drop_duplicates(KEY, keep="last"), on=KEY, how="left")
    d["馬場"] = d["馬場"].astype(str).str.strip()
    d = d[d["馬場"].isin(BABA)]
    d["月"] = d["競走年月日"] // 100 % 100
    rng = np.random.default_rng(0)

    def roi(x, v):
        x = x.dropna(subset=[v + "_投"])
        return x[v + "_払"].sum() / max(x[v + "_投"].sum(), 1) * 100

    def ci(x, v):
        x = x.dropna(subset=[v + "_投"])
        i, p = x[v + "_投"].to_numpy(), x[v + "_払"].to_numpy()
        idx = rng.integers(0, len(x), (2000, len(x)))
        return "%.0f〜%.0f" % tuple(np.percentile(p[idx].sum(1) / np.maximum(i[idx].sum(1), 1) * 100, [5, 95]))

    def row(x):
        return {"レース": len(x), "市場だけ%": roi(x, "市場だけ"), "ルール%": roi(x, "全部"), "配分%": roi(x, "配分"),
                "配分の90%区間": ci(x, "配分"), "1番人気勝率%": np.nan}

    rows = []
    print("対象:", len(d), "レース")
    print("\n== 馬場ごと")
    for b in BABA:
        rows.append({"区分": "馬場", "馬場": b, **row(d[d["馬場"] == b])})
    d["重い"] = np.where(d["馬場"].isin(["重", "不良"]), "重・不良", "良・稍重")
    for b in ["良・稍重", "重・不良"]:
        rows.append({"区分": "馬場2", "馬場": b, **row(d[d["重い"] == b])})
    print("\n== 月ごと (重・不良の割合)")
    for m, g in d.groupby("月"):
        rows.append({"区分": "月", "月": m, "重・不良の割合%": (g["重い"] == "重・不良").mean() * 100, **row(g)})
        for b in ["良・稍重", "重・不良"]:
            rows.append({"区分": "月×馬場", "月": m, "馬場": b, **row(g[g["重い"] == b])})
    for s in ["1強", "2強", "3強", "4強"]:
        for b in ["良・稍重", "重・不良"]:
            rows.append({"区分": "形×馬場", "形": s, "馬場": b, **row(d[(d["形"] == s) & (d["重い"] == b)])})
    t = pd.DataFrame(rows).drop(columns=["1番人気勝率%"])
    cols = ["区分", "月", "形", "馬場", "重・不良の割合%", "レース", "市場だけ%", "ルール%", "配分%", "配分の90%区間"]
    t = t.reindex(columns=cols)
    pd.set_option("display.width", 220, "display.max_rows", 100)
    print(t.round(0).to_string(index=False))
    # 月の差のうち, 馬場の割合で説明できる分: 馬場ごとの全期間の配分回収率 × その月の馬場の割合
    base = {b: roi(d[d["重い"] == b], "配分") for b in ["良・稍重", "重・不良"]}
    print("\n== 馬場の割合だけで見込む月の配分回収率 と 実際")
    for m, g in d.dropna(subset=["配分_投"]).groupby("月"):
        w = g.groupby("重い")["配分_投"].sum() / g["配分_投"].sum()
        exp = sum(w.get(b, 0) * base[b] for b in base)
        print(f"{m}月: 見込み {exp:.0f}% / 実際 {roi(g, '配分'):.0f}%")
    t.round(1).to_csv(ROOT / "out" / "baba_roi.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
