"""単勝オッズ5倍以下の頭数・10倍以下の頭数で, レースの決まり方と買い方の回収率がどう変わるか.
使い方: python3 analysis/odds_count.py   (先に contrib.py を実行して out/contrib_races.csv を作る)
確定オッズ (data/odds) の単勝を使う. 対象は contrib.py と同じレース (ばんえいを除く, 5頭以上, 2026年3月・6〜11月).
見るもの: 1番人気の勝率, 勝ち馬が5倍以下/10倍以下の馬だった割合, ３連単の平均人気と1万円以上の割合,
  市場だけ (毎回 ３連単6点+馬単4点) / ルール各100円 / 1,000円配分 の回収率 (配分は90%区間つき).
出力: analysis/out/odds_count.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
KEY = ["競馬場", "競走年月日", "レース番号"]


def main():
    d = pd.read_csv(ROOT / "out" / "contrib_races.csv", encoding="utf-8-sig")
    fs = sorted((DATA / "odds").glob("2026*_odds.csv"), key=lambda f: "final" in f.name)
    od = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["賭式", "番号1", "オッズ"]).query("賭式 == '単勝'") for f in fs])
    od = od.drop_duplicates(KEY + ["番号1"], keep="last")
    od["o"] = pd.to_numeric(od["オッズ"], errors="coerce")
    od = od[od["o"] > 0]
    c = od.groupby(KEY).agg(頭数=("o", "size"), 五倍以下=("o", lambda x: (x <= 5).sum()), 十倍以下=("o", lambda x: (x <= 10).sum())).reset_index()
    pf = [DATA / "pay2026.csv"] + [DATA / f"2026{m:02d}_payback.csv" for m in (9, 10, 11)]
    pay = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in pf if f.exists()]).drop_duplicates(KEY, keep="last")
    pay = pay[KEY + ["単勝組番", "３連単払戻金（円）", "３連単人気"]]
    d = d.merge(c, on=KEY, how="left").merge(pay, on=KEY, how="left")
    win = od.rename(columns={"番号1": "単勝組番", "o": "勝ち馬オッズ"})[KEY + ["単勝組番", "勝ち馬オッズ"]]
    d["単勝組番"] = pd.to_numeric(d["単勝組番"], errors="coerce")
    d = d.merge(win, on=KEY + ["単勝組番"], how="left")
    rng = np.random.default_rng(0)

    def roi(x, v):
        x = x.dropna(subset=[v + "_投"])
        return x[v + "_払"].sum() / max(x[v + "_投"].sum(), 1) * 100

    def ci(x, v):
        x = x.dropna(subset=[v + "_投"])
        if len(x) < 20:
            return ""
        i, p = x[v + "_投"].to_numpy(), x[v + "_払"].to_numpy()
        idx = rng.integers(0, len(x), (2000, len(x)))
        return "%.0f〜%.0f" % tuple(np.percentile(p[idx].sum(1) / np.maximum(i[idx].sum(1), 1) * 100, [5, 95]))

    def row(x):
        return {"レース": len(x), "割合%": len(x) / len(d) * 100,
                "1番人気勝率%": (x["勝ち馬オッズ"] <= x["1番人気オッズ"] + 1e-9).mean() * 100,
                "勝ち馬5倍以下%": (x["勝ち馬オッズ"] <= 5).mean() * 100, "勝ち馬10倍以下%": (x["勝ち馬オッズ"] <= 10).mean() * 100,
                "３連単平均人気": x["３連単人気"].mean(), "３連単中央値": x["３連単払戻金（円）"].median(),
                "３連単1万円以上%": (x["３連単払戻金（円）"] >= 10000).mean() * 100,
                "市場だけ%": roi(x, "市場だけ"), "ルール%": roi(x, "全部"), "配分%": roi(x, "配分"), "配分90%区間": ci(x, "配分")}

    d["5倍以下の頭数"] = d["五倍以下"].clip(upper=4).astype(int).astype(str).replace({"4": "4頭以上"})
    d["10倍以下の頭数"] = d["十倍以下"].clip(lower=2, upper=7).astype(int).astype(str).replace({"2": "2頭以下", "7": "7頭以上"})
    rows = []
    for col in ["5倍以下の頭数", "10倍以下の頭数"]:
        for k, g in d.groupby(col):
            rows.append({"分け方": col, "値": k, **row(g)})
    # 組み合わせ: 5倍以下 × 10倍以下
    d["組"] = d["五倍以下"].clip(upper=3).astype(str) + "頭/" + d["十倍以下"].clip(upper=6).astype(str) + "頭"
    for k, g in d.groupby("組"):
        if len(g) >= 150:
            rows.append({"分け方": "5倍以下/10倍以下", "値": k, **row(g)})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_rows", 100, "display.max_columns", 20)
    print("対象:", len(d), "レース")
    print(t.round(1).to_string(index=False))
    # 形との関係
    print("\n== 形 × 5倍以下の頭数 (レース数)\n", pd.crosstab(d["形"], d["5倍以下の頭数"]).to_string())
    # 月ごとの平均頭数
    d["月"] = d["競走年月日"] // 100 % 100
    print("\n== 月ごとの平均 (5倍以下/10倍以下の頭数) と配分回収率")
    for m, g in d.groupby("月"):
        print(m, round(g["五倍以下"].mean(), 2), round(g["十倍以下"].mean(), 2), round(roi(g, "配分")))
    t.round(2).to_csv(ROOT / "out" / "odds_count.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
