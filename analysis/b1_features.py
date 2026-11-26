"""３連単 1番人気→4,5,6番人気→4,5,6番人気 (6点, tri_flow.py の B1) が当たりやすいレースの特徴.

使い方: python3 analysis/b1_features.py
対象: 確定オッズ (data/odds) のある 2026年3月・6〜11月, ばんえいを除く, 6頭以上.
人気は確定単勝オッズの順 (同じオッズは馬番の小さい方を先).
特徴: オッズの形 (odds_bets.shape_of), 1番人気のオッズ, 4番人気・6番人気のオッズ, 10倍以下の頭数,
      4〜6番人気と2〜3番人気の差 (4番人気オッズ ÷ 3番人気オッズ), 頭数, クラス, 競馬場, 馬場, 距離.
各帯で 的中率・回収率 (6点×100円) と90%区間 (ブートストラップ). 前半 (3・6〜8月) と後半 (9〜11月) で向きが同じかも見る.
出力: analysis/out/b1_features.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import odds_bets as OB
from race_profile import race_class

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
    rf = [DATA / "race2026.csv"] + [DATA / f"2026{m:02d}_racelist.csv" for m in (9, 10, 11)]
    rl = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["レース名", "馬場", "距離"]) for f in rf if f.exists()])
    rl = rl.drop_duplicates(KEY, keep="last").set_index(KEY)
    rows = []
    for k, g in od.groupby(KEY):
        if k not in pay.index or len(g) < 6:
            continue
        g = g.sort_values(["o", "番号1"])
        nos, oo = g["番号1"].astype(int).tolist(), g["o"].tolist()
        pr = pay.loc[k]
        r3 = [int(pr[f"３連単組番馬番{i}"]) for i in (1, 2, 3)]
        hit = r3[0] == nos[0] and r3[1] in nos[3:6] and r3[2] in nos[3:6]
        shape, _ = OB.shape_of(oo)
        name = rl["レース名"].get(k) if k in rl.index else None
        rows.append({"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "形": shape, "頭数": len(g),
                     "1番人気オッズ": oo[0], "3番人気オッズ": oo[2], "4番人気オッズ": oo[3], "6番人気オッズ": oo[5],
                     "10倍以下": int(sum(o <= 10 for o in oo)), "クラス": race_class(name) if isinstance(name, str) else None,
                     "馬場": str(rl["馬場"].get(k)).strip() if k in rl.index else None,
                     "距離": pd.to_numeric(rl["距離"].get(k), errors="coerce") if k in rl.index else np.nan,
                     "1着が1番人気": r3[0] == nos[0], "的中": hit, "払": pr["３連単払戻金（円）"] if hit else 0})
    d = pd.DataFrame(rows)
    d["期"] = np.where(d["競走年月日"] < 20260901, "前半(3,6-8月)", "後半(9-11月)")
    d["1番人気帯"] = pd.cut(d["1番人気オッズ"], [0, 1.5, 2.0, 2.5, 3.0, 4.0, 99], labels=["〜1.5", "1.6〜2.0", "2.1〜2.5", "2.6〜3.0", "3.1〜4.0", "4.1〜"])
    d["4番人気帯"] = pd.cut(d["4番人気オッズ"], [0, 8, 12, 16, 25, 999], labels=["〜8", "8〜12", "12〜16", "16〜25", "25〜"])
    d["6番人気帯"] = pd.cut(d["6番人気オッズ"], [0, 15, 25, 40, 9999], labels=["〜15", "15〜25", "25〜40", "40〜"])
    d["4人気÷3人気"] = pd.cut(d["4番人気オッズ"] / d["3番人気オッズ"], [0, 1.3, 1.7, 2.2, 3.0, 99], labels=["〜1.3", "1.3〜1.7", "1.7〜2.2", "2.2〜3.0", "3.0〜"])
    d["頭数帯"] = pd.cut(d["頭数"], [0, 8, 10, 12, 20], labels=["6〜8", "9〜10", "11〜12", "13〜"])
    d["10倍以下帯"] = d["10倍以下"].clip(1, 5).map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5以上"})
    d["距離帯"] = pd.cut(d["距離"], [0, 1000, 1400, 1600, 1800, 9999], labels=["〜1000", "1001〜1400", "1401〜1600", "1601〜1800", "1801〜"])
    rng = np.random.default_rng(0)

    def summ(x):
        n = len(x)
        ret = x["払"].to_numpy()
        res = {"レース": n, "的中": int(x["的中"].sum()), "的中率%": x["的中"].mean() * 100, "回収率%": ret.sum() / (n * 600) * 100,
               "1番人気の勝率%": x["1着が1番人気"].mean() * 100}
        if n >= 50:
            idx = rng.integers(0, n, (1000, n))
            rr = ret[idx].sum(1) / (n * 600) * 100
            res["90%区間"] = "%.0f〜%.0f" % tuple(np.percentile(rr, [5, 95]))
        for p, gp in x.groupby("期"):
            res[p] = gp["払"].sum() / (len(gp) * 600) * 100
        return res

    out = [{"特徴": "全体", "値": "全体", **summ(d)}]
    for col in ["形", "1番人気帯", "4番人気帯", "6番人気帯", "4人気÷3人気", "10倍以下帯", "頭数帯", "クラス", "馬場", "距離帯", "競馬場"]:
        for v, g in d.groupby(col, observed=True):
            if len(g) >= 100:
                out.append({"特徴": col, "値": str(v), **summ(g)})
    t = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print("対象:", len(d), "レース")
    print(t.round(1).to_string(index=False))
    # 組み合わせ: 形 × 1番人気帯
    print("\n== 形 × 1番人気帯 (回収率%, レース数)")
    for (s, b), g in d.groupby(["形", "1番人気帯"], observed=True):
        if len(g) >= 150:
            r = summ(g)
            print(f"{s} {b}: {r['レース']}R 的中{r['的中率%']:.1f}% 回収{r['回収率%']:.0f}% ({r.get('90%区間', '')}) 前半{r.get('前半(3,6-8月)', np.nan):.0f} 後半{r.get('後半(9-11月)', np.nan):.0f}")
    t.round(2).to_csv(ROOT / "out" / "b1_features.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
