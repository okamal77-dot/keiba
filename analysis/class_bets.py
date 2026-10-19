"""クラス (A/B/C級・2歳・3歳・重賞) によって, 人気順の買い目 (第8章) の成績が変わるかを確かめる.

使い方: python3 analysis/class_bets.py <data>
  (1) 2021年〜2026年10月の全レース (人気順): ３連単 1→2,3→2,3 / 1→2,3,4→2,3,4, 馬単 1→2〜5 のクラス別回収率と, 3期間 (2021-22, 2023-24, 2025-26) の安定性
  (2) 2026年9〜10月 (確定オッズ): オッズの形のルール (odds_bets.py) のクラス別回収率
  (3) ★を足した分 (star_add.py の sa_races.csv がある場合): クラス別
クラスの判定は class_transfer.race_class と同じ (レース名と競走種類名称から).
出力: analysis/out/cb_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import odds_bets as OB  # noqa: E402
from class_transfer import race_class  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
YEARS = [("2021_horselist_全月まとめ.csv", "race2021.csv", "pay2021.csv"),
         ("2022_horselist_全月まとめ.csv", "race2022.csv", "pay2022.csv"),
         ("2023_horselist_全月まとめ.csv", "race2023.csv", "pay2023.csv"),
         ("2024_horselist_全月まとめ.csv", "race2024.csv", "pay2024.csv"),
         ("2025_horselist_全月まとめ.csv", "race2025.csv", "pay2025.csv"),
         ("2026_horselist_1-8月まとめ.csv", "race2026.csv", "pay2026.csv"),
         ("202609_horselist.csv", "202609_racelist.csv", "202609_payback.csv"),
         ("202610_horselist.csv", "202610_racelist.csv", "202610_payback.csv")]
ORDER = ["2歳", "3歳", "C", "B", "A", "重賞", "その他"]
C3 = ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]
C2 = ["馬単組番1", "馬単組番2", "馬単払戻金（円）"]


def read(f, **kw):
    return pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, **kw)


def load():
    hs, rs, ps = [], [], []
    for hf, rf, pf in YEARS:
        if not all((DATA / f).exists() for f in (hf, rf, pf)):
            continue
        hs.append(read(hf, usecols=KEY + ["馬番", "人気", "着順"]))
        rs.append(read(rf, usecols=KEY + ["レース名", "競走種類名称"]))
        ps.append(read(pf, usecols=KEY + C3 + C2))
    h = pd.concat(hs, ignore_index=True)
    for c in ["馬番", "人気", "着順"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h[h["人気"].notna() & h["着順"].notna()].drop_duplicates(KEY + ["馬番"])
    r = pd.concat(rs).drop_duplicates(KEY).set_index(KEY)
    p = pd.concat(ps).drop_duplicates(KEY).dropna().set_index(KEY)
    return h, r, p


def tri(fav, mates):
    return {(fav, a, b) for a in mates for b in mates if a != b}


def roi(g, b):
    return g[b + "_払"].sum() / max(g[b + "_点"].sum() * 100, 1) * 100


def main():
    h, r, p = load()
    files = sorted((DATA / "odds").glob("2026*_odds.csv"))
    od = OB.read_odds(files) if files else pd.DataFrame(columns=KEY + ["馬番", "単勝オッズ"])
    odds = {k: g.sort_values(["単勝オッズ", "馬番"]) for k, g in od.groupby(KEY)}
    rows = []
    for k, x in h.groupby(KEY):
        if k not in p.index or k not in r.index or len(x) < 5:
            continue
        x = x.sort_values(["人気", "馬番"])
        n = x["馬番"].astype(int).tolist()
        pr = p.loc[k]
        r3 = tuple(int(pr[c]) for c in C3[:3])
        r2 = (int(pr[C2[0]]), int(pr[C2[1]]))
        cls = race_class(r.loc[k, "レース名"], r.loc[k, "競走種類名称"])
        row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "クラス": cls, "頭数": len(x)}
        for name, s in [("３連単2点", tri(n[0], n[1:3])), ("３連単6点", tri(n[0], n[1:4]))]:
            row[name + "_点"], row[name + "_払"] = len(s), (pr[C3[3]] if r3 in s else 0)
        s2 = {(n[0], m) for m in n[1:5]}
        row["馬単4点_点"], row["馬単4点_払"] = 4, (pr[C2[2]] if r2 in s2 else 0)
        if k in odds and len(odds[k]) >= 5:      # オッズの形のルール
            g = odds[k]
            on, oo = g["馬番"].astype(int).tolist(), g["単勝オッズ"].tolist()
            b = OB.bets(on, oo)
            row["形"] = b["形"]
            s = {6: tri(on[0], on[1:4]), 2: tri(on[0], on[1:3])}.get(b["点数3"], set())
            row["ルール３連単_点"], row["ルール３連単_払"] = len(s), (pr[C3[3]] if r3 in s else 0)
            s2 = {(on[0], m) for m in on[1:5]}
            row["ルール馬単_点"], row["ルール馬単_払"] = 4, (pr[C2[2]] if r2 in s2 else 0)
        rows.append(row)
    d = pd.DataFrame(rows)
    d["期"] = np.select([d["競走年月日"] < 20230101, d["競走年月日"] < 20250101], ["2021-22", "2023-24"], "2025-26")
    d.to_csv(OUT / "cb_races.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 40)

    bets = ["３連単2点", "３連単6点", "馬単4点"]
    t = []
    for cls, g in d.groupby("クラス"):
        row = {"クラス": cls, "レース数": len(g), "平均頭数": g["頭数"].mean()}
        for b in bets:
            row[f"{b} 的中%"] = (g[b + "_払"] > 0).mean() * 100
            row[f"{b} 回収率%"] = roi(g, b)
            for per, gg in g.groupby("期"):
                row[f"{b} {per}"] = roi(gg, b)
        t.append(row)
    t = pd.DataFrame(t).set_index("クラス").reindex(ORDER).dropna(how="all")
    allrow = {"レース数": len(d), "平均頭数": d["頭数"].mean()}
    for b in bets:
        allrow[f"{b} 的中%"] = (d[b + "_払"] > 0).mean() * 100
        allrow[f"{b} 回収率%"] = roi(d, b)
        for per, gg in d.groupby("期"):
            allrow[f"{b} {per}"] = roi(gg, b)
    t.loc["全体"] = allrow
    print(f"== (1) 人気順の買い目のクラス別回収率 ({d['競走年月日'].min()}〜{d['競走年月日'].max()})")
    print(t.round(1).T.to_string())
    t.round(2).to_csv(OUT / "cb_class.csv", encoding="utf-8-sig")

    # 頭数を揃えた比較 (クラスごとに頭数の分布が違うため): 頭数帯 × クラス の ３連単6点・馬単
    d["頭数帯"] = pd.cut(d["頭数"], [0, 8, 10, 12, 99], labels=["~8頭", "9-10頭", "11-12頭", "13頭~"])
    t2 = d.pivot_table(index="クラス", columns="頭数帯", values="３連単6点_払", aggfunc="sum", observed=True) / \
        d.pivot_table(index="クラス", columns="頭数帯", values="３連単6点_点", aggfunc="sum", observed=True) / 100 * 100
    print("\n== ３連単6点 の回収率% (クラス × 頭数帯)\n", t2.reindex(ORDER).dropna(how="all").round(1).to_string())
    t2.round(2).to_csv(OUT / "cb_class_field.csv", encoding="utf-8-sig")

    # (2) オッズの形のルール (9〜10月)
    o = d[d["形"].notna()]
    if len(o):
        t3 = o.groupby("クラス").apply(lambda g: pd.Series({
            "レース数": len(g), "３連単を買う率%": (g["ルール３連単_点"] > 0).mean() * 100,
            "ルール３連単 回収率%": roi(g, "ルール３連単"), "ルール３連単 的中(買ったレース)%": (g["ルール３連単_払"] > 0).sum() / max((g["ルール３連単_点"] > 0).sum(), 1) * 100,
            "ルール馬単 回収率%": roi(g, "ルール馬単"), "ルール馬単 的中%": (g["ルール馬単_払"] > 0).mean() * 100}))
        t3 = t3.reindex(ORDER).dropna(how="all")
        t3.loc["全体"] = [len(o), (o["ルール３連単_点"] > 0).mean() * 100, roi(o, "ルール３連単"),
                        (o["ルール３連単_払"] > 0).sum() / (o["ルール３連単_点"] > 0).sum() * 100, roi(o, "ルール馬単"),
                        (o["ルール馬単_払"] > 0).mean() * 100]
        print("\n== (2) オッズの形のルール (2026年9〜10月) のクラス別\n", t3.round(1).to_string())
        t3.round(2).to_csv(OUT / "cb_rule_class.csv", encoding="utf-8-sig")
        t4 = o.pivot_table(index="クラス", columns="形", values="ルール３連単_払", aggfunc="sum") / \
            o.pivot_table(index="クラス", columns="形", values="ルール３連単_点", aggfunc="sum").replace(0, np.nan) / 100 * 100
        n4 = o.pivot_table(index="クラス", columns="形", values="ルール３連単_点", aggfunc=lambda s: (s > 0).sum())
        print("\n== ルール３連単 の回収率% (クラス × 形, ( )は買ったレース数)")
        print(t4.reindex(ORDER).dropna(how="all").round(0).astype("Int64").astype(str).radd("").add(
            " (" + n4.reindex(ORDER).dropna(how="all").astype(int).astype(str) + ")").to_string())

    # (3) ★を足した分 (差3)
    sa = OUT / "sa_races.csv"
    if sa.exists():
        s = pd.read_csv(sa, encoding="utf-8-sig")
        s = s[s["★の差"] == 3].merge(d[KEY + ["クラス"]], on=KEY, how="left")
        rows = []
        for cls, g in s.groupby("クラス"):
            row = {"クラス": cls, "レース数": len(g)}
            for base, plus in [("2点", "2点+★"), ("6点", "6点+★"), ("馬単4点", "馬単4点+★")]:
                dp = g[plus + "_点"].sum() - g[base + "_点"].sum()
                row[f"{base} 元"] = g[base + "_払"].sum() / (g[base + "_点"].sum() * 100) * 100
                row[f"{plus} 足した分"] = (g[plus + "_払"].sum() - g[base + "_払"].sum()) / max(dp * 100, 1) * 100
            rows.append(row)
        t5 = pd.DataFrame(rows).set_index("クラス").reindex(ORDER).dropna(how="all")
        print("\n== (3) ★ (差3) を足した分の回収率% (2025年1月〜2026年10月)\n", t5.round(1).to_string())
        t5.round(2).to_csv(OUT / "cb_star_class.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
