"""騎手の補正: 人気帯 × 競馬場ごとに, 騎手が人気から見込まれるより勝つか (A/E) を求め, その再現性を確かめる.

使い方: python3 analysis/jockey_adjust.py <data>
  A/E = 実際の勝ち数 ÷ 期待勝ち数. 期待勝ち数は「同じ競馬場・年・人気 (10番人気以下はまとめる)・頭数帯」の平均勝率の合計.
  件数の少ない騎手は 1 に寄せる: 縮小A/E = (勝ち数 + M) / (期待勝ち数 + M), M = 5.
  (1) 再現性: 2021〜2024年の縮小A/E で騎手を分け, 2025〜2026年の成績 (A/E・単勝回収率) が同じ向きかを見る.
  (2) 予想用の表: 2021年〜最新の全期間で, 騎手 × 競馬場 × 人気帯 の縮小A/E (analysis/out/jk_table.csv).
  (3) 1番人気の騎手で分けた, オッズの形の買い目 (odds_bets.py) の回収率: 2026年8月までの成績で騎手を分け, 9〜10月の確定オッズで検証.
出力: analysis/out/jk_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import odds_bets as OB  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
HORSE = ["2021_horselist_全月まとめ.csv", "2022_horselist_全月まとめ.csv", "2023_horselist_全月まとめ.csv",
         "2024_horselist_全月まとめ.csv", "2025_horselist_全月まとめ.csv", "2026_horselist_1-8月まとめ.csv",
         "202609_horselist.csv", "202610_horselist.csv"]
PAY = ["pay2021.csv", "pay2022.csv", "pay2023.csv", "pay2024.csv", "pay2025.csv", "pay2026.csv",
       "202609_payback.csv", "202610_payback.csv"]
BANDS = ["1番人気", "2〜3番人気", "4〜6番人気", "7番人気以下"]
M = 5.0


def band(p):
    return np.select([p == 1, p <= 3, p <= 6], BANDS[:3], BANDS[3])


def load():
    cols = KEY + ["馬番", "騎手名", "人気", "着順"]
    h = pd.concat([pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, usecols=cols)
                   for f in HORSE if (DATA / f).exists()], ignore_index=True)
    for c in ["人気", "着順", "馬番"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h[h["人気"].notna() & h["着順"].notna()].drop_duplicates(KEY + ["馬番"])
    h["頭数"] = h.groupby(KEY)["馬番"].transform("size")
    h = h[h["頭数"] >= 5]
    h["騎手名"] = h["騎手名"].astype(str).str.replace(r"[\s　]", "", regex=True)
    h["年"] = h["競走年月日"] // 10000
    h["勝"] = (h["着順"] == 1).astype(float)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99]).astype(str)
    h["期待勝"] = h.groupby(["競馬場", "年", "人気c", "頭数帯"])["勝"].transform("mean")
    h["人気帯"] = band(h["人気"])
    # 単勝の払戻 (回収率用)
    pay = pd.concat([pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["単勝組番", "単勝払戻金（円）"])
                     for f in PAY if (DATA / f).exists()], ignore_index=True)
    pay["馬番"] = pd.to_numeric(pay["単勝組番"], errors="coerce")
    pay = pay.dropna(subset=["馬番"]).drop_duplicates(KEY + ["馬番"])
    h = h.merge(pay[KEY + ["馬番", "単勝払戻金（円）"]], on=KEY + ["馬番"], how="left")
    h["単勝払戻"] = h["単勝払戻金（円）"].fillna(0) * h["勝"]
    return h


def ae_table(h):
    g = h.groupby(["騎手名", "競馬場", "人気帯"]).agg(騎乗=("勝", "size"), 勝=("勝", "sum"), 期待勝=("期待勝", "sum"),
                                                 払戻=("単勝払戻", "sum"))
    g["A/E"] = g["勝"] / g["期待勝"]
    g["縮小A/E"] = (g["勝"] + M) / (g["期待勝"] + M)
    g["勝率%"] = g["勝"] / g["騎乗"] * 100
    g["期待勝率%"] = g["期待勝"] / g["騎乗"] * 100
    g["単勝回収率%"] = g["払戻"] / g["騎乗"]
    return g


def ae_bin(x):
    return pd.cut(x, [0, 0.85, 0.95, 1.05, 1.15, 99], labels=["~0.85", "0.85~0.95", "0.95~1.05", "1.05~1.15", "1.15~"])


def main():
    h = load()
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print(f"対象: {h['競走年月日'].min()}〜{h['競走年月日'].max()}, {h.groupby(KEY).ngroups}レース, {len(h)}騎乗")

    # ---- (1) 再現性: 2021〜2024 → 2025〜2026
    tr, te = h[h["年"] <= 2024], h[h["年"] >= 2025]
    a = ae_table(tr)[["騎乗", "縮小A/E"]].rename(columns={"騎乗": "過去騎乗", "縮小A/E": "過去A/E"})
    x = te.merge(a, left_on=["騎手名", "競馬場", "人気帯"], right_index=True, how="left")
    x = x[x["過去騎乗"] >= 30]
    x["過去A/E帯"] = ae_bin(x["過去A/E"])
    r = x.groupby(["人気帯", "過去A/E帯"], observed=True).agg(
        騎手数=("騎手名", "nunique"), 騎乗=("勝", "size"), 勝=("勝", "sum"), 期待勝=("期待勝", "sum"), 払戻=("単勝払戻", "sum"))
    r["2025〜26のA/E"] = r["勝"] / r["期待勝"]
    r["2025〜26の勝率%"] = r["勝"] / r["騎乗"] * 100
    r["2025〜26の期待勝率%"] = r["期待勝"] / r["騎乗"] * 100
    r["2025〜26の単勝回収率%"] = r["払戻"] / r["騎乗"]
    r = r.reindex(BANDS, level=0)
    print("\n== (1) 2021〜24年の騎手A/E (競馬場×人気帯ごと, 30騎乗以上) で分けた, 2025〜26年の成績")
    print(r.drop(columns=["勝", "期待勝", "払戻"]).round(2).to_string())
    r.round(3).to_csv(OUT / "jk_validation.csv", encoding="utf-8-sig")

    # 競馬場をまたいだ同じ騎手の一貫性: 騎手×人気帯 (競馬場をまとめる) でも同じ検証
    a2 = tr.groupby(["騎手名", "人気帯"]).agg(勝=("勝", "sum"), 期待勝=("期待勝", "sum"), 騎乗=("勝", "size"))
    a2["過去A/E"] = (a2["勝"] + M) / (a2["期待勝"] + M)
    x2 = te.merge(a2[["過去A/E", "騎乗"]].rename(columns={"騎乗": "過去騎乗"}), left_on=["騎手名", "人気帯"], right_index=True)
    x2 = x2[x2["過去騎乗"] >= 30]
    x2["過去A/E帯"] = ae_bin(x2["過去A/E"])
    r2 = x2.groupby(["人気帯", "過去A/E帯"], observed=True).agg(騎乗=("勝", "size"), 勝=("勝", "sum"), 期待勝=("期待勝", "sum"),
                                                          払戻=("単勝払戻", "sum"))
    r2["2025〜26のA/E"] = r2["勝"] / r2["期待勝"]
    r2["2025〜26の単勝回収率%"] = r2["払戻"] / r2["騎乗"]
    print("\n== (参考) 競馬場をまとめた騎手A/E で分けた場合\n", r2.reindex(BANDS, level=0).drop(columns=["勝", "期待勝", "払戻"]).round(2).to_string())

    # 競馬場ごとの「上位騎手の勝ち方」: 1番人気での騎手A/Eのばらつき (場ごとの騎手の差の大きさ)
    full = ae_table(h)
    full.round(3).to_csv(OUT / "jk_table.csv", encoding="utf-8-sig")
    f1 = full.xs("1番人気", level="人気帯")
    f1 = f1[f1["騎乗"] >= 50].reset_index()
    rows = []
    for c, g in f1.groupby("競馬場"):
        g = g.sort_values("縮小A/E")
        rows.append({"競馬場": c, "騎手数(1番人気50回以上)": len(g),
                     "1番人気で勝つ騎手 (縮小A/E上位3)": " ".join(f"{r.騎手名}{r._8:.2f}" for r in g.tail(3)[::-1].itertuples()),
                     "1番人気で勝ち切れない騎手 (下位3)": " ".join(f"{r.騎手名}{r._8:.2f}" for r in g.head(3).itertuples())})
    t = pd.DataFrame(rows)
    print("\n== 1番人気での騎手の縮小A/E (2021年〜最新, 1番人気50回以上の騎手)\n", t.to_string(index=False))
    t.to_csv(OUT / "jk_fav_by_track.csv", index=False, encoding="utf-8-sig")

    # ---- (3) 1番人気の騎手で分けた, オッズの形の買い目の回収率 (9〜10月)
    files = sorted((DATA / "odds").glob("2026*_odds.csv"))
    if not files:
        return
    od = OB.read_odds(files)
    past = ae_table(h[h["競走年月日"] < 20260901])
    pay = pd.concat([pd.read_csv(DATA / f, encoding="utf-8-sig") for f in ["202609_payback.csv", "202610_payback.csv"]
                     if (DATA / f).exists()]).drop_duplicates(KEY).set_index(KEY)
    jk = h.set_index(KEY + ["馬番"])["騎手名"]
    rows = []
    for k, g in od.groupby(KEY):
        if k not in pay.index or len(g) < 5:
            continue
        p = pay.loc[k]
        if p[["３連単組番馬番1", "馬単組番1"]].isna().any():
            continue
        g = g.sort_values(["単勝オッズ", "馬番"])
        nos, odds = g["馬番"].astype(int).tolist(), g["単勝オッズ"].tolist()
        b = OB.bets(nos, odds)
        n1, n2, n3, n4, n5 = nos[:5]
        tri = {6: {(n1, a_, c) for a_ in (n2, n3, n4) for c in (n2, n3, n4) if a_ != c},
               2: {(n1, n2, n3), (n1, n3, n2)}}.get(b["点数3"], set())
        r3 = (int(p["３連単組番馬番1"]), int(p["３連単組番馬番2"]), int(p["３連単組番馬番3"]))
        r2 = (int(p["馬単組番1"]), int(p["馬単組番2"]))
        j = jk.get(k + (n1,))
        ae = past["縮小A/E"].get((j, k[0], "1番人気"), np.nan) if j is not None else np.nan
        nride = past["騎乗"].get((j, k[0], "1番人気"), 0) if j is not None else 0
        rows.append({"1番人気の騎手A/E": ae if nride >= 30 else np.nan, "1番人気勝": r2[0] == n1,
                     "３連単点数": len(tri), "３連単払戻": p["３連単払戻金（円）"] if r3 in tri else 0,
                     "馬単払戻": p["馬単払戻金（円）"] if (r2[0] == n1 and r2[1] in (n2, n3, n4, n5)) else 0})
    d = pd.DataFrame(rows)
    d["騎手A/E帯"] = pd.cut(d["1番人気の騎手A/E"], [0, 0.95, 1.05, 99], labels=["~0.95 (勝ち切れない)", "0.95~1.05", "1.05~ (よく勝つ)"])
    d["騎手A/E帯"] = d["騎手A/E帯"].cat.add_categories("騎乗30回未満").fillna("騎乗30回未満")
    s = d.groupby("騎手A/E帯", observed=True).agg(
        レース数=("馬単払戻", "size"), 一番人気の勝率=("1番人気勝", "mean"),
        三連単点数=("３連単点数", "sum"), 三連単払戻=("３連単払戻", "sum"), 馬単払戻=("馬単払戻", "sum"))
    s["1番人気の勝率%"] = s.pop("一番人気の勝率") * 100
    s["３連単 回収率%"] = s["三連単払戻"] / (s["三連単点数"] * 100) * 100
    s["馬単1→2〜5 回収率%"] = s["馬単払戻"] / (s["レース数"] * 400) * 100
    print("\n== (3) 1番人気の騎手の過去A/E (2026年8月まで, 同じ競馬場の1番人気) で分けた, 9〜10月のオッズの形の買い目")
    print(s.drop(columns=["三連単払戻", "馬単払戻"]).round(1).to_string())
    s.round(2).to_csv(OUT / "jk_odds_bets.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
