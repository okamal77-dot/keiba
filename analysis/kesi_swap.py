"""人気順で組む買い目 (odds_bets.py) で, 能力モデルの「消し条件」の馬を入れ替えると回収率が上がるかの検証.

使い方: python3 analysis/kesi_swap.py <data>
  先に全場で koryu_backtest.py を実行しておく (analysis/out/koryu_rows_*.csv: 2025年1月〜2026年10月,
  各期間より前のデータだけで学習したモデルの 印順・消し).
入れ替え方:
  そのまま   : 人気順のまま (今のルール)
  次の人気   : 消し条件の馬を飛ばし, 次の人気の馬で埋める
  印の上位   : 消し条件の馬の代わりに, まだ入っていない馬のうち印の順 (◎○▲△…) が最も上の馬を入れる
(1) 9〜10月 (確定オッズあり): オッズの形のルールどおり (3強・4強=３連単6点, 1強=2点, 馬単=1→2〜5)
(2) 2025年1月〜2026年10月 (人気順のみ): ３連単 1→2,3,4→2,3,4 / 1→2,3→2,3 / 馬単 1→2〜5 を全レースで
1番人気が消し条件のレースは, 入れ替えずに数だけ数える.
出力: analysis/out/ks_*.csv
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import odds_bets as OB  # noqa: E402
import trifecta_factors as T  # noqa: E402

KEY = T.KEY
METHODS = ["そのまま", "次の人気", "印の上位"]


def pick(order_pop, kesi, mark_rank, k):
    """人気2位以下から k 頭を選ぶ. order_pop: 人気順の馬番, kesi: 馬番→bool, mark_rank: 馬番→印順"""
    out = {}
    cand = order_pop[1:]
    out["そのまま"] = cand[:k]
    out["次の人気"] = [n for n in cand if not kesi.get(n, False)][:k]
    base = cand[:k]
    keep = [n for n in base if not kesi.get(n, False)]
    rest = sorted([n for n in cand if n not in base and not kesi.get(n, False)], key=lambda n: mark_rank.get(n, 99))
    out["印の上位"] = keep + rest[:k - len(keep)]
    return out


def tri_set(fav, mates):
    return {(fav, a, b) for a in mates for b in mates if a != b}


def main():
    rows = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "koryu_rows_*.csv"))])
    rows = rows.drop_duplicates(KEY + ["馬番"])
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    c3 = ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]
    c2 = ["馬単組番1", "馬単組番2", "馬単払戻金（円）"]
    pay = pay.dropna(subset=c3 + c2).set_index(KEY)
    files = sorted((T.DATA / "odds").glob("2026*_odds.csv"))
    od = OB.read_odds(files) if files else pd.DataFrame(columns=KEY + ["馬番", "単勝オッズ"])
    odds = {k: g.sort_values(["単勝オッズ", "馬番"]) for k, g in od.groupby(KEY)}
    out = []
    for k, x in rows.groupby(KEY):
        if k not in pay.index or len(x) < 5 or x["人気"].isna().any():
            continue
        pr = pay.loc[k]
        r3 = tuple(int(pr[c]) for c in c3[:3])
        r2 = (int(pr[c2[0]]), int(pr[c2[1]]))
        kesi = dict(zip(x["馬番"].astype(int), x["消し"].astype(bool)))
        mark = dict(zip(x["馬番"].astype(int), x["印順"]))
        base = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "1番人気が消し": False}
        for src in ["人気", "オッズ"]:
            if src == "人気":
                order = x.sort_values(["人気", "馬番"])["馬番"].astype(int).tolist()
                shape, tri_k = "全レース", None
            else:
                if k not in odds:
                    continue
                g = odds[k]
                order = g["馬番"].astype(int).tolist()
                o = g["単勝オッズ"].tolist()
                if len(order) < 5:
                    continue
                shape, _ = OB.shape_of(o)
                tri_k = 0 if (o[0] >= 3.0 or shape == "2強") else (3 if shape in ("3強", "4強") else 2)
            fav = order[0]
            row = dict(base, 区分=src, 形=shape, **{"1番人気が消し": kesi.get(fav, False)})
            p3 = pick(order, kesi, mark, 3)
            p2 = pick(order, kesi, mark, 2)
            p4 = pick(order, kesi, mark, 4)
            row["入れ替えあり"] = p3["そのまま"] != p3["次の人気"] or p4["そのまま"] != p4["次の人気"]
            for m in METHODS:
                if src == "人気":
                    sets = {"３連単6点": tri_set(fav, p3[m]), "３連単2点": tri_set(fav, p2[m])}
                else:
                    mates = {3: p3[m], 2: p2[m]}.get(tri_k, [])
                    sets = {"３連単ルール": tri_set(fav, mates) if tri_k else set()}
                for name, s in sets.items():
                    row[f"{name}_{m}_点"] = len(s)
                    row[f"{name}_{m}_払"] = pr[c3[3]] if r3 in s else 0
                ex = {(fav, n) for n in p4[m]}
                row[f"馬単4点_{m}_点"] = len(ex)
                row[f"馬単4点_{m}_払"] = pr[c2[2]] if r2 in ex else 0
            out.append(row)
    d = pd.DataFrame(out)
    d.to_csv(T.OUT / "ks_races.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 40)

    def summary(g, bets):
        res = {"レース数": len(g)}
        for b in bets:
            for m in METHODS:
                pts, ret = g[f"{b}_{m}_点"].sum(), g[f"{b}_{m}_払"].sum()
                res[f"{b} {m} 的中%"] = (g[f"{b}_{m}_払"] > 0).mean() * 100
                res[f"{b} {m} 回収率%"] = ret / max(pts * 100, 1) * 100
        return pd.Series(res)

    def ci_diff(g, b, m, n=2000):
        """入れ替えと「そのまま」の回収率の差の90%区間 (レース単位の復元抽出)"""
        a_r, a_p = g[f"{b}_{m}_払"].values, g[f"{b}_{m}_点"].values * 100
        b_r, b_p = g[f"{b}_そのまま_払"].values, g[f"{b}_そのまま_点"].values * 100
        idx = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
        diff = a_r[idx].sum(1) / np.maximum(a_p[idx].sum(1), 1) - b_r[idx].sum(1) / np.maximum(b_p[idx].sum(1), 1)
        return np.percentile(diff * 100, [5, 95])

    res = []
    for src, bets in [("人気", ["３連単6点", "３連単2点", "馬単4点"]), ("オッズ", ["３連単ルール", "馬単4点"])]:
        x = d[(d["区分"] == src) & ~d["1番人気が消し"]]
        for label, g in [("全レース", x), ("消し条件の馬が買い目にかかったレース", x[x["入れ替えあり"]])]:
            s = summary(g, bets)
            print(f"\n== {src}順 / {label} ({len(g)}レース)")
            tab = pd.DataFrame({m: {b: f"{s[f'{b} {m} 回収率%']:.1f}% (的中{s[f'{b} {m} 的中%']:.1f}%)" for b in bets} for m in METHODS})
            for m in METHODS[1:]:
                tab[f"{m}−そのまま 90%区間"] = [" 〜 ".join(f"{v:+.1f}" for v in ci_diff(g, b, m)) for b in bets]
            print(tab.to_string())
            res.append(s.rename((src, label)))
        if src == "人気":
            x = x.assign(年半=np.where(x["競走年月日"] < 20250701, "2025前半", np.where(x["競走年月日"] < 20260101, "2025後半",
                                                                           np.where(x["競走年月日"] < 20260701, "2026前半", "2026後半"))))
            y = x[x["入れ替えあり"]]
            per = y.groupby("年半").apply(lambda g: pd.Series({f"{b} {m}": g[f"{b}_{m}_払"].sum() / max(g[f"{b}_{m}_点"].sum() * 100, 1) * 100
                                                           for b in ["３連単6点", "馬単4点"] for m in METHODS}))
            print("\n== 人気順 / 消し条件の馬が買い目にかかったレース: 半年ごとの回収率%\n", per.round(1).to_string())
            per.round(2).to_csv(T.OUT / "ks_by_period.csv", encoding="utf-8-sig")
    pd.DataFrame(res).round(2).to_csv(T.OUT / "ks_summary.csv", encoding="utf-8-sig")
    print(f"\n1番人気が消し条件のレース: 人気順 {int(d[(d['区分'] == '人気')]['1番人気が消し'].sum())} / "
          f"オッズ {int(d[(d['区分'] == 'オッズ')]['1番人気が消し'].sum())} (入れ替えの対象外)")


if __name__ == "__main__":
    main()
