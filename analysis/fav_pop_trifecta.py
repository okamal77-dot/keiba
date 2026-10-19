"""◎ (能力モデルの本命) の人気別に, ３連単の回収率を比べる.

使い方: python3 analysis/fav_pop_trifecta.py <data>
  先に koryu_backtest.py <data> <競馬場> を全場分実行しておく (analysis/out/koryu_rows_*.csv を使う).
  koryu_rows は 2025年1月〜2026年10月を半年ごとに区切り, その期間より前のデータだけで学習したモデルの印.
  人気は確定時の単勝人気 (締切後に分かる値) で分ける.
買い方 (印順: ◎=1, ○=2, ▲=3, △=4,5):
  2頭軸18点      : ◎○ → ◎○▲△ → ◎○▲△△ (predict_race.suggest の ◎勝率25〜40% の形)
  ◎1着固定12点   : ◎ → ○▲△△ → ○▲△△
  ◎2・3着付け24点: ○▲△△ → ◎ → ○▲△△ と ○▲△△ → ○▲△△ → ◎
  ◎軸マルチ36点  : ◎ と ○▲△△ のうち2頭で, 3頭の並びはすべて
出力: analysis/out/fpt_summary.csv, fpt_by_period.csv, fpt_by_track.csv
"""
import glob
import itertools

import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY
PERIODS = ["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01", "2027-01-01"]
PLABEL = ["2025前半", "2025後半", "2026前半", "2026後半"]


def tickets(m):
    """m: 印順1〜5の馬番 (足りない場合は短い). 買い方ごとの組のリスト."""
    if len(m) < 5:
        return None
    f, rest = m[0], m[1:5]
    perm = lambda A, B, C: [(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3]  # noqa: E731
    return {
        "2頭軸18点": perm(m[:2], m[:4], m[:5]),
        "◎1着固定12点": perm([f], rest, rest),
        "◎2・3着付け24点": perm(rest, [f], rest) + perm(rest, rest, [f]),
        "◎軸マルチ36点": [c for pair in itertools.combinations(rest, 2) for c in itertools.permutations((f,) + pair)],
    }


def boot_ci(ret, cost, n=2000, seed=0):
    """レース単位で復元抽出した回収率の90%区間"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ret), (n, len(ret)))
    r = ret[idx].sum(1) / cost[idx].sum(1) * 100
    return np.percentile(r, [5, 95])


def main():
    rows = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "koryu_rows_*.csv"))])
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    cols = ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）", "単勝組番", "単勝払戻金（円）"]
    pay = pay.dropna(subset=cols[:4]).set_index(KEY)[cols]
    out = []
    for k, x in rows.groupby(KEY):
        if k not in pay.index or x["着順"].notna().sum() < 3:
            continue
        x = x.sort_values("印順")
        m = x["馬番"].astype(int).tolist()
        tk = tickets(m)
        if tk is None:
            continue
        pr = pay.loc[k]
        res = (int(pr.iloc[0]), int(pr.iloc[1]), int(pr.iloc[2]))
        fav = x.iloc[0]
        row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "◎人気": fav["人気"], "◎勝率": fav["勝率"],
               "◎着順": fav["着順"], "頭数": len(x), "３連単配当": pr["３連単払戻金（円）"],
               "単勝◎_点数": 1, "単勝◎_払戻": pr["単勝払戻金（円）"] if str(int(fav["馬番"])) == str(pr["単勝組番"]).split(".")[0] else 0}
        for name, combos in tk.items():
            row[name + "_点数"] = len(combos)
            row[name + "_払戻"] = pr["３連単払戻金（円）"] if res in combos else 0
        out.append(row)
    d = pd.DataFrame(out)
    d["日付"] = pd.to_datetime(d["競走年月日"].astype(str))
    d["期間"] = pd.cut(d["日付"], pd.to_datetime(PERIODS), labels=PLABEL, right=False)
    d["◎人気帯"] = d["◎人気"].map(lambda p: f"{int(p)}番人気" if p <= 5 else "6番人気以下")
    d["ルール"] = np.select([d["◎勝率"] >= 0.40, d["◎勝率"] >= 0.25], ["本命(単勝)", "2頭軸"], "混戦(3連複)")
    bets = ["2頭軸18点", "◎1着固定12点", "◎2・3着付け24点", "◎軸マルチ36点", "単勝◎"]
    order = ["1番人気", "2番人気", "3番人気", "4番人気", "5番人気", "6番人気以下"]

    def summ(g, ci=True):
        s = {"レース数": len(g), "◎勝率%": (g["◎着順"] == 1).mean() * 100, "◎3着内%": (g["◎着順"] <= 3).mean() * 100,
             "３連単配当中央値": g["３連単配当"].median()}
        for b in bets:
            ret, cost = g[b + "_払戻"].values, g[b + "_点数"].values * 100
            s[b + " 的中%"] = (ret > 0).mean() * 100
            s[b + " 回収率%"] = ret.sum() / cost.sum() * 100
            if ci and b != "単勝◎" and len(g) >= 30:
                lo, hi = boot_ci(ret, cost)
                s[b + " 90%区間"] = f"{lo:.0f}〜{hi:.0f}"
        return pd.Series(s)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    t = d.groupby("◎人気帯").apply(summ).reindex(order)
    t.loc["全体"] = summ(d)
    print(f"== ◎の人気別 ３連単の回収率 ({d['日付'].min():%Y-%m-%d}〜{d['日付'].max():%Y-%m-%d}, {len(d)}レース, 14場)")
    print(t.round(1).T.to_string())
    t.round(2).to_csv(T.OUT / "fpt_summary.csv", encoding="utf-8-sig")

    # 期間ごと (安定性)
    tp = d.groupby(["◎人気帯", "期間"], observed=True).apply(lambda g: summ(g, ci=False)[
        ["レース数"] + [b + " 回収率%" for b in bets]]).reset_index()
    tp["◎人気帯"] = pd.Categorical(tp["◎人気帯"], order)
    tp = tp.sort_values(["◎人気帯", "期間"])
    print("\n== 半年ごとの回収率%\n", tp.round(1).to_string(index=False))
    tp.round(2).to_csv(T.OUT / "fpt_by_period.csv", index=False, encoding="utf-8-sig")

    # ルールの買い目区分 × ◎人気 (実際にルールで３連単を買うのは 2頭軸 のレース)
    tr = d.groupby(["ルール", "◎人気帯"]).apply(lambda g: summ(g, ci=False)[
        ["レース数", "◎勝率%", "2頭軸18点 回収率%", "◎1着固定12点 回収率%", "◎2・3着付け24点 回収率%", "単勝◎ 回収率%"]]).reset_index()
    tr["◎人気帯"] = pd.Categorical(tr["◎人気帯"], order)
    tr = tr.sort_values(["ルール", "◎人気帯"])
    print("\n== ルールの区分 × ◎人気\n", tr.round(1).to_string(index=False))
    tr.round(2).to_csv(T.OUT / "fpt_by_rule.csv", index=False, encoding="utf-8-sig")

    # 競馬場別 (◎1〜3番人気の主な買い方)
    tt = d.groupby(["競馬場", "◎人気帯"]).apply(lambda g: summ(g, ci=False)[
        ["レース数", "2頭軸18点 回収率%", "◎1着固定12点 回収率%", "◎2・3着付け24点 回収率%"]]).reset_index()
    tt.round(2).to_csv(T.OUT / "fpt_by_track.csv", index=False, encoding="utf-8-sig")
    d.to_csv(T.OUT / "fpt_races.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
