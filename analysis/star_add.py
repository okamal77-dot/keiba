"""人気順の３連単 (odds_bets.py) の2・3着に, 能力モデルの★ (過小評価候補) を加えると良くなるかの検証.

使い方: python3 analysis/star_add.py <data>
  先に全場で koryu_backtest.py を実行しておく (analysis/out/koryu_rows_*.csv).
★の判定: 想定人気は過去の印に残っていないので, 確定の単勝人気で判定する (第17節の「実際の人気で判定」).
  モデル勝率順が5位以内 かつ 人気が勝率順より3つ以上低い (差2も参考に出す) かつ 消し条件でない. 1番人気は除く.
比べる買い方 (人気番号, ★は2・3着の欄に足す):
  2点      : 1→2,3→2,3
  2点+★    : 1→2,3,★→2,3,★
  2点+○▲  : 1→2,3,○▲→2,3,○▲ (★の代わりに印の2・3番手. 参考)
  6点      : 1→2,3,4→2,3,4
  6点+★    : 1→2,3,4,★→2,3,4,★
  馬単4点   : 1→2〜5 /  馬単4点+★: 1→2〜5,★
(1) 9〜10月の確定オッズで「1強」のレース, (2) 同じく全形, (3) 2025年1月〜2026年10月の全レース (人気順のみ)
出力: analysis/out/sa_*.csv
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
BETS = ["2点", "2点+★", "2点+○▲", "6点", "6点+★", "馬単4点", "馬単4点+★"]


def tri(fav, mates):
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
    shapes = {}
    for k, g in od.groupby(KEY):
        g = g.sort_values(["単勝オッズ", "馬番"])
        if len(g) >= 5:
            o = g["単勝オッズ"].tolist()
            shapes[k] = (OB.shape_of(o)[0], o[0])
    out = []
    for k, x in rows.groupby(KEY):
        if k not in pay.index or len(x) < 5 or x["人気"].isna().any():
            continue
        x = x.sort_values(["人気", "馬番"])
        nos = x["馬番"].astype(int).tolist()
        fav, p2, p3, p4, p5 = nos[:5]
        wrank = (-x["勝率"]).rank(method="first").astype(int)
        pop = x["人気"].astype(int)
        kesi = x["消し"].astype(bool)
        res = {}
        for gap in (3, 2):
            star = x[(wrank <= 5) & (pop >= wrank + gap) & ~kesi & (x["馬番"] != fav)]["馬番"].astype(int).tolist()
            res[gap] = star
        marks = x.sort_values("印順")["馬番"].astype(int).tolist()
        om = [n for n in marks[:3] if n != fav][:2]          # 印の上位 (◎が1番人気でなければ◎も入る) から2頭
        pr = pay.loc[k]
        r3 = tuple(int(pr[c]) for c in c3[:3])
        r2 = (int(pr[c2[0]]), int(pr[c2[1]]))
        sh, o1 = shapes.get(k, (None, None))
        for gap, star in res.items():
            sets3 = {"2点": tri(fav, [p2, p3]), "2点+★": tri(fav, list(dict.fromkeys([p2, p3] + star))),
                     "2点+○▲": tri(fav, list(dict.fromkeys([p2, p3] + om))),
                     "6点": tri(fav, [p2, p3, p4]), "6点+★": tri(fav, list(dict.fromkeys([p2, p3, p4] + star)))}
            sets2 = {"馬単4点": {(fav, n) for n in (p2, p3, p4, p5)},
                     "馬単4点+★": {(fav, n) for n in dict.fromkeys([p2, p3, p4, p5] + star)}}
            row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "★の差": gap, "★の数": len(star),
                   "★が人気2〜5外": len([s for s in star if s not in (p2, p3, p4, p5)]), "形": sh, "1人気オッズ": o1}
            for name, s in sets3.items():
                row[name + "_点"], row[name + "_払"] = len(s), (pr[c3[3]] if r3 in s else 0)
            for name, s in sets2.items():
                row[name + "_点"], row[name + "_払"] = len(s), (pr[c2[2]] if r2 in s else 0)
            out.append(row)
    d = pd.DataFrame(out)
    d.to_csv(T.OUT / "sa_races.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 40)

    def table(g):
        t = {}
        for b in BETS:
            pts, ret = g[b + "_点"].sum(), g[b + "_払"].sum()
            t[b] = {"点数/レース": pts / len(g), "的中%": (g[b + "_払"] > 0).mean() * 100, "回収率%": ret / max(pts * 100, 1) * 100}
        # 足した分だけの回収率 (★を足して増えた点数と払戻)
        for base, plus in [("2点", "2点+★"), ("6点", "6点+★"), ("馬単4点", "馬単4点+★"), ("2点", "2点+○▲")]:
            dp = g[plus + "_点"].sum() - g[base + "_点"].sum()
            dr = g[plus + "_払"].sum() - g[base + "_払"].sum()
            t[f"{plus}の足した分"] = {"点数/レース": dp / len(g), "的中%": ((g[plus + "_払"] > 0) & (g[base + "_払"] == 0)).mean() * 100,
                                  "回収率%": dr / max(dp * 100, 1) * 100}
        return pd.DataFrame(t).T

    def ci_added(g, base, plus, n=2000):
        dp = (g[plus + "_点"] - g[base + "_点"]).values * 100.0
        dr = (g[plus + "_払"] - g[base + "_払"]).values
        idx = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
        r = dr[idx].sum(1) / np.maximum(dp[idx].sum(1), 1) * 100
        return np.percentile(r, [5, 95])

    res = []
    for gap in (3, 2):
        dg = d[d["★の差"] == gap]
        cases = [("9〜10月・1強", dg[dg["形"] == "1強"]), ("9〜10月・全形", dg[dg["形"].notna()]),
                 ("2025年1月〜・全レース", dg)]
        for label, g in cases:
            withstar = g[g["★が人気2〜5外"] > 0]
            t = table(g)
            print(f"\n== ★=人気が勝率順より{gap}つ以上低い / {label} ({len(g)}レース, うち★が2〜5番人気の外にいる {len(withstar)}レース)")
            print(t.round(1).to_string())
            for base, plus in [("2点", "2点+★"), ("6点", "6点+★"), ("馬単4点", "馬単4点+★")]:
                lo, hi = ci_added(g, base, plus)
                print(f"  {plus}の足した分 回収率の90%区間: {lo:.0f}〜{hi:.0f}%")
            t.insert(0, "区分", f"差{gap}・{label}")
            res.append(t)
        if gap == 3:
            g = dg.assign(期=np.where(dg["競走年月日"] < 20250701, "2025前半", np.where(dg["競走年月日"] < 20260101, "2025後半",
                                                                            np.where(dg["競走年月日"] < 20260701, "2026前半", "2026後半"))))
            per = g.groupby("期").apply(lambda x: pd.Series({
                f"{p}の足した分": (x[p + "_払"].sum() - x[b + "_払"].sum()) / max((x[p + "_点"].sum() - x[b + "_点"].sum()) * 100, 1) * 100
                for b, p in [("2点", "2点+★"), ("6点", "6点+★"), ("馬単4点", "馬単4点+★")]} | {
                f"{b}": x[b + "_払"].sum() / (x[b + "_点"].sum() * 100) * 100 for b in ["2点", "6点", "馬単4点"]}))
            print("\n== 差3・全レース: 半年ごとの回収率%\n", per.round(1).to_string())
            per.round(2).to_csv(T.OUT / "sa_by_period.csv", encoding="utf-8-sig")
    pd.concat(res).round(2).to_csv(T.OUT / "sa_summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
