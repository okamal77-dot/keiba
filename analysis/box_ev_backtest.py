"""混戦レース (３連複5頭BOX) に 印以外で期待値1.2〜2.0の馬を足す買い方の検証.

使い方: python3 analysis/box_ev_backtest.py <data>
予想: koryu_rows_*.csv (半年ごとに それ以前のデータだけで学習したモデル) の勝率を win_calibrate.py で補正.
期待値 = 補正後の勝率 × 確定単勝オッズ (data/odds/2026*_odds.csv がある 2026年9月〜10月).
混戦 = 補正後の◎の勝率 30%未満. 消しの馬は足さない. 1点100円.
出力: analysis/out/box_ev_races.csv
"""
import glob

import numpy as np
import pandas as pd

import trifecta_factors as T
import win_calibrate as WC

KEY = T.KEY


def main():
    r = pd.concat([pd.read_csv(f) for f in glob.glob(str(T.OUT / "koryu_rows_*.csv"))])
    r = r[r["競走年月日"] >= 20260901].copy()
    o = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob("data/odds/2026*_odds.csv")])
    o = o[o["賭式"] == "単勝"][["競馬場", "競走年月日", "レース番号", "番号1", "オッズ"]].rename(columns={"番号1": "馬番"})
    o["馬番"] = o["馬番"].astype(int)
    r = r.merge(o, on=KEY + ["馬番"], how="inner")
    r["rid"] = r["競馬場"] + r["競走年月日"].astype(str) + "_" + r["レース番号"].astype(str)
    wm = WC.load_map()
    r["補正勝率"] = WC.apply_df(r, wm)
    r["期待値"] = r["補正勝率"] * r["オッズ"]
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY).set_index(KEY)
    res = []
    for k, g in r.groupby(KEY):
        if k not in pay.index or g["着順"].isna().all():
            continue
        g = g.sort_values("印順")
        if g["補正勝率"].iloc[0] >= WC.MID:
            continue
        p = pay.loc[k]
        try:
            t3 = {int(p["３連複組番馬番1"]), int(p["３連複組番馬番2"]), int(p["３連複組番馬番3"])}
            win_no = int(p["単勝組番"])
        except (ValueError, TypeError):
            continue
        fp, wp = float(p["３連複払戻金（円）"]), float(p["単勝払戻金（円）"])
        five = g["馬番"].astype(int).tolist()[:5]
        rest = g.iloc[5:]
        ev = rest[(rest["期待値"] >= 1.2) & (rest["期待値"] <= 2.0) & ~rest["消し"].astype(bool)].sort_values("期待値", ascending=False)
        add = ev["馬番"].astype(int).tolist()
        fin = dict(zip(g["馬番"].astype(int), g["着順"]))
        row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "期待値馬の数": len(add),
               "5頭BOX_c": 1000, "5頭BOX_r": fp * t3.issubset(five)}
        box6 = five + add[:1]
        n6 = len(box6)
        row["+期待値1頭BOX_c"] = 100 * n6 * (n6 - 1) * (n6 - 2) // 6
        row["+期待値1頭BOX_r"] = fp * t3.issubset(box6)
        box7 = five + add[:2]
        n7 = len(box7)
        row["+期待値2頭まで_c"] = 100 * n7 * (n7 - 1) * (n7 - 2) // 6
        row["+期待値2頭まで_r"] = fp * t3.issubset(box7)
        # ☆の2頭目 (5番手) を期待値の馬と入れ替える (10点のまま)
        swap = five[:4] + add[:1] if add else five
        row["5番手と入替_c"] = 1000
        row["5番手と入替_r"] = fp * t3.issubset(swap)
        row["期待値馬の単勝_c"] = 100 * len(add[:1])
        row["期待値馬の単勝_r"] = wp * (len(add) > 0 and win_no == add[0])
        row["期待値馬の3着内"] = sum(fin.get(a, 99) <= 3 for a in add[:1])
        res.append(row)
    x = pd.DataFrame(res)
    x.to_csv(T.OUT / "box_ev_races.csv", index=False, encoding="utf-8-sig")
    names = ["5頭BOX", "+期待値1頭BOX", "+期待値2頭まで", "5番手と入替", "期待値馬の単勝"]

    def show(y, label):
        print(f"\n== {label}: {len(y)}レース")
        for n in names:
            c, rt = y[n + "_c"].sum(), y[n + "_r"].sum()
            if c == 0:
                continue
            v, cc = y[n + "_r"].values, y[n + "_c"].values
            rng = np.random.default_rng(0)
            bs = [v[i].sum() / cc[i].sum() * 100 for i in (rng.integers(0, len(v), len(v)) for _ in range(2000)) if cc[i].sum() > 0]
            top = np.sort(v)[-3:].sum()
            print(f"  {n:12s} 的中 {(v > 0).sum():3d}  投資 {c:7,.0f}円  回収率 {rt / c * 100:5.0f}%  "
                  f"90%区間 {np.percentile(bs, 5):.0f}-{np.percentile(bs, 95):.0f}%  上位3本除く {(rt - top) / c * 100:.0f}%")
    show(x, "混戦レース全体")
    y = x[x["期待値馬の数"] > 0]
    show(y, "うち印以外に期待値1.2〜2.0の馬がいるレース")
    print(f"  期待値の馬(1頭目)の3着内率: {y['期待値馬の3着内'].mean():.1%}")
    for m, g in y.groupby(y["競走年月日"] // 100):
        print(f"  {m}: {len(g)}R 5頭BOX {g['5頭BOX_r'].sum() / g['5頭BOX_c'].sum() * 100:.0f}% / "
              f"+1頭 {g['+期待値1頭BOX_r'].sum() / g['+期待値1頭BOX_c'].sum() * 100:.0f}% / 入替 {g['5番手と入替_r'].sum() / g['5番手と入替_c'].sum() * 100:.0f}%")


if __name__ == "__main__":
    main()
