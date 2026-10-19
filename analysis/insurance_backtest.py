"""大本命 (◎の勝率40%以上) の単勝に 馬単の保険 (◎が2着の目) を足す買い方の比較.

使い方: python3 analysis/insurance_backtest.py <data>
予想は koryu_backtest.py が出した koryu_rows_*.csv (2025年1月〜, 半年ごとに それ以前のデータだけで学習したモデル, 全14場).
相手 ○▲△ は印の順 (◎○ は勝率順, ▲△ は 2〜3着補正の順). 1点100円.
出力: analysis/out/insurance_races.csv, insurance_summary.csv
"""
import glob

import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY


def main():
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(str(T.OUT / "koryu_rows_*.csv"))])
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY).set_index(KEY)
    res = []
    for k, g in rows.groupby(KEY):
        if k not in pay.index or g["着順"].isna().all():
            continue
        g = g.sort_values("印順")
        if g["勝率"].iloc[0] < 0.40 or len(g) < 5:
            continue
        p = pay.loc[k]
        m, a, b, c = g["馬番"].astype(int).iloc[:4]
        try:
            ex = (int(p["馬単組番1"]), int(p["馬単組番2"]))
            win_no = int(p["単勝組番"])
        except (ValueError, TypeError):
            continue
        ep, wp = float(p["馬単払戻金（円）"]), float(p["単勝払戻金（円）"])
        exr = lambda s: ep * (ex in s)
        res.append({"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "◎勝率": g["勝率"].iloc[0],
                    "◎人気": g["人気"].iloc[0], "◎着順": g["着順"].iloc[0],
                    "単勝◎": wp * (win_no == m),
                    "馬単◎→○▲": exr({(m, a), (m, b)}),
                    "馬単○▲→◎": exr({(a, m), (b, m)}),
                    "馬単○▲△→◎": exr({(a, m), (b, m), (c, m)}),
                    "馬単◎⇄○▲": exr({(m, a), (m, b), (a, m), (b, m)}),
                    "馬単2着◎(全頭→◎)": ep * (ex[1] == m)})
    x = pd.DataFrame(res)
    n_all = rows.groupby(KEY).ngroups
    x["頭数"] = x.apply(lambda r: 0, axis=1)
    x.to_csv(T.OUT / "insurance_races.csv", index=False, encoding="utf-8-sig")
    # 全頭→◎ の点数は (頭数-1) 点
    heads = rows.groupby(KEY).size()
    x["全頭点数"] = heads.reindex(pd.MultiIndex.from_frame(x[KEY])).values - 1

    plans = {  # 名前: [(列, 点数)]
        "単勝◎だけ (1点)": [("単勝◎", 1)],
        "馬単 ◎→○▲ (2点)": [("馬単◎→○▲", 2)],
        "馬単 ○▲→◎ (2点)": [("馬単○▲→◎", 2)],
        "馬単 ◎⇄○▲ (4点)": [("馬単◎⇄○▲", 4)],
        "単勝◎ + 保険 馬単○▲→◎ (3点)": [("単勝◎", 1), ("馬単○▲→◎", 2)],
        "単勝◎ + 保険 馬単○▲△→◎ (4点)": [("単勝◎", 1), ("馬単○▲△→◎", 3)],
        "単勝◎×2 + 保険 馬単○▲→◎ (4点)": [("単勝◎", 2), ("馬単○▲→◎", 2)],
        "参考: 馬単 全頭→◎ (頭数-1点)": [("馬単2着◎(全頭→◎)", None)],
    }

    def evaluate(y):
        out = {}
        for name, legs in plans.items():
            cost = sum((y["全頭点数"] if pts is None else pts) * 100 for _, pts in legs)
            ret = sum(y[col] * (1 if pts is None or col != "単勝◎" else pts) for col, pts in legs)
            cost = cost.sum() if hasattr(cost, "sum") else cost * len(y)
            hit = (sum(y[col] for col, _ in legs) > 0).mean() * 100
            r = ret.sum() / cost * 100
            rng = np.random.default_rng(0)
            v = ret.values
            cs = cost / len(y)
            bs = [v[rng.integers(0, len(v), len(v))].sum() / cost * 100 for _ in range(1000)]
            out[name] = {"的中率%": hit, "回収率%": r, "90%区間": f"{np.percentile(bs, 5):.0f}-{np.percentile(bs, 95):.0f}"}
        return pd.DataFrame(out).T

    print(f"対象: ◎の勝率40%以上 {len(x)}レース (全 {n_all}レース中). ◎の勝率 {(x['◎着順'] == 1).mean():.0%}, 2着 {(x['◎着順'] == 2).mean():.0%}")
    allr = evaluate(x)
    print("\n== 全体\n", allr.round(1).to_string())
    x["期"] = np.where(x["競走年月日"] < 20250701, "2025前半", np.where(x["競走年月日"] < 20260101, "2025後半",
                     np.where(x["競走年月日"] < 20260701, "2026前半", "2026後半")))
    by = {}
    for k, y in x.groupby("期"):
        by[k] = evaluate(y)["回収率%"]
    by = pd.DataFrame(by)
    print("\n== 期間ごとの回収率%\n", by.round(0).to_string())
    bands = pd.cut(x["◎勝率"], [0.40, 0.50, 0.60, 1.01], right=False, labels=["40-50%", "50-60%", "60%以上"])
    bb = {}
    for k, y in x.groupby(bands, observed=True):
        bb[f"{k} ({len(y)}R, ◎勝{(y['◎着順'] == 1).mean():.0%})"] = evaluate(y)["回収率%"]
    bb = pd.DataFrame(bb)
    print("\n== ◎の勝率の帯ごとの回収率%\n", bb.round(0).to_string())
    fav = x["◎人気"] == 1
    pf = {f"◎が1番人気 ({fav.sum()}R)": evaluate(x[fav])["回収率%"], f"◎が1番人気でない ({(~fav).sum()}R)": evaluate(x[~fav])["回収率%"]}
    pf = pd.DataFrame(pf)
    print("\n== ◎の人気別の回収率%\n", pf.round(0).to_string())
    pd.concat({"全体": allr["回収率%"], **{c: by[c] for c in by}, **{c: bb[c] for c in bb}, **{c: pf[c] for c in pf}}, axis=1) \
        .to_csv(T.OUT / "insurance_summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
