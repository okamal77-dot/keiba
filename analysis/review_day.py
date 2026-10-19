"""予想の答え合わせ (日付指定, 汎用).

使い方: python3 analysis/review_day.py <data> <日付YYYYMMDD>
  analysis/out/pred_<日付>_*.csv (旧モデル) と pred2_<日付>_*.csv (改良版) を, data/ 配下の結果・払戻と突き合わせる.
  買い目は predict_race.suggest と同じ規則: 本命の勝率40%以上=単勝◎, 25-40%=2頭軸の３連単, それ未満=３連複5頭BOX.
  (2026年10月11日以前の予想時は 本命40%以上=軸1頭の３連単 だったが, 検証は新しい規則で行う)
出力: analysis/out/review_<日付>.csv と標準出力の集計
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import trifecta_factors as T  # noqa: E402

KEY = T.KEY
OUT = Path(__file__).parent / "out"


def tri_points(A, B, C):
    return [(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3]


def bets(nos, w, hi=0.40, mid=0.25):
    """predict_race.suggest と同じ規則で (３連単の組, ３連複BOXの馬, 単勝の馬) を返す.
    予想ファイルに「補正前勝率%」があれば 勝率%は補正後なので 境目は 50%/30% (win_calibrate.py)"""
    if w[0] >= hi:
        return [], [], nos[0]
    if w[0] >= mid:
        return tri_points(nos[:2], nos[:4], nos[:5]), [], None
    return [], list(nos[:5]), None


def main():
    date = int(sys.argv[2])
    h = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][0]], ignore_index=True)
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True)
    h = h[h["競走年月日"] == date].drop_duplicates(KEY + ["馬番"])
    pay = pay[pay["競走年月日"] == date].drop_duplicates(KEY)
    for c in ["着順", "人気", "馬番"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    rows = []
    for f in sorted(OUT.glob(f"pred*_{date}_*.csv")):
        m = re.match(rf"(pred2?)_{date}_(.+)\.csv$", f.name)
        if not m or f.name.endswith("_valid.csv"):
            continue
        model = "改良版" if m.group(1) == "pred2" else "旧モデル"
        course = m.group(2)
        p = pd.read_csv(f, encoding="utf-8-sig")
        for rno, g in p.groupby("レース"):
            x = h[(h["競馬場"] == course) & (h["レース番号"] == rno)]
            pr = pay[(pay["競馬場"] == course) & (pay["レース番号"] == rno)]
            if x["着順"].notna().sum() < 3 or pr.empty:
                continue
            pr = pr.iloc[0]
            win = (int(pr["３連単組番馬番1"]), int(pr["３連単組番馬番2"]), int(pr["３連単組番馬番3"]))
            g = g.sort_values("勝率%", ascending=False)
            nos = g["馬番"].astype(int).tolist()
            w = (g["勝率%"] / 100).tolist()
            pos = x.set_index("馬番")["着順"]
            popn = x.set_index("馬番")["人気"]
            fav = x[x["人気"] == 1]
            pts, box, tan = bets(nos, w, *((0.50, 0.30) if "補正前勝率%" in g else (0.40, 0.25)))
            tan_pay = pr["単勝払戻金（円）"] if tan is not None and int(pr["単勝組番"]) == tan else 0
            hit_t = win in pts
            hit_b = bool(box) and set(win) <= set(box)
            kesi = g[g["消し"].astype(str) == "True"]["馬番"].tolist() if "消し" in g else \
                g[g["注記"].fillna("").str.contains("消し")]["馬番"].tolist()
            rows.append({
                "モデル": model, "競馬場": course, "R": rno, "◎": nos[0], "◎の人気": popn.get(nos[0]),
                "◎の着順": pos.get(nos[0]), "1番人気の着順": fav["着順"].iloc[0] if len(fav) else np.nan,
                "印5頭の3着内数": int(sum(pos.get(n, 99) <= 3 for n in nos[:5])),
                "消し該当": len(kesi), "消しの3着内": int(sum(pos.get(n, 99) <= 3 for n in kesi)),
                "結果": "-".join(map(str, win)), "３連単配当": pr["３連単払戻金（円）"],
                "３連単点数": len(pts), "３連単払戻": pr["３連単払戻金（円）"] if hit_t else 0,
                "３連複点数": 10 if box else 0, "３連複払戻": pr["３連複払戻金（円）"] if hit_b else 0,
                "単勝◎払戻": pr["単勝払戻金（円）"] if int(pr["単勝組番"]) == nos[0] else 0,
                "推奨単勝点数": 1 if tan is not None else 0, "推奨単勝払戻": tan_pay,
            })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"review_{date}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)

    def summ(g):
        return pd.Series({
            "レース数": len(g), "◎の勝率%": (g["◎の着順"] == 1).mean() * 100, "◎の3着内率%": (g["◎の着順"] <= 3).mean() * 100,
            "1番人気の勝率%": (g["1番人気の着順"] == 1).mean() * 100, "1番人気の3着内率%": (g["1番人気の着順"] <= 3).mean() * 100,
            "◎が1番人気%": (g["◎の人気"] == 1).mean() * 100,
            "単勝◎ 回収率% (全レース)": g["単勝◎払戻"].sum() / (len(g) * 100) * 100,
            "推奨単勝 的中": f"{(g['推奨単勝払戻'] > 0).sum()}/{g['推奨単勝点数'].sum()}",
            "推奨単勝 回収率%": g["推奨単勝払戻"].sum() / max(g["推奨単勝点数"].sum() * 100, 1) * 100,
            "３連単 的中": f"{(g['３連単払戻'] > 0).sum()}/{(g['３連単点数'] > 0).sum()}",
            "３連単 回収率%": g["３連単払戻"].sum() / max(g["３連単点数"].sum() * 100, 1) * 100,
            "３連複 的中": f"{(g['３連複払戻'] > 0).sum()}/{(g['３連複点数'] > 0).sum()}",
            "３連複 回収率%": g["３連複払戻"].sum() / max(g["３連複点数"].sum() * 100, 1) * 100,
            "推奨買い目 合計回収率%": (g["推奨単勝払戻"].sum() + g["３連単払戻"].sum() + g["３連複払戻"].sum())
            / max((g["推奨単勝点数"].sum() + g["３連単点数"].sum() + g["３連複点数"].sum()) * 100, 1) * 100,
            "消しの3着内": f"{g['消しの3着内'].sum()}/{g['消し該当'].sum()}",
        })
    s = t.groupby(["モデル", "競馬場"]).apply(summ)
    tot = t.groupby("モデル").apply(summ)
    tot.index = pd.MultiIndex.from_product([tot.index, ["計"]])
    res = pd.concat([s, tot]).sort_index()
    print(res.round(1).to_string())
    print()
    print(t[["モデル", "競馬場", "R", "◎", "◎の人気", "◎の着順", "1番人気の着順", "印5頭の3着内数", "結果", "３連単配当", "３連単払戻", "３連複払戻"]].to_string(index=False))


if __name__ == "__main__":
    main()
