"""2026年10月9日の予想の答え合わせ.

対象: 笠松全12R (人気を使わない事前情報モデル, pred_20261009_笠松.csv),
      大井9R・10R, 園田12R (オッズ＋補正, 会話で示した印・買い目)
入力: data/results/20261009_{horselist,payback}.csv
出力: analysis/out/review_20261009.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import predict_race as P  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]

# オッズ＋補正で予想したレース: 印 (◎○▲△△) と買い目
ODDS_RACES = {
    ("大井", 9): {"印": [8, 7, 3, 9, 1], "３連単": ([8], [3, 7, 9, 1], [3, 7, 9, 1]), "３連複BOX": [1, 3, 7, 8, 9]},
    ("大井", 10): {"印": [11, 6, 13, 9, 4], "３連単": ([6, 11], [6, 11, 13, 9], [6, 11, 13, 9, 4]),
                  "３連複BOX": [4, 6, 9, 11, 13]},
    ("園田", 12): {"印": [11, 2, 8, 10, 12], "３連単": ([11, 8], [11, 8, 2, 10], [11, 8, 2, 10, 12]),
                  "３連複BOX": [2, 8, 10, 11, 12]},
}


def tri_points(A, B, C):
    return [(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3]


def main():
    h = pd.read_csv(DATA / "results/20261009_horselist.csv", encoding="utf-8-sig")
    pay = pd.read_csv(DATA / "results/20261009_payback.csv", encoding="utf-8-sig").drop_duplicates(KEY)
    for c in ["着順", "人気", "馬番"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    rows = []

    def judge(course, rno, marks, A, B, C, box, kind):
        x = h[(h["競馬場"] == course) & (h["レース番号"] == rno)]
        pr = pay[(pay["競馬場"] == course) & (pay["レース番号"] == rno)].iloc[0]
        win = (int(pr["３連単組番馬番1"]), int(pr["３連単組番馬番2"]), int(pr["３連単組番馬番3"]))
        pos = x.set_index("馬番")["着順"]
        fav = x[x["人気"] == 1]
        pts = tri_points(A, B, C)
        hit_t = win in pts
        boxpts = len(box) * (len(box) - 1) * (len(box) - 2) // 6 if box else 0
        hit_b = bool(box) and set(win) <= set(box)
        rows.append({
            "競馬場": course, "R": rno, "方式": kind, "◎": marks[0], "◎の着順": pos.get(marks[0]),
            "◎の人気": x.set_index("馬番")["人気"].get(marks[0]),
            "1番人気の着順": fav["着順"].iloc[0] if len(fav) else np.nan,
            "印5頭の3着内数": int(sum(pos.get(m, 99) <= 3 for m in marks)),
            "結果(３連単)": "-".join(map(str, win)), "３連単配当": pr["３連単払戻金（円）"],
            "３連単買い目点数": len(pts), "３連単的中": hit_t, "３連単払戻": pr["３連単払戻金（円）"] if hit_t else 0,
            "３連複BOX点数": boxpts, "３連複的中": hit_b, "３連複払戻": pr["３連複払戻金（円）"] if hit_b else 0,
        })

    # 笠松: 事前情報モデル (勝率順の印, 買い目は suggest() と同じ規則)
    pred = pd.read_csv(OUT / "pred_20261009_笠松.csv", encoding="utf-8-sig")
    for rno, g in pred.groupby("レース"):
        g = g.sort_values("勝率%", ascending=False)
        nos = g["馬番"].astype(int).values
        w = g["勝率%"].values / 100
        o = list(range(len(nos)))
        marks = list(nos[:5])
        if w[0] >= 0.40:
            A, B, C = [nos[0]], list(nos[1:5]), list(nos[1:5])
            pts = tri_points(A, B, C)
            if w[1] >= 0.20:
                pts += tri_points([nos[1]], [nos[0]], list(nos[2:5]))
            box = []
        elif w[0] >= 0.25:
            A, B, C = list(nos[:2]), list(nos[:4]), list(nos[:5])
            box = []
        else:
            A = B = C = []
            box = list(nos[:5])
        judge("笠松", rno, marks, A, B, C, box, "事前情報モデル")
        if w[0] >= 0.40 and w[1] >= 0.20:  # 2本目の流しを含める
            x = h[(h["競馬場"] == "笠松") & (h["レース番号"] == rno)]
            pr = pay[(pay["競馬場"] == "笠松") & (pay["レース番号"] == rno)].iloc[0]
            win = (int(pr["３連単組番馬番1"]), int(pr["３連単組番馬番2"]), int(pr["３連単組番馬番3"]))
            hit = win in pts
            rows[-1].update({"３連単買い目点数": len(pts), "３連単的中": hit, "３連単払戻": pr["３連単払戻金（円）"] if hit else 0})
    for (course, rno), v in ODDS_RACES.items():
        judge(course, rno, v["印"], *v["３連単"], v["３連複BOX"], "オッズ＋補正")

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "review_20261009.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    for kind, g in t.groupby("方式"):
        cost_t = g["３連単買い目点数"].sum() * 100
        cost_b = g["３連複BOX点数"].sum() * 100
        print(f"\n[{kind}] {len(g)}レース  ◎の勝率 {(g['◎の着順'] == 1).mean():.0%}  ◎の3着内率 {(g['◎の着順'] <= 3).mean():.0%}"
              f"  / 1番人気の勝率 {(g['1番人気の着順'] == 1).mean():.0%}  3着内率 {(g['1番人気の着順'] <= 3).mean():.0%}")
        if cost_t:
            print(f"  ３連単: {g['３連単的中'].sum()}/{(g['３連単買い目点数'] > 0).sum()}レース的中  購入 {cost_t:,}円  払戻 {g['３連単払戻'].sum():,.0f}円")
        if cost_b:
            print(f"  ３連複BOX: {g['３連複的中'].sum()}/{(g['３連複BOX点数'] > 0).sum()}レース的中  購入 {cost_b:,}円  払戻 {g['３連複払戻'].sum():,.0f}円")


if __name__ == "__main__":
    main()
