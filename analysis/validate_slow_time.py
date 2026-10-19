"""「前走の着順は良いが時計が遅い」(第19節 (a)) を改良版モデルに加えると予想が良くなるかの検証.

改良版 (predict_v2.build) に次の2項目を足したモデルと比べる (直近3か月で検証, 6場).
  前走5着以内で時計が遅い : 前走着順 5着以内 かつ 前走指数が全体の下位1/3
  前走1着で時計が速い     : 前走1着 かつ 前走指数が全体の上位1/3
指標: 本命の勝率, 本命の3着内率, 勝ち馬の対数尤度 (大きいほど良い), 加えた項目の係数 (標準化後, +で勝ちやすい)
出力: analysis/out/vst_*.csv
"""
import numpy as np
import pandas as pd

import ability_index as A
import favorite_extra as X
import predict_race as P
import predict_v2 as V
import trifecta_factors as T

KEY = T.KEY
COURSES = ["佐賀", "盛岡", "名古屋", "川崎", "金沢", "門別"]


def evaluate(tr, va, f):
    Xt, Mt, Ot, _ = P.pad(tr, f.loc[tr.index])
    b, mu, sd = P.fit(Xt, Mt, Ot)
    Xv, Mv, Ov, _ = P.pad(va, f.loc[va.index])
    Pw = P.win_probs(Xv, Mv, b, mu, sd)
    ok = Ov[:, 0] >= 0
    top = Pw.argmax(1)
    return {"検証レース数": int(ok.sum()), "本命の勝率%": (top == Ov[:, 0])[ok].mean() * 100,
            "本命の3着内率%": np.mean([t in o for t, o in zip(top[ok], Ov[ok])]) * 100,
            "勝ち馬の対数尤度": np.log(Pw[np.arange(len(Pw)), Ov[:, 0]][ok]).mean()}, pd.Series(b, index=f.columns)


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    races = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][1]], ignore_index=True).drop_duplicates(KEY)
    h = h.drop(columns=["レース名"], errors="ignore").merge(races[KEY + ["レース名"]], on=KEY, how="left")
    h = T.features(h)
    h = A.add_runs(h)
    h = X.extra(h)
    h = V.class_features(h)
    q = h.loc[h["年"] >= 2024, "指数"].quantile([1 / 3, 2 / 3]).values
    jr_a = A.jockey_rates(h)
    rows, coefs = [], []
    for course in COURSES:
        jr_c = P.jockey_rate(h, course)
        d = h[(h["競馬場"] == course) & (h["年"] >= 2024)].copy()
        d.index = range(len(d))
        f = V.build(d, jr_c, jr_a)
        g = f.copy()
        g["前走5着以内で時計が遅い"] = ((d["前走着順"] <= 5) & (d["前走指数"] <= q[0])).astype(float).values
        g["前走1着で時計が速い"] = ((d["前走着順"] == 1) & (d["前走指数"] >= q[1])).astype(float).values
        cut = d["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
        tr, va = d[d["日付"] <= cut], d[d["日付"] > cut]
        for label, ff in [("改良版 (いま)", f), ("＋着順は良いが時計が遅い", g)]:
            r, b = evaluate(tr, va, ff)
            rows.append({"競馬場": course, "モデル": label, **r})
            if label.startswith("＋"):
                coefs.append(b[["前走5着以内で時計が遅い", "前走1着で時計が速い"]].rename(course))
        print(pd.DataFrame(rows[-2:]).round(3).to_string(index=False), flush=True)
    t = pd.DataFrame(rows).set_index(["競馬場", "モデル"])
    c = pd.DataFrame(coefs)
    t.to_csv(T.OUT / "vst_valid.csv", encoding="utf-8-sig")
    c.to_csv(T.OUT / "vst_coef.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
