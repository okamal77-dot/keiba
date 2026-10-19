"""オッズを使わない予想 (モデルの勝率) と 結果のずれの確認と補正.

使い方: python3 analysis/win_calibrate.py <data>
予想は koryu_backtest.py の koryu_rows_*.csv (2025年1月〜, 半年ごとに それ以前のデータだけで学習, 全14場).
モデルの勝率は なだらかすぎる (5%未満は実際より高く, 20〜60%は実際より低い).
補正: 勝率を40の区間 (分位点) に分け, 区間ごとの倍率を求め, 区間の中央を結んだ単調な折れ線で変換して レースごとに合計1へ割り直す
      (順位は変わらない).
      倍率は「割り直した後の予想の平均 = 実際の勝率」になるまで繰り返して決める.
      (勝率^α で割り直す方法も試したが, 上の方で行き過ぎた: 50-60%帯で 予想55% 実際48%)
2025年で決めて 2026年で確かめる → 全期間で決め直して analysis/out/winc_map.csv に保存 (predict_v2.py が使う).
買い方の境目: 補正後の勝率では 単勝◎ 50%以上, 2頭軸 30〜50% (補正前の 40%/25% より 2026年の回収率が良い).
"""
import glob

import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY
NBIN = 40
HI, MID = 0.50, 0.30  # 補正後の勝率での 買い方の境目


def fit_map(df, iters=6):
    edges = np.quantile(df["勝率"], np.linspace(0, 1, NBIN + 1))
    edges[0], edges[-1] = 0.0, 1.01
    b = np.digitize(df["勝率"], edges) - 1
    ratio = np.ones(NBIN)
    for _ in range(iters):
        q = df["勝率"] * ratio[b]
        q = q / q.groupby(df["rid"]).transform("sum")
        g = pd.DataFrame({"b": b, "q": q, "w": df["win"]}).groupby("b")
        ratio = ratio * (g["w"].mean() / g["q"].mean()).reindex(range(NBIN)).fillna(1).values
    return pd.DataFrame({"下限": edges[:-1], "上限": edges[1:], "倍率": ratio})


def transform(w, m):
    """区間の中央での 勝率×倍率 を結んだ折れ線 (増え方が逆転しないよう単調にする). 割り直す前の値を返す"""
    c = ((m["下限"] + m["上限"].clip(upper=1.0)) / 2).values
    y = np.maximum.accumulate(c * m["倍率"].values)
    return np.interp(w, np.r_[0.0, c, 1.0], np.r_[0.0, y, max(y[-1], 1.0)])


def apply_df(df, m):
    q = pd.Series(transform(df["勝率"].values, m), index=df.index)
    return q / q.groupby(df["rid"]).transform("sum")


def load_map():
    return pd.read_csv(T.OUT / "winc_map.csv", encoding="utf-8-sig")


def calibrate(w, m):
    """1レース分の勝率の配列を補正する"""
    q = transform(np.asarray(w, float), m)
    return q / q.sum()


def reliability(df, col):
    b = pd.cut(df[col], [0, .02, .05, .1, .2, .3, .4, .5, .6, .7, 1.01], right=False)
    return df.groupby(b, observed=True).agg(頭数=("win", "size"), 予想=(col, "mean"), 実際=("win", "mean"))


def rule_roi(df, col, pay, hi=0.40, mid=0.25):
    cost = ret = 0
    for k, g in df.groupby(KEY):
        if k not in pay.index:
            continue
        p = pay.loc[k]
        g = g.sort_values("印順")
        nos = g["馬番"].astype(int).tolist()
        w0 = g[col].iloc[0]
        try:
            t3 = (int(p["３連単組番馬番1"]), int(p["３連単組番馬番2"]), int(p["３連単組番馬番3"]))
            win_no = int(p["単勝組番"])
        except (ValueError, TypeError):
            continue
        if w0 >= hi:
            cost += 100
            ret += float(p["単勝払戻金（円）"]) * (win_no == nos[0])
        elif w0 >= mid:
            pts = [(a, b, c) for a in nos[:2] for b in nos[:4] for c in nos[:5] if len({a, b, c}) == 3]
            cost += 100 * len(pts)
            ret += float(p["３連単払戻金（円）"]) * (t3 in pts)
        else:
            cost += 1000
            ret += float(p["３連複払戻金（円）"]) * set(t3).issubset(nos[:5])
    return ret / cost * 100


def main():
    r = pd.concat([pd.read_csv(f) for f in glob.glob(str(T.OUT / "koryu_rows_*.csv"))])
    r = r[r["着順"].notna()].copy()
    r["win"] = (r["着順"] == 1).astype(int)
    r["rid"] = r["競馬場"] + r["競走年月日"].astype(str) + "_" + r["レース番号"].astype(str)
    r = r[r.groupby("rid")["win"].transform("sum") == 1]
    tr, te = r[r["競走年月日"] < 20260101].copy(), r[r["競走年月日"] >= 20260101].copy()
    m = fit_map(tr)
    te["補正勝率"] = apply_df(te, m)
    tr["補正勝率"] = apply_df(tr, m)
    print(f"== 2026年での確認 ({te['rid'].nunique()}レース, 補正は2025年で決めたもの)")
    ll0, ll1 = np.log(te.loc[te["win"] == 1, "勝率"]).mean(), np.log(te.loc[te["win"] == 1, "補正勝率"]).mean()
    print(f"勝ち馬の対数尤度: 補正前 {ll0:.4f} → 補正後 {ll1:.4f}")
    rel = pd.concat({"補正前": reliability(te, "勝率"), "補正後": reliability(te, "補正勝率")}, axis=1)
    print(rel.round(3).to_string())
    top = te[te["印順"] == 1]
    print(f"◎: 予想 補正前 {top['勝率'].mean():.3f} / 補正後 {top['補正勝率'].mean():.3f} / 実際 {top['win'].mean():.3f}")
    pay = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY).set_index(KEY)
    rows = []
    for name, d in [("2025年 (補正を決めた期間)", tr), ("2026年 (確認)", te)]:
        rows.append({"期間": name, "補正前の勝率 40%/25%": rule_roi(d, "勝率", pay),
                     "補正後の勝率 40%/25%": rule_roi(d, "補正勝率", pay),
                     "補正後の勝率 50%/30%": rule_roi(d, "補正勝率", pay, HI, MID)})
    roi = pd.DataFrame(rows).set_index("期間")
    print("\n== ルールの買い目の回収率% (単勝◎の境目/2頭軸の境目)\n", roi.round(1).to_string())
    full = fit_map(r)
    full.to_csv(T.OUT / "winc_map.csv", index=False, encoding="utf-8-sig")
    rel.to_csv(T.OUT / "winc_reliability_2026.csv", encoding="utf-8-sig")
    roi.to_csv(T.OUT / "winc_rule_roi.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
