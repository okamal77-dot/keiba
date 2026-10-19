"""人気（オッズ）を土台に, 市場が見落としている要因で補正した予想.

使い方: python3 analysis/predict_with_odds.py <data> <horselist.csv> <racelist.csv> <odds.csv> <競馬場> <レース番号>
  odds.csv: 競馬場,競走年月日,レース番号,馬番,単勝オッズ,人気,馬体重,馬体重増減

モデル:
  1) 着順 (1〜3着) を「人気(対数) + favorite_extra.py の全項目」で説明する順位ロジットを作り,
     人気以外の項目の合計を「補正点」とする (人気だけのモデルとも直近3か月で比較).
  2) 確定オッズがある月 (T.ODDS_MONTH) で, 着順 ~ log(オッズから出した市場の勝率) + 補正点 を推定
     (前半で推定・後半で検証, 市場の勝率だけのモデルと比較) し, 当日のオッズに当てはめる.
出力: analysis/out/odds_pred_<日付>_<場>_<R>.csv
"""
import sys

import numpy as np
import pandas as pd

import ability_index as A
import favorite_extra as X
import predict_race as P
import trifecta_factors as T

KEY = T.KEY


def evaluate(va, fva, fit_ret, label):
    b, mu, sd = fit_ret
    Xv, Mv, Ov, _ = P.pad(va, fva)
    Pw = P.win_probs(Xv, Mv, b, mu, sd)
    ok = Ov[:, 0] >= 0
    top = Pw.argmax(1)
    win = (top == Ov[:, 0])[ok].mean() * 100
    in3 = np.mean([t in o for t, o in zip(top[ok], Ov[ok])]) * 100
    ll = np.log(Pw[np.arange(len(Pw)), Ov[:, 0]][ok]).mean()
    return {"モデル": label, "レース数": int(ok.sum()), "本命の勝率%": win, "本命の3着内率%": in3, "勝ち馬の対数尤度": ll}


def main():
    data, hfile, rfile, ofile, course, rno = sys.argv[1:7]
    rno = int(rno)
    h, _, _ = T.load()
    e, r = P.load_entries(hfile, rfile, course)
    e = e[e["レース番号"] == rno].copy()
    o = pd.read_csv(ofile, encoding="utf-8-sig")
    e = e.drop(columns=["人気", "馬体重", "馬体重増減"]).merge(o, on=KEY + ["馬番"], how="left")
    h = pd.concat([h, e], ignore_index=True)
    h["予想対象"] = h["予想対象"].astype("boolean").fillna(False).astype(bool)
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    h = X.extra(h)
    prize = r.set_index(KEY)["1着賞金(円)"]
    m = h["予想対象"]
    h.loc[m, "1着賞金(円)"] = prize.reindex(pd.MultiIndex.from_frame(h.loc[m, KEY])).values
    jr = A.jockey_rates(h)
    d = h[(h["年"] >= 2024) & h["人気"].notna()].copy()
    d.index = range(len(d))
    f = X.build(d, jr)
    f["人気(対数)"] = np.log(d["人気"])
    f_pop = f[["人気(対数)"]]

    hist = d[~d["予想対象"]]
    cut = hist["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = hist[hist["日付"] <= cut], hist[hist["日付"] > cut]
    rows = []
    for label, ff in [("人気のみ", f_pop), ("人気＋補正", f)]:
        Xt, Mt, Ot, _ = P.pad(tr, ff.loc[tr.index])
        fr = P.fit(Xt, Mt, Ot)
        rows.append(evaluate(va, ff.loc[va.index], fr, f"{label}（全場）"))
        vc = va[va["競馬場"] == course]
        rows.append(evaluate(vc, ff.loc[vc.index], fr, f"{label}（{course}）"))
    t = pd.DataFrame(rows).set_index("モデル")
    print(f"== 検証 {va['日付'].min():%Y-%m}〜{va['日付'].max():%Y-%m}\n", t.round(3).to_string())

    # ---- オッズを土台にした補正 ----
    # 補正点 = 人気(対数)以外の項目の合計 (8月までで学習した係数). 確定オッズのある月で
    # 着順 ~ α・log(市場の勝率) + β・補正点 を推定し (前半で推定, 後半で検証), 予想に使う
    odds_month = T.ODDS_MONTH
    pre = hist[hist["競走年月日"] // 100 < odds_month]
    Xt, Mt, Ot, _ = P.pad(pre, f.loc[pre.index])
    b, mu, sd = P.fit(Xt, Mt, Ot)
    coef = pd.Series(b, index=f.columns)
    corr_cols = [c for c in f.columns if c != "人気(対数)"]
    zc = ((f - mu) / sd)[corr_cols]
    d["補正点"] = (zc * coef[corr_cols]).sum(axis=1)
    d["補正点"] -= d.groupby(KEY)["補正点"].transform("mean")

    files = sorted((T.DATA / "odds").glob(f"{odds_month}_*_odds.csv"))
    od = pd.concat([T.read(f"odds/{x.name}", usecols=KEY + ["賭式", "番号1", "オッズ"]) for x in files])
    od = od[od["賭式"] == "単勝"].rename(columns={"番号1": "馬番", "オッズ": "単勝オッズ_確定"}).drop(columns="賭式")
    od["馬番"] = pd.to_numeric(od["馬番"], errors="coerce")
    sep = d[(d["競走年月日"] // 100 == odds_month) & ~d["予想対象"]].reset_index().merge(
        od, on=KEY + ["馬番"], how="inner").set_index("index")
    sep = sep[sep["単勝オッズ_確定"] > 0]
    sep = sep[sep.groupby(KEY)["馬番"].transform("size") == sep.groupby(KEY)["頭数"].transform("first")]

    def mlog(x, col):
        q = 1 / x[col]
        return np.log(q / q.groupby([x[k] for k in KEY]).transform("sum"))

    sep["市場"] = mlog(sep, "単勝オッズ_確定")
    half = sep["競走年月日"] < odds_month * 100 + 16
    rows = []
    fits = {}
    for label, cols in [("市場の勝率のみ", ["市場"]), ("市場の勝率＋補正点", ["市場", "補正点"])]:
        a, v = sep[half], sep[~half]
        Xa, Ma, Oa, _ = P.pad(a, a[cols])
        fr = P.fit(Xa, Ma, Oa, l2=0.1)
        rows.append(evaluate(v, v[cols], fr, f"{label}（9月後半, 全場）"))
        vc = v[v["競馬場"] == course]
        if len(vc):
            rows.append(evaluate(vc, vc[cols], fr, f"{label}（9月後半, {course}）"))
        Xa, Ma, Oa, _ = P.pad(sep, sep[cols])
        fits[label] = P.fit(Xa, Ma, Oa, l2=0.1)
    t2 = pd.DataFrame(rows).set_index("モデル")
    print("\n== オッズを土台にした検証\n", t2.round(3).to_string())
    bb, mm, ss = fits["市場の勝率＋補正点"]
    print("係数 (標準化後): 市場", round(bb[0] / ss[0], 3), " 補正点", round(bb[1] / ss[1], 3))

    # 予想 (全期間で学習した補正点の係数を使う)
    Xt, Mt, Ot, _ = P.pad(hist, f.loc[hist.index])
    b, mu, sd = P.fit(Xt, Mt, Ot)
    coef = pd.Series(b, index=f.columns)
    zc = ((f - mu) / sd)[corr_cols]
    contrib_all = zc * coef[corr_cols]
    tg = d[d["予想対象"]].sort_values("馬番").copy()
    tg["市場"] = mlog(tg, "単勝オッズ")
    contrib = contrib_all.loc[tg.index]
    contrib = contrib - contrib.mean()
    tg["補正点"] = contrib.sum(axis=1)
    Xv, Mv, _, _ = P.pad(tg, tg[["市場", "補正点"]], with_order=False)
    w = P.win_probs(Xv, Mv, bb, mm, ss)[0, :len(tg)]
    top3, tri = P.pl_probs(w)
    mk = np.exp(tg["市場"].values)
    main_cols = ["平均走破タイム", "近3走ベスト指数", "位置取り", "騎手3着内率", "コンビ3着内率", "調教師3着内率",
                 "前走勝ち馬との差(秒)", "昇級(賞金比,対数)", "年齢", "牝馬", "前走追込", "体重+10kg以上", "遠征", "近走なし"]
    out = pd.DataFrame({
        "馬番": tg["馬番"].astype(int).values, "馬名": tg["馬名"].values, "騎手": tg["騎手名"].values,
        "単勝オッズ": tg["単勝オッズ"].values, "人気": tg["人気"].astype(int).values,
        "市場の勝率%": mk * 100, "補正後の勝率%": w * 100, "補正後の3着内率%": top3 * 100,
        "補正後÷市場": w / mk, "補正点": tg["補正点"].values,
    })
    for c in main_cols:
        out[c] = contrib[c].values
    out = out.sort_values("補正後の勝率%", ascending=False)
    date = int(tg["競走年月日"].iloc[0])
    out.to_csv(T.OUT / f"odds_pred_{date}_{course}_{rno}R.csv", index=False, encoding="utf-8-sig")
    pd.concat([t, t2]).to_csv(T.OUT / f"odds_pred_{date}_{course}_{rno}R_valid.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    nos = tg["馬番"].astype(int).values
    best = sorted(tri.items(), key=lambda kv: -kv[1])[:12]
    print("３連単 上位:", " / ".join(f"{nos[a]}-{nos[bb_]}-{nos[c]} {p*100:.1f}%" for (a, bb_, c), p in best))


if __name__ == "__main__":
    main()
