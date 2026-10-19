"""期待値の実績補正 (勝率の補正 と オッズの補正).

ev_select.py の期待値 = モデル勝率 × 想定オッズ は, 人気薄で大きく出すぎる. 原因は次の2つが考えられる.
 1) モデルの勝率が高すぎる → 勝率の補正:
    (モデル勝率の帯 × 想定人気の帯) ごとの 実際の勝率 に置き換える (少ない区分は モデル勝率の平均へ縮める)
 2) 想定オッズが高すぎる → オッズの補正:
    実際に勝った馬について 実際のオッズ (単勝払戻÷100) ÷ 想定オッズ を 想定人気の帯ごとに平均した比を掛ける
    (期待値に効くのは「勝ったときのオッズ」なので, 勝った馬だけで比を取る)
補正後の期待値 = 補正勝率 × 想定オッズ × オッズの比
検証: 偶数日・奇数日に分け, 片方で求めた補正をもう片方に当てる (2分割の交差検証).
対象: 直近3か月 (pred2_underrated_rows_*.csv の7場). 出力: analysis/out/evc_*.csv
"""
import glob

import numpy as np
import pandas as pd

import ev_select as EV
import trifecta_factors as T

KEY = T.KEY
P_BINS = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01]
P_LAB = ["2%未満", "2-5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%以上"]
POP_BINS = [0, 1, 3, 6, 9, 99]
POP_LAB = ["1", "2-3", "4-6", "7-9", "10以下"]
SHRINK = 30


def bands(m):
    return (pd.cut(m["勝率"], P_BINS, labels=P_LAB, right=False), pd.cut(m["想定人気"], POP_BINS, labels=POP_LAB))


def fit(tr):
    pb, qb = bands(tr)
    g = tr.groupby([pb, qb], observed=True)
    win = (g["着順"].apply(lambda s: (s == 1).sum()) + SHRINK * g["勝率"].mean()) / (g.size() + SHRINK)
    pcal = (win / g["勝率"].mean()).rename("勝率の比")  # 実際の勝率 ÷ モデル勝率 (縮約後)
    w = tr[tr["着順"] == 1]
    r = (w["単勝"] / 100) / w["想定オッズ"]
    qg = r.groupby(pd.cut(w["想定人気"], POP_BINS, labels=POP_LAB), observed=True)
    ocal = ((qg.sum() + SHRINK * 1.0) / (qg.size() + SHRINK)).rename("オッズの比")
    return pcal, ocal


def apply(te, pcal, ocal):
    pb, qb = bands(te)
    pr = pcal.reindex(pd.MultiIndex.from_arrays([pb, qb])).fillna(1.0).values
    orr = ocal.reindex(qb).fillna(1.0).values
    return te["勝率"].values * pr, te["想定オッズ"].values * orr


def load_fit():
    """予想で使う補正 (直近3か月の全データで求めたもの)"""
    pc = pd.read_csv(T.OUT / "evc_fit_win.csv", encoding="utf-8-sig", index_col=[0, 1])["勝率の比"]
    oc = pd.read_csv(T.OUT / "evc_fit_odds.csv", encoding="utf-8-sig", index_col=0)["オッズの比"]
    pc.index = pd.MultiIndex.from_arrays([pc.index.get_level_values(0).astype(str), pc.index.get_level_values(1).astype(str)])
    oc.index = oc.index.astype(str)
    return pc, oc


def calibrated(win, pop, odds, pc, oc):
    """補正勝率, 補正オッズ, 補正期待値"""
    t = pd.DataFrame({"勝率": np.asarray(win, float), "想定人気": np.asarray(pop, float), "想定オッズ": np.asarray(odds, float)})
    pb, qb = bands(t)
    pr = pc.reindex(pd.MultiIndex.from_arrays([pb.astype(str), qb.astype(str)])).fillna(1.0).values
    orr = oc.reindex(qb.astype(str)).fillna(1.0).values
    p, o = t["勝率"].values * pr, t["想定オッズ"].values * orr
    return p, o, p * o


def main():
    h, _, _ = T.load()
    h = T.features(h)
    q, rate = EV.market_table(h)
    m = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(T.OUT / "pred2_underrated_rows_*.csv"))])
    m = m.merge(h[KEY + ["馬番", "頭数"]], on=KEY + ["馬番"], how="left")
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True).drop_duplicates(KEY)
    tan = raw[KEY + ["単勝組番", "単勝払戻金（円）"]].dropna()
    tan = tan.assign(馬番=tan["単勝組番"].astype(int)).groupby(KEY + ["馬番"])["単勝払戻金（円）"].sum().rename("単勝")
    m = m.merge(tan.reset_index(), on=KEY + ["馬番"], how="left").fillna({"単勝": 0})
    m = m[m["人気"].notna() & m["頭数"].notna()].reset_index(drop=True)
    m["期待値"], m["想定オッズ"] = EV.expected_value(m, m["勝率"], m["想定人気"], q, rate)
    m = m[m["想定オッズ"].notna()].reset_index(drop=True)

    # 全期間で見た ずれ (説明用)
    pcal_all, ocal_all = fit(m)
    pcal_all.to_csv(T.OUT / "evc_fit_win.csv", encoding="utf-8-sig")
    ocal_all.to_csv(T.OUT / "evc_fit_odds.csv", encoding="utf-8-sig")
    pb, qb = bands(m)
    g = m.groupby([pb, qb], observed=True)
    calib = pd.DataFrame({"頭数": g.size(), "モデル勝率の平均%": g["勝率"].mean() * 100,
                          "実際の勝率%": g["着順"].apply(lambda s: (s == 1).mean() * 100)})
    calib["勝率の比(縮約後)"] = pcal_all
    w = m[m["着順"] == 1]
    og = w.groupby(pd.cut(w["想定人気"], POP_BINS, labels=POP_LAB), observed=True)
    oddt = pd.DataFrame({"勝った頭数": og.size(), "想定オッズの平均": og["想定オッズ"].mean(),
                         "実際のオッズの平均": og["単勝"].mean() / 100, "オッズの比(縮約後)": ocal_all})
    calib.to_csv(T.OUT / "evc_win_calibration.csv", encoding="utf-8-sig")
    oddt.to_csv(T.OUT / "evc_odds_calibration.csv", encoding="utf-8-sig")

    # 交差検証
    fold = (m["競走年月日"] % 2).values
    m["補正勝率"], m["補正オッズ"] = np.nan, np.nan
    for f in (0, 1):
        tr, te = m[fold != f], m[fold == f]
        pc, oc = fit(tr)
        p, o = apply(te, pc, oc)
        m.loc[te.index, "補正勝率"], m.loc[te.index, "補正オッズ"] = p, o
    m["補正期待値"] = m["補正勝率"] * m["補正オッズ"]
    ok = ~m["消し"]
    rows = []
    for name, col in [("補正前の期待値", "期待値"), ("補正後の期待値", "補正期待値")]:
        b = pd.cut(m[col], [0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 99])
        t = m[ok].groupby(b[ok], observed=True).apply(lambda x: pd.Series({
            "頭数": len(x), "期待値の平均": x[col].mean(), "実際の単勝回収率%": x["単勝"].mean(),
            "勝率%": (x["着順"] == 1).mean() * 100, "平均人気": x["人気"].mean()}))
        t.index = [f"{name} {i}" for i in t.index]
        rows.append(t)
    tb = pd.concat(rows)
    tb.to_csv(T.OUT / "evc_ev_bins.csv", encoding="utf-8-sig")
    sel = []
    for lo in [0.9, 1.0, 1.1, 1.2]:
        x = m[ok & (m["補正期待値"] >= lo)]
        sel.append({"補正後の期待値の下限": lo, "頭数": len(x), "1レースあたり": len(x) / m.groupby(KEY).ngroups,
                    "勝率%": (x["着順"] == 1).mean() * 100, "平均人気": x["人気"].mean(), "単勝回収率%": x["単勝"].mean(),
                    "上位3本を除く%": (x["単勝"].sum() - x["単勝"].nlargest(3).sum()) / max(len(x), 1)})
    m["補正期待値順"] = m[ok].groupby(KEY)["補正期待値"].rank(ascending=False, method="first")
    for r in (1, 2):
        x = m[m["補正期待値順"] == r]
        sel.append({"補正後の期待値の下限": f"各レース{r}位", "頭数": len(x), "勝率%": (x["着順"] == 1).mean() * 100,
                    "平均人気": x["人気"].mean(), "単勝回収率%": x["単勝"].mean(),
                    "上位3本を除く%": (x["単勝"].sum() - x["単勝"].nlargest(3).sum()) / len(x)})
    ts = pd.DataFrame(sel)
    ts.to_csv(T.OUT / "evc_selection.csv", index=False, encoding="utf-8-sig")
    m["月"] = m["競走年月日"] // 100
    x = m[ok & (m["補正期待値"] >= 1.0)]
    mo = x.groupby("月")["単勝"].agg(["size", "mean"])
    co = x.groupby("競馬場")["単勝"].agg(["size", "mean"])
    pd.set_option("display.width", 250)
    for k, t in [("勝率のずれ", calib), ("オッズのずれ (勝った馬)", oddt), ("期待値の帯ごとの実際の回収率 (交差検証)", tb),
                 ("補正後の期待値で選んだ単勝", ts), ("補正後1.0以上 月別", mo), ("補正後1.0以上 場別", co)]:
        print(f"\n== {k}\n", t.round(3).to_string())


if __name__ == "__main__":
    main()
