"""改良版の予想モデル: これまでの分析結果をすべて取り込んだ事前情報モデル (人気・オッズは使わない).

使い方: python3 analysis/predict_v2.py <data> <horselist.csv> <racelist.csv> <競馬場>

特徴量 = predict_race.py の項目
       + 能力値 (近5走の平均走破タイム・平均上がり3F・位置取り・先行率)            ... 第13節
       + 前走の着差, 昇級・降級 (賞金比), 調教師の成績, 通算勝率                    ... 第14節
       + クラス変化 (前走から / 4走前から), 近4走平均着順, 割引・加点の組み合わせ   ... 第16節
対象場の 2024年1月〜検証開始前で学習し, 直近3か月で predict_race.py の旧モデルと比較 → 全期間で再学習して予想.
過小評価候補: 勝率順位5位以内なのに想定人気 (同じ項目で人気の順番を予測したモデル, 実際の人気が分かればそれ) が
             勝率順位より2つ以上低い馬 (消し条件は除く). 直近3か月で実際の人気に対する A/E を検証して表示する.
出力: analysis/out/pred2_<日付>_<競馬場>.csv
"""
import sys

import numpy as np
import pandas as pd

import ability_index as A
import class_change as C
import favorite_drivers as F
import favorite_extra as X
import ev_calibrate as EC
import ev_select as EV
import place_backtest as PB
import predict_race as P
import trifecta_factors as T

KEY = T.KEY
UR_RANK, UR_GAP = 5, 2  # 過小評価候補: 勝率順位 UR_RANK 位以内, 想定人気が勝率順位より UR_GAP 以上低い
# 交流戦の区分 (第24節 koryu_backtest.py): 開催地の地区. 南関東4場・東海2場はそれぞれ1地区とみなす
HOST = {"佐賀": "佐賀", "名古屋": "東海", "園田": "兵庫", "姫路": "兵庫", "大井": "南関", "川崎": "南関", "船橋": "南関",
        "浦和": "南関", "水沢": "岩手", "盛岡": "岩手", "笠松": "東海", "金沢": "金沢", "門別": "北海道", "高知": "高知"}
AREA = {"大井": "南関", "川崎": "南関", "船橋": "南関", "浦和": "南関", "愛知": "東海", "笠松": "東海"}


def other_area(x, course):
    """調教師所属が開催地と別の地区の地方馬か (中央所属は含まない)"""
    a = x["調教師所属"].map(lambda s: AREA.get(s, s))
    return (a != HOST.get(course)) & (x["調教師所属"] != "JRA")


def race_kind(x, course):
    """中央所属が2頭以上 = 中央交流, それ以外で他地区の馬が2頭以上 = 地方間交流, ほかは通常戦"""
    if (x["調教師所属"] == "JRA").sum() >= 2:
        return "中央交流"
    return "地方間交流" if other_area(x, course).sum() >= 2 else "通常戦"


def class_features(h):
    """クラス (級・段・組) の変化. h は馬ID・日付順に並べ替える"""
    h = h.sort_values(["馬ID", "日付", "レース番号"]).copy()
    h["所属地"] = h["競馬場"].map(T.HOME).fillna(h["競馬場"])
    h["クラス"] = [C.parse_class(n, c) for n, c in zip(h["レース名"], h["競馬場"])]
    g = h.groupby("馬ID")
    for i in range(1, 5):
        h[f"{i}走前クラス"] = g["クラス"].shift(i)
        h[f"{i}走前着順"] = g["着順"].shift(i)
        h[f"{i}走前所属地"] = g["所属地"].shift(i)
    same = h["1走前所属地"] == h["所属地"]
    h["前走からの変化"] = [C.change_label(p, c) if s else "比較できない"
                       for p, c, s in zip(h["1走前クラス"], h["クラス"], same)]
    tiers = []
    for rec in h[["クラス", "所属地"] + [f"{i}走前クラス" for i in (2, 3, 4)] + [f"{i}走前所属地" for i in (2, 3, 4)]].itertuples(index=False):
        cur, home = rec[0], rec[1]
        val = np.nan
        if isinstance(cur, tuple):
            for p, ph in zip(rec[2:5][::-1], rec[5:8][::-1]):  # 4走前 → 2走前の順
                if isinstance(p, tuple) and ph == home:
                    val = (p[0] * 10 + p[1]) - (cur[0] * 10 + cur[1])
                    if val == 0 and p[2] is not None and cur[2] is not None:
                        val = (p[2] - cur[2]) / 100  # 同じ段なら組の差を小さな値で
                    break
        tiers.append(val)
    h["4走前からの段の変化"] = tiers
    h["近4走平均着順"] = h[[f"{i}走前着順" for i in range(1, 5)]].mean(axis=1)
    # 中央からの転入: 地方 (データ内) の出走が0〜2走なのに, 通算出走 (全成績) がそれより3走以上多い馬.
    # 直近3か月でモデルの勝率より実際は約1.3倍勝っていた (front_transfer.py)
    n, _ = T.rec_n(h["全成績"])
    h["地方出走数"] = g.cumcount().values
    h["中央からの転入"] = ((h["地方出走数"] <= 2) & (n.values >= h["地方出走数"] + 3)).astype(float)
    # 直近2走のどちらかで4角3番手以内 (7番人気以下では人気より走る: 勝ちA/E 1.10)
    p1, p2 = g["4角位置"].shift(1), g["4角位置"].shift(2)
    h["直近2走で前"] = ((p1 <= 3) | (p2 <= 3)).astype(float)
    return h


def build(d, jr_course, jr_all):
    f = P.build_features(d, jr_course)
    a = A.features(d, jr_all)
    for c in ["平均走破タイム", "平均上がり3F", "先行率", "位置取り"]:
        f[c] = a[c].values
    f["前走勝ち馬との差(秒)"] = d["前走勝ち馬との差"].fillna(2.0).values
    f["昇級(賞金比,対数)"] = np.log(d["1着賞金(円)"] / d["前走賞金"]).fillna(0).clip(-1.5, 1.5).values
    f["調教師3着内率"] = d["調教師率"].values
    n, _ = T.rec_n(d["全成績"])
    w = d["全成績"].astype(str).str.extract(r"^(\d+)-")[0].astype(float)
    f["通算勝率"] = ((w.fillna(0) + 0.5) / (n.fillna(0) + 5)).values
    f["近4走平均着順"] = d["近4走平均着順"].fillna(6).values
    ch = d["前走からの変化"]
    for k in ["昇級(段・級が上がる)", "組上がり", "組下がり", "降級(段・級が下がる)", "同じ組"]:
        f["前走から:" + k] = (ch == k).astype(float).values
    up = ch.isin(["組上がり", "昇級(段・級が上がる)"])
    f["前走4着以下で組・段が上がった"] = (up & (d["前走着順"] >= 4)).astype(float).values
    f["前走1着で組・段が上がった"] = (up & (d["前走着順"] == 1)).astype(float).values
    t = d["4走前からの段の変化"]
    f["4走前から上がった"] = (t > 0).astype(float).values
    f["4走前から下がった"] = (t < 0).astype(float).values
    f["好成績で組・段が下がった"] = ((t < 0) & (d["近4走平均着順"] <= 2.5)).astype(float).values
    f["中央からの転入"] = d["中央からの転入"].values
    return f


def main():
    data, hfile, rfile, course = sys.argv[1:5]
    h, _, _ = T.load()
    e, r = P.load_entries(hfile, rfile, course)
    h = pd.concat([h, e], ignore_index=True)
    h["予想対象"] = h["予想対象"].astype("boolean").fillna(False).astype(bool)
    races = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][1]], ignore_index=True).drop_duplicates(KEY)
    names = pd.concat([races[KEY + ["レース名"]], r[KEY + ["レース名"]]]).drop_duplicates(KEY)
    h = h.drop(columns=["レース名"], errors="ignore").merge(names, on=KEY, how="left")
    h = T.features(h)
    h = A.add_runs(h)
    h = X.extra(h)
    prize = r.set_index(KEY)["1着賞金(円)"]
    m = h["予想対象"]
    h.loc[m, "1着賞金(円)"] = prize.reindex(pd.MultiIndex.from_frame(h.loc[m, KEY])).values
    h = class_features(h)
    jr_c = P.jockey_rate(h, course)
    jr_a = A.jockey_rates(h)
    q_idx = h.loc[h["年"] >= 2024, "指数"].quantile([1 / 3, 2 / 3]).values
    d = h[(h["競馬場"] == course) & (h["年"] >= 2024)].copy()
    d.index = range(len(d))
    f_old = P.build_features(d, jr_c)
    f_new = build(d, jr_c, jr_a)

    hist = d[~d["予想対象"]]
    cut = hist["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = hist[hist["日付"] <= cut], hist[hist["日付"] > cut]
    rows = []
    for label, f in [("旧モデル (predict_race)", f_old), ("改良版", f_new)]:
        Xt, Mt, Ot, _ = P.pad(tr, f.loc[tr.index])
        b, mu, sd = P.fit(Xt, Mt, Ot)
        Xv, Mv, Ov, _ = P.pad(va, f.loc[va.index])
        Pw = P.win_probs(Xv, Mv, b, mu, sd)
        ok = Ov[:, 0] >= 0
        top = Pw.argmax(1)
        rows.append({"モデル": label, "検証レース数": int(ok.sum()),
                     "本命の勝率%": (top == Ov[:, 0])[ok].mean() * 100,
                     "本命の3着内率%": np.mean([t in o for t, o in zip(top[ok], Ov[ok])]) * 100,
                     "勝ち馬の対数尤度": np.log(Pw[np.arange(len(Pw)), Ov[:, 0]][ok]).mean()})
    fav = va[va["人気"] == 1]
    rows.append({"モデル": "参考: 1番人気", "本命の勝率%": (fav["着順"] == 1).mean() * 100,
                 "本命の3着内率%": (fav["着順"] <= 3).mean() * 100})
    vt = pd.DataFrame(rows).set_index("モデル")
    print(f"== 検証 {course} {va['日付'].min():%Y-%m}〜{va['日付'].max():%Y-%m}\n", vt.round(3).to_string())

    # ---- 過小評価候補: 勝率順位が上位なのに, 想定人気 (データから予測した人気順) が3つ以上低い馬 ----
    d["人気c"] = d["人気"].clip(upper=10)
    d["頭数帯"] = pd.cut(d["頭数"], [0, 8, 10, 12, 99])
    gk = d.groupby(["年", "人気c", "頭数帯"], observed=True)
    d["期待"] = gk["top3"].transform("mean")
    d["期待勝"] = gk["着順"].transform(lambda s: (s == 1).mean())

    def ranks(sub, f, bw, mw, sw, bp, mp, sp):
        Xs, Ms, _, rix = P.pad(sub, f.loc[sub.index], with_order=False)
        pw = P.win_probs(Xs, Ms, bw, mw, sw)
        pp = P.win_probs(Xs, Ms, bp, mp, sp)
        out = pd.DataFrame(index=sub.index, columns=["勝率", "勝率順", "想定人気"], dtype=float)
        for i, ix in enumerate(rix):
            k = len(ix)
            out.loc[ix, "勝率"] = pw[i, :k]
            out.loc[ix, "勝率順"] = (-pw[i, :k]).argsort().argsort() + 1
            out.loc[ix, "想定人気"] = (-pp[i, :k]).argsort().argsort() + 1
        return out

    def underrated(sub, r, rank_max=UR_RANK, gap=UR_GAP):
        kesi = T.kesi(sub)
        return (r["勝率順"] <= rank_max) & (r["想定人気"] >= r["勝率順"] + gap) & ~kesi

    Xt, Mt, Ot, _ = P.pad(tr, f_new.loc[tr.index])
    bw, mw, sw = P.fit(Xt, Mt, Ot)
    Xt, Mt, Ot, _ = P.pad(F.order_by(tr, "人気"), f_new.loc[tr.index])
    bp, mp, sp = P.fit(Xt, Mt, Ot)
    rv = ranks(va, f_new, bw, mw, sw, bp, mp, sp)
    vr = d.loc[va.index, ["競馬場", "競走年月日", "レース番号", "馬番", "人気", "着順", "top3", "期待", "期待勝"]].join(rv)
    vr["消し"] = T.kesi(va).values
    vr.to_csv(T.OUT / f"pred2_underrated_rows_{course}.csv", index=False, encoding="utf-8-sig")
    fl = underrated(va, rv)
    x = d.loc[va.index][fl]
    ur = pd.Series({"該当頭数": len(x), "平均の実際の人気": x["人気"].mean(),
                    "勝率%": (x["着順"] == 1).mean() * 100, "3着内率%": (x["top3"] == 1).mean() * 100,
                    "勝ちA/E(実際の人気比)": (x["着順"] == 1).sum() / x["期待勝"].sum(),
                    "3着内A/E(実際の人気比)": x["top3"].sum() / x["期待"].sum()})
    print("\n== 過小評価候補の検証 (直近3か月)\n", ur.round(2).to_string())
    ur.to_csv(T.OUT / f"pred2_underrated_valid_{course}.csv", encoding="utf-8-sig")

    Xt, Mt, Ot, _ = P.pad(F.order_by(hist, "人気"), f_new.loc[hist.index])
    bp, mp, sp = P.fit(Xt, Mt, Ot)
    Xt, Mt, Ot, _ = P.pad(hist, f_new.loc[hist.index])
    b, mu, sd = P.fit(Xt, Mt, Ot)
    tg = d[d["予想対象"]].sort_values(KEY + ["馬番"])
    rt = ranks(tg, f_new, b, mu, sd, bp, mp, sp)
    # 期待値 (想定): モデル勝率 × 想定人気から見込んだオッズ. 想定人気はモデルの値を使う (ev_select.py の検証で
    # 実際の人気で計算するより成績が良かった: 期待値1.0以上の単勝回収率 85% 対 74%)
    q_mk, rate_mk = EV.market_table(h)
    rt["期待値"], rt["想定オッズ"] = EV.expected_value(tg, rt["勝率"], rt["想定人気"], q_mk, rate_mk)
    # 実績とのずれを補正 (ev_calibrate.py): 人気薄ほど モデル勝率も想定オッズも高く出すぎる
    pc, oc = EC.load_fit()
    rt["補正勝率"], rt["補正オッズ"], rt["補正期待値"] = EC.calibrated(rt["勝率"], rt["想定人気"], rt["想定オッズ"], pc, oc)
    # 実際の人気が分かっていればそちらを使う
    has_pop = tg["人気"].notna() & (tg.groupby(KEY)["人気"].transform("nunique") > 1)
    rt.loc[has_pop, "想定人気"] = tg.loc[has_pop, "人気"]
    rt["過小評価候補"] = underrated(tg, rt)
    Xv, Mv, _, races_ix = P.pad(tg, f_new.loc[tg.index], with_order=False)
    Pw = P.win_probs(Xv, Mv, b, mu, sd)
    out = []
    for i, ix in enumerate(races_ix):
        x = tg.loc[ix]
        w = Pw[i, :len(ix)]
        top3, tri = P.pl_probs(w)
        nos = x["馬番"].astype(int).values
        # ◎○ は勝率順, ▲△△ 以降は 2〜3着補正 (第20節) を掛けた順 (place_backtest.py で ３連単・３連複の回収率が改善)
        o = list((-w).argsort())
        adj = w * PB.mult(x.reset_index(drop=True), q_idx).values
        # 消しの馬は ▲△ に入れない. 地方間交流では消しを使わない (第24節: 3着内率14%で通常戦の倍)
        rk = race_kind(x, course)
        oth = other_area(x, course).values
        kesi = (T.kesi(x) & (rk != "地方間交流")).values
        order = o[:2] + sorted(o[2:], key=lambda i: (kesi[i], -adj[i]))
        kind, buy, npts, hit = P.suggest(w, tri, nos, order)
        rank = np.empty(len(w), dtype=int)
        rank[order] = np.arange(len(w))
        marks = np.array(["◎", "○", "▲", "△", "△"] + [""] * 30)[rank]
        ff = f_new.loc[ix]
        for j, (_, row) in enumerate(x.iterrows()):
            flags = []
            if ff.iloc[j]["前走4着以下で組・段が上がった"]:
                flags.append("前走4着以下で組上がり(割引)")
            if ff.iloc[j]["前走1着で組・段が上がった"]:
                flags.append("前走1着で昇級(加点)")
            if ff.iloc[j]["好成績で組・段が下がった"]:
                flags.append("好成績で降級(加点)")
            if row["中央からの転入"]:
                flags.append("中央からの転入(人気を重視)")
            if row["直近2走で前"] and rt.loc[ix[j], "想定人気"] >= 7:
                flags.append("人気薄の先行馬(3着候補)")
            if kesi[j]:
                flags.append("消し条件")
            if rank[j] == 0 and rk == "中央交流":
                # 第24節: 中央交流では ◎の勝率24% (1番人気38%). ◎が地方の馬なら 19%, ルールの買い目の回収率26%
                flags.append("中央交流(印は参考程度・人気を重視)" + ("/◎が地方の馬(割引)" if row["調教師所属"] != "JRA" else ""))
            if rank[j] == 0 and rk == "地方間交流":
                # 第24節: ◎が他地区の馬なら ◎の勝率30%・単勝回収率63% (1番人気は43%)
                flags.append("地方間交流" + ("/◎が他地区の馬(1番人気を重視)" if oth[j] else "/◎が地元の馬"))
            out.append({
                "レース": int(row["レース番号"]), "レース名": row["レース名"], "距離": row["距離"], "印": marks[j],
                "馬番": int(row["馬番"]), "馬名": row["馬名"], "騎手": row["騎手名"],
                "勝率%": round(w[j] * 100, 1), "3着内率%": round(top3[j] * 100, 1),
                "前走": f"{'' if pd.isna(row['前走着順']) else int(row['前走着順'])}着"
                        f"({'' if pd.isna(row['前走人気']) else int(row['前走人気'])}人気)",
                "クラス変化": row["前走からの変化"], "近4走平均着順": None if pd.isna(row["近4走平均着順"]) else round(row["近4走平均着順"], 1),
                "指数順": None if pd.isna(row["指数順"]) else int(row["指数順"]),
                "位置取り": None if pd.isna(row["位置取り"]) else round(row["位置取り"], 2),
                "想定人気": int(rt.loc[ix[j], "想定人気"]),
                "過小評価候補": "★" if rt.loc[ix[j], "過小評価候補"] else "",
                "期待値(補正)": None if pd.isna(rt.loc[ix[j], "補正期待値"]) else round(float(rt.loc[ix[j], "補正期待値"]), 2),
                "補正オッズ": None if pd.isna(rt.loc[ix[j], "補正オッズ"]) else round(float(rt.loc[ix[j], "補正オッズ"]), 1),
                "補正勝率%": None if pd.isna(rt.loc[ix[j], "補正勝率"]) else round(float(rt.loc[ix[j], "補正勝率"]) * 100, 1),
                "注記": " / ".join(flags),
                "買い目": f"[{kind}] {buy} ({npts}点, モデル上の的中確率 {hit*100:.0f}%)" if j == 0 else "",
            })
    o = pd.DataFrame(out)
    # 各レースの期待値上位 (消しを除き, 補正後の期待値1.0以上を高い順に3頭まで).
    # 交差検証では 1.2〜2.0 の帯が最も良く (回収率 92〜104%), 2.0超は補正後も出すぎ (71%)
    o["期待値順"] = None
    for k, g in o.groupby("レース"):
        c = g[(g["期待値(補正)"].fillna(0) >= 1.0) & ~g["注記"].str.contains("消し条件")]
        c = c.sort_values("期待値(補正)", ascending=False).head(3)
        o.loc[c.index, "期待値順"] = range(1, len(c) + 1)
    date = int(tg["競走年月日"].iloc[0])
    o.to_csv(T.OUT / f"pred2_{date}_{course}.csv", index=False, encoding="utf-8-sig")
    vt.to_csv(T.OUT / f"pred2_{date}_{course}_valid.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for k, g in o.groupby("レース"):
        print(f"\n## {k}R {g['レース名'].iloc[0]} {g['距離'].iloc[0]}m")
        print(g.sort_values("勝率%", ascending=False).drop(columns=["レース", "レース名", "距離", "買い目"]).to_string(index=False))
        print("買い目:", g["買い目"].iloc[0])
        e = g[g["期待値順"].notna()].sort_values("期待値順")
        if len(e):
            print("期待値上位:", " / ".join(f"{r.馬番} {r.馬名} (期待値{r['期待値(補正)']}, 補正オッズ{r.補正オッズ}倍)" for _, r in e.iterrows()))
        u = g[g["過小評価候補"] == "★"]
        if len(u):
            print("過小評価候補:", " / ".join(f"{r.馬番} {r.馬名} (勝率{r['勝率%']}%, 想定{r.想定人気}番人気)" for _, r in u.iterrows()))


if __name__ == "__main__":
    main()
