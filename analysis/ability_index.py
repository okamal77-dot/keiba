"""能力値: 平均走破タイム・平均上がり3F・脚質・位置取り・騎手との相性・距離適性のクロス分析と換算.

各馬の近5走 (レース前に分かる情報のみ) から6要素を作る:
  平均走破タイム : 走破タイムの偏差値 (競馬場×距離×馬場状態ごとの z, 速いほど+) の近5走平均
  平均上がり3F   : 上がり3Fの偏差値 (同上) の近5走平均
  脚質 (先行率)  : 近5走のうち逃げ・先行だった割合
  位置取り       : 近5走の最終コーナー位置 ÷ 頭数 の平均 (小さいほど前)
  騎手との相性   : この騎手でこの馬の3着内率 (縮約) と, 騎手の前年3着内率
  距離適性       : 当場当距離の3着内率 − 全成績の3着内率 (いずれも縮約), 近5走平均距離からの距離変更(m)

 1) 2要素ずつのクロス表 (レース内順位で区分, 3着内率と人気補正 A/E)
 2) 6要素の上位3着順位ロジットで「能力値」(偏差値, 平均50・標準偏差10) に換算し, 直近3か月で検証
 3) 出走表があれば当てはめる: python3 analysis/ability_index.py <data> [<horselist.csv> <racelist.csv> <競馬場>]

対象はばんえいを除く14場 (上がり・コーナー通過順がないため).
出力: analysis/out/ab_*.csv
"""
import sys

import numpy as np
import pandas as pd

import predict_race as P
import trifecta_factors as T

KEY = T.KEY
N_PAST = 5
FEATS = ["平均走破タイム", "平均上がり3F", "先行率", "位置取り", "コンビ3着内率", "騎手3着内率", "距離適性", "距離変更",
         "上がり×位置", "近走なし"]


def shrink(rec, k=4, prior=0.3):
    n, t3 = T.rec_n(rec)
    return (t3.fillna(0) + k * prior) / (n.fillna(0) + k), n.fillna(0)


def add_runs(h):
    """1走ごとの値と, その馬の近5走平均 (当該レースは含まない)"""
    h = h.sort_values(["馬ID", "日付", "レース番号"]).copy()
    g = h.groupby(["競馬場", "距離", "馬場G"])["上がり3F"]
    h["上がり指数"] = -(h["上がり3F"] - g.transform("mean")) / g.transform("std")
    h.loc[g.transform("size") < 30, "上がり指数"] = np.nan
    h["先行"] = h["脚質"].isin(["逃げ", "先行"]).astype(float).where(h["脚質"].notna())
    h["位置率"] = h["4角位置"] / h["頭数"]
    for c, name in [("指数", "平均走破タイム"), ("上がり指数", "平均上がり3F"), ("先行", "先行率"),
                    ("位置率", "位置取り"), ("距離", "平均距離")]:
        v = h[c].clip(-3, 3) if c in ("指数", "上がり指数") else h[c]
        h[name] = v.groupby(h["馬ID"]).transform(lambda s: s.shift(1).rolling(N_PAST, min_periods=1).mean())
    h["近走数"] = h.groupby("馬ID").cumcount().clip(upper=N_PAST)
    return h


def jockey_rates(h):
    """騎手の3着内率 (競馬場ごと, 前年1年分, 30騎乗で場平均へ縮約). 予想日は直近365日"""
    x = h[h["top3"].notna()]
    out = {}
    for c, xc in x.groupby("競馬場"):
        base = xc["top3"].mean()
        for y in xc["年"].unique():
            g = xc[xc["年"] == y - 1].groupby("騎手名")["top3"].agg(["sum", "size"])
            out[(c, y)] = (((g["sum"] + 30 * base) / (g["size"] + 30)).to_dict(), base)
        w = xc[xc["日付"] > xc["日付"].max() - pd.Timedelta(days=365)]
        g = w.groupby("騎手名")["top3"].agg(["sum", "size"])
        out[(c, None)] = (((g["sum"] + 30 * base) / (g["size"] + 30)).to_dict(), base)
    return out


def features(d, jr):
    f = pd.DataFrame(index=d.index)
    grp = [d[k] for k in KEY]
    for c in ["平均走破タイム", "平均上がり3F", "先行率", "位置取り"]:
        f[c] = d[c].fillna(d[c].groupby(grp).transform("mean")).fillna(d[c].mean())
    f["コンビ3着内率"], _ = shrink(d["騎手成績"])
    keys = [(c, None if t else y) for c, t, y in zip(d["競馬場"], d["予想対象"], d["年"])]
    f["騎手3着内率"] = [jr.get(k, jr.get((k[0], None), ({}, 0.3)))[0].get(j, jr.get(k, ({}, 0.3))[1])
                     for k, j in zip(keys, d["騎手名"])]
    dist, _ = shrink(d["うち当距離成績"])
    allr, _ = shrink(d["全成績"])
    f["距離適性"] = dist - allr
    f["距離変更"] = (d["距離"] - d["平均距離"]).abs().fillna(0).clip(0, 600) / 100
    f["上がり×位置"] = (f["平均上がり3F"] - f["平均上がり3F"].mean()) * (0.5 - f["位置取り"])
    f["近走なし"] = (d["近走数"] == 0).astype(float)
    return f[FEATS]


def rank_group(s, d, asc=False, labels=("1-3位", "4-6位", "7位~")):
    r = s.groupby([d[k] for k in KEY]).rank(ascending=asc, method="min")
    return pd.cut(r, [0, 3, 6, 99], labels=list(labels))


def cross(d, a, b, name):
    """2要素のクロス: 3着内率 と A/E"""
    g = d.groupby([a, b], observed=True)
    t3 = (g["top3"].mean() * 100).unstack()
    ae = (g["top3"].sum() / g["期待"].sum()).unstack()
    n = g.size().unstack()
    out = pd.concat({"3着内率": t3, "A/E": ae, "頭数": n}, axis=1)
    out.to_csv(T.OUT / f"ab_cross_{name}.csv", encoding="utf-8-sig")
    print(f"\n== {name}\n", out.round(2).to_string())


def main():
    args = sys.argv[1:]
    h, _, _ = T.load()
    h["予想対象"] = False
    if len(args) >= 4:
        e, _ = P.load_entries(args[1], args[2], args[3])
        h = pd.concat([h, e], ignore_index=True)
        h["予想対象"] = h["予想対象"].astype("boolean").fillna(False).astype(bool)
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = add_runs(h)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待"] = h.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)["top3"].transform("mean")
    jr = jockey_rates(h)
    d = h[(h["年"] >= 2024)].copy()
    f = features(d, jr)
    hist = d[~d["予想対象"]]

    # ---- 1) クロス分析 (レース内順位で3区分) ----
    c = pd.DataFrame(index=hist.index)
    c["走破タイム"] = rank_group(f.loc[hist.index, "平均走破タイム"], hist)
    c["上がり3F"] = rank_group(f.loc[hist.index, "平均上がり3F"], hist)
    c["位置取り"] = pd.cut(hist["位置取り"], [-1, 0.35, 0.65, 2], labels=["前(~35%)", "中(35-65%)", "後(65%~)"])
    c["脚質"] = pd.cut(hist["先行率"], [-1, 0.2, 0.6, 2], labels=["先行率~20%", "20-60%", "60%~"])
    cb, _ = shrink(hist["騎手成績"])
    nn, _ = T.rec_n(hist["騎手成績"])
    c["騎手相性"] = np.select([nn.fillna(0) < 3, cb >= 0.45, cb <= 0.2], ["3戦未満", "好相性", "低相性"], "普通")
    c["距離適性"] = pd.cut(f.loc[hist.index, "距離適性"], [-1, -0.08, 0.08, 1], labels=["当距離で劣る", "差なし", "当距離で優る"])
    c["距離変更"] = pd.cut(f.loc[hist.index, "距離変更"], [-0.01, 1, 2, 99], labels=["±100m以内", "100-200m", "200m超"])
    x = hist.join(c, rsuffix="_c")
    for a, b in [("走破タイム", "上がり3F"), ("位置取り", "上がり3F"), ("脚質", "走破タイム"),
                 ("騎手相性", "走破タイム"), ("距離適性", "走破タイム"), ("距離変更", "位置取り")]:
        cross(x, a + ("_c" if a in hist.columns else ""), b + ("_c" if b in hist.columns else ""), f"{a}×{b}")

    # ---- 2) 能力値への換算 ----
    cut = hist["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = hist[hist["日付"] <= cut], hist[hist["日付"] > cut]
    X, M, O, _ = P.pad(tr, f.loc[tr.index])
    b, mu, sd = P.fit(X, M, O)
    coef = pd.Series(b, index=FEATS)
    print("\n== 係数 (標準化後)\n", coef.round(3).to_string())
    coef.to_csv(T.OUT / "ab_coef.csv", encoding="utf-8-sig")
    score = ((f - mu) / sd) @ b
    s_mu, s_sd = score.loc[tr.index].mean(), score.loc[tr.index].std()
    d["能力値"] = 50 + 10 * (score - s_mu) / s_sd
    d["能力値順"] = d.groupby(KEY)["能力値"].rank(ascending=False, method="first")

    v = d.loc[va.index]
    print(f"\n== 検証 {v['日付'].min():%Y-%m}〜{v['日付'].max():%Y-%m}  {v.groupby(KEY).ngroups}レース")
    q = v.groupby(pd.cut(v["能力値順"], [0, 1, 2, 3, 5, 99], labels=["1位", "2位", "3位", "4-5位", "6位~"]), observed=True)
    t = pd.DataFrame({"頭数": q.size(), "勝率": q["着順"].apply(lambda s: (s == 1).mean() * 100),
                      "3着内率": q["top3"].mean() * 100, "A/E(人気補正)": q["top3"].sum() / q["期待"].sum()})
    pop = v.groupby(pd.cut(v["人気"], [0, 1, 2, 3, 5, 99], labels=["1位", "2位", "3位", "4-5位", "6位~"]), observed=True)
    t["参考: 人気順の勝率"] = pop["着順"].apply(lambda s: (s == 1).mean() * 100)
    t["参考: 人気順の3着内率"] = pop["top3"].mean() * 100
    print(t.round(2).to_string())
    t.to_csv(T.OUT / "ab_valid_rank.csv", encoding="utf-8-sig")
    # 6番人気以下で能力値上位
    lo = v[v["人気"] >= 6]
    q = lo.groupby(pd.cut(lo["能力値順"], [0, 3, 5, 99], labels=["能力値1-3位", "4-5位", "6位~"]), observed=True)
    t = pd.DataFrame({"頭数": q.size(), "3着内率": q["top3"].mean() * 100, "A/E(人気補正)": q["top3"].sum() / q["期待"].sum()})
    print("\n6番人気以下\n", t.round(2).to_string())
    t.to_csv(T.OUT / "ab_valid_longshot.csv", encoding="utf-8-sig")
    # 競馬場別: 能力値1位の勝率・3着内率と1番人気
    g = v[v["能力値順"] == 1].groupby("競馬場")
    fav = v[v["人気"] == 1].groupby("競馬場")
    t = pd.DataFrame({"能力値1位 勝率": g["着順"].apply(lambda s: (s == 1).mean() * 100),
                      "能力値1位 3着内率": g["top3"].mean() * 100,
                      "1番人気 勝率": fav["着順"].apply(lambda s: (s == 1).mean() * 100),
                      "1番人気 3着内率": fav["top3"].mean() * 100}).reindex(T.ORDER).dropna()
    print("\n競馬場別\n", t.round(1).to_string())
    t.to_csv(T.OUT / "ab_valid_course.csv", encoding="utf-8-sig")

    # ---- 3) 出走表への当てはめ (全期間で再学習) ----
    if len(args) >= 4:
        X, M, O, _ = P.pad(hist, f.loc[hist.index])
        b, mu, sd = P.fit(X, M, O)
        score = ((f - mu) / sd) @ b
        s_mu, s_sd = score.loc[hist.index].mean(), score.loc[hist.index].std()
        d["能力値"] = 50 + 10 * (score - s_mu) / s_sd
        tg = d[d["予想対象"]].copy()
        tg["能力値順"] = tg.groupby(KEY)["能力値"].rank(ascending=False, method="first").astype(int)
        cols = {"平均走破タイム": "走破T", "平均上がり3F": "上がり", "先行率": "先行率", "位置取り": "位置",
                "コンビ3着内率": "コンビ", "騎手3着内率": "騎手", "距離適性": "距離適性", "距離変更": "距離変更"}
        out = tg[["レース番号", "馬番", "馬名", "騎手名", "距離", "能力値", "能力値順", "近走数"]].copy()
        for k, nm in cols.items():
            out[nm] = f.loc[tg.index, k].values
        out["平均距離"] = tg["平均距離"].round(0)
        out = out.sort_values(["レース番号", "能力値順"])
        date = int(tg["競走年月日"].iloc[0])
        out.to_csv(T.OUT / f"ab_{date}_{args[3]}.csv", index=False, encoding="utf-8-sig")
        pd.set_option("display.width", 250)
        for k, gg in out.groupby("レース番号"):
            print(f"\n## {k}R {int(gg['距離'].iloc[0])}m")
            print(gg.drop(columns=["レース番号", "距離"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
