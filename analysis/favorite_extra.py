"""1番人気の説明に, まだ使っていないデータ項目を足すとどこまで説明できるか.

favorite_drivers.py の特徴量に, 次を追加して比べる:
  調教師の3着内率 (前年, 同場), 斤量 (レース内の相対値)・見習い騎手の減量, 前走の勝ち馬とのタイム差,
  昇級・降級 (1着賞金の前走比), 乗り替わりでの騎手の格の上下, 馬体重, 通算勝率・通算勝利数,
  父馬の3着内率 (前年まで), 前走が同じ競馬場か
検証: 直近3か月で「モデルの1位 = 実際の1番人気」の割合と, 人気順位の対数尤度.
  あわせて追加項目の, 人気への影響と人気を差し引いた後の影響を出す.

出力: analysis/out/favx_*.csv
"""
import numpy as np
import pandas as pd

import ability_index as A
import favorite_drivers as F
import predict_race as P
import trifecta_factors as T

KEY = T.KEY


def rate_prev_year(h, by, k=30):
    """by ごとの3着内率 (競馬場別, 前年1年分, k 走で場平均へ縮約) を各行に付ける"""
    x = h[h["top3"].notna()]
    base = x.groupby("競馬場")["top3"].mean()
    g = x.groupby(["競馬場", "年", by])["top3"].agg(["sum", "size"]).reset_index()
    g["年"] += 1  # 翌年の行に使う
    m = h[["競馬場", "年", by]].merge(g, on=["競馬場", "年", by], how="left")
    b = h["競馬場"].map(base).values
    return ((m["sum"].fillna(0).values + k * b) / (m["size"].fillna(0).values + k))


def extra(h):
    h = h.sort_values(["馬ID", "日付", "レース番号"]).copy()
    r = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][1]], ignore_index=True).drop_duplicates(KEY)
    h = h.merge(r[KEY + ["1着賞金(円)"]], on=KEY, how="left")
    h = h.sort_values(["馬ID", "日付", "レース番号"])
    g = h.groupby("馬ID")
    h["勝ち馬との差"] = (h["秒"] - h.groupby(KEY)["秒"].transform("min")).clip(0, 5)
    h["前走勝ち馬との差"] = g["勝ち馬との差"].shift(1)
    h["前走賞金"] = g["1着賞金(円)"].shift(1)
    h["前走騎手名"] = g["騎手名"].shift(1)
    h["斤量"] = pd.to_numeric(h["負担重量"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    h["減量騎手"] = h["負担重量"].astype(str).str.contains(r"[☆▲◇△★]").astype(float)
    h["調教師率"] = rate_prev_year(h, "調教師")
    h["騎手率"] = rate_prev_year(h, "騎手名")
    jr = h[["競馬場", "年", "騎手名", "騎手率"]].drop_duplicates(["競馬場", "年", "騎手名"])
    prev = h[["競馬場", "年", "前走騎手名"]].merge(
        jr.rename(columns={"騎手名": "前走騎手名", "騎手率": "前走騎手率"}), on=["競馬場", "年", "前走騎手名"], how="left")
    h["前走騎手率"] = prev["前走騎手率"].values
    h["父率"] = rate_prev_year(h, "父馬名", k=100)
    return h


def build(d, jr):
    f = F.build(d, jr)
    grp = [d[k] for k in KEY]
    f["調教師3着内率"] = d["調教師率"].values
    f["斤量(レース内差)"] = (d["斤量"] - d["斤量"].groupby(grp).transform("mean")).fillna(0).clip(-5, 5)
    f["減量騎手"] = d["減量騎手"]
    f["前走勝ち馬との差(秒)"] = d["前走勝ち馬との差"].fillna(2.0)
    f["昇級(賞金比,対数)"] = np.log(d["1着賞金(円)"] / d["前走賞金"]).fillna(0).clip(-1.5, 1.5)
    f["騎手の格上げ"] = (d["騎手率"] - d["前走騎手率"]).fillna(0)
    f["馬体重"] = d["馬体重"].fillna(d["馬体重"].mean())
    n, _ = T.rec_n(d["全成績"])
    w = d["全成績"].astype(str).str.extract(r"^(\d+)-")[0].astype(float)
    f["通算勝率"] = ((w.fillna(0) + 0.5) / (n.fillna(0) + 5)).values
    f["通算勝利数(対数)"] = np.log1p(w.fillna(0)).values
    f["父馬3着内率"] = d["父率"].values
    f["前走同場"] = (d["前走競馬場"] == d["競馬場"]).astype(float)
    return f


def evaluate(d, f, label):
    cut = d["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = d[d["日付"] <= cut], d[d["日付"] > cut]
    X, M, O, _ = P.pad(F.order_by(tr, "人気"), f.loc[tr.index])
    b, mu, sd = P.fit(X, M, O)
    X, M, O, _ = P.pad(F.order_by(va, "人気"), f.loc[va.index])
    Pw = P.win_probs(X, M, b, mu, sd)
    hit = (Pw.argmax(1) == O[:, 0]).mean() * 100
    ll = np.log(Pw[np.arange(len(Pw)), O[:, 0]]).mean()
    print(f"{label}: 1番人気の的中 {hit:.1f}%  対数尤度 {ll:.3f}")
    return {"モデル": label, "1番人気の的中%": hit, "1番人気の対数尤度": ll}


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    h = extra(h)
    jr = A.jockey_rates(h)
    d = h[(h["年"] >= 2024) & h["人気"].notna()].copy()
    d.index = range(len(d))
    f0 = F.build(d, jr)
    f1 = build(d, jr)
    rows = [evaluate(d, f0, "前回の項目"), evaluate(d, f1, "追加項目あり")]
    # 1項目ずつ足したときの伸び
    for c in [c for c in f1.columns if c not in f0.columns]:
        rows.append(evaluate(d, pd.concat([f0, f1[[c]]], axis=1), f"前回＋{c}"))
    pd.DataFrame(rows).to_csv(T.OUT / "favx_predictability.csv", index=False, encoding="utf-8-sig")

    # 係数 (全期間)
    X, M, O, _ = P.pad(F.order_by(d, "人気"), f1)
    b_pop, _, _ = P.fit(X, M, O)
    f2 = f1.copy()
    f2["人気(対数)"] = np.log(d["人気"])
    X, M, O, _ = P.pad(d, f2)
    b_adj, _, _ = P.fit(X, M, O)
    t = pd.DataFrame({"人気への影響": b_pop, "人気を差し引いた後の影響": b_adj[:-1]}, index=f1.columns)
    t["値が大きい馬は"] = np.select([t["人気を差し引いた後の影響"] <= -0.05, t["人気を差し引いた後の影響"] >= 0.05],
                              ["人気ほど走らない", "人気以上に走る"], "ほぼ人気どおり")
    t["追加項目"] = ~t.index.isin(f0.columns)
    t = t.sort_values("人気への影響", key=abs, ascending=False)
    t.to_csv(T.OUT / "favx_drivers.csv", encoding="utf-8-sig")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
