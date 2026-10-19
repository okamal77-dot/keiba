"""穴馬（6番人気以下）の能力値と回収率の検証（2023〜2026年9月・地方15場）.

trifecta_factors.py の特徴量を使い, 穴馬を「好材料数」などで分けたときの
 - 3着内率と A/E (人気から見た期待との比)
 - 単勝・複勝の回収率 (各馬に100円ずつ)
 - ３連単の追加買い目の回収率
を集計する.

３連単は「1着 = 1〜2番人気, 2着 = 1〜4番人気」を軸に, 3着にその穴馬を加えたときに
増える6点 (= 2 × 3) の払戻 ÷ 600円 を, その穴馬の「3着追加の回収率」とした.

好材料 (各1点, 計5点):
  近3走ベスト指数がレース内3位以内 / 前走上がり3F 3位以内 / 前走脚質が先行か差し /
  馬体重増減 -9〜+3kg / 低相性コンビ(3戦以上・3着内率20%以下)ではない
消し条件: 近3走ベスト指数がレース内6位以下 かつ 前走追込

入力: trifecta_factors.py と同じ data/ 配下のファイル
出力: analysis/out/ls_*.csv
"""
import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY
FIRST, SECOND = 2, 4  # ３連単の軸: 1着 = 1〜2番人気, 2着 = 1〜4番人気
N_ADD = FIRST * (SECOND - 1)  # 3着に穴馬を1頭加えると増える点数


def payouts():
    """払戻の生データ (同着行を含む) から 単勝・複勝・３連単 を縦持ちにする"""
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True)
    tan = raw[KEY + ["単勝組番", "単勝払戻金（円）"]].dropna()
    tan.columns = KEY + ["馬番", "単勝払戻"]
    fk = []
    for i in (1, 2, 3):
        x = raw[KEY + [f"複勝組番{i}", f"複勝払戻金{i}（円）"]].dropna()
        x.columns = KEY + ["馬番", "複勝払戻"]
        fk.append(x)
    fuku = pd.concat(fk)
    tri = raw[KEY + ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]].dropna()
    tri.columns = KEY + ["b1", "b2", "b3", "３連単払戻"]
    out = []
    for t in (tan, fuku, tri):
        t = t.drop_duplicates()
        for c in t.columns[3:]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        out.append(t)
    return out


def prepare():
    h, r, p = T.load()
    h = T.features(h)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待"] = h.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)["top3"].transform("mean")
    tan, fuku, tri = payouts()
    h = (h.merge(tan.groupby(KEY + ["馬番"], as_index=False)["単勝払戻"].sum(), on=KEY + ["馬番"], how="left")
          .merge(fuku.groupby(KEY + ["馬番"], as_index=False)["複勝払戻"].sum(), on=KEY + ["馬番"], how="left"))
    h[["単勝払戻", "複勝払戻"]] = h[["単勝払戻", "複勝払戻"]].fillna(0)
    # ３連単: 1着が1〜2番人気, 2着が1〜4番人気の的中組番について, 3着馬に払戻を付ける
    pop = h.set_index(KEY + ["馬番"])["人気"]
    t = tri.copy()
    for k in ("b1", "b2"):
        t["人気" + k] = pop.reindex(pd.MultiIndex.from_frame(t[KEY + [k]].rename(columns={k: "馬番"}))).values
    t = t[(t["人気b1"] <= FIRST) & (t["人気b2"] <= SECOND)]
    t3 = t.groupby(KEY + ["b3"], as_index=False)["３連単払戻"].sum().rename(columns={"b3": "馬番", "３連単払戻": "3着追加払戻"})
    h = h.merge(t3, on=KEY + ["馬番"], how="left")
    h["3着追加払戻"] = h["3着追加払戻"].fillna(0)

    h["3着追加払戻率"] = h["3着追加払戻"] / N_ADD
    h["人気区分"] = pd.cut(h["人気"], [0, 1, 2, 3, 5, 7, 9, 99],
                        labels=["1番", "2番", "3番", "4-5番", "6-7番", "8-9番", "10番~"])
    a = h[h["人気"] >= 6].copy()
    a["好材料数"] = ((a["指数順"] <= 3).astype(int) + (a["前走上がり順"] <= 3).astype(int)
                  + a["前走脚質"].isin(["先行", "差し"]).astype(int)
                  + a["馬体重増減"].between(-9, 3).astype(int)
                  + (~((a["コンビ数"] >= 3) & (a["コンビ3着内"] / a["コンビ数"] <= 0.2))).astype(int))
    a["消し"] = (a["指数順"] >= 6) & (a["前走脚質"] == "追込")
    a["区分"] = np.select([a["消し"], a["好材料数"] >= 4, a["好材料数"] == 3],
                        ["消し(指数6位~かつ前走追込)", "好材料4以上", "好材料3"], "その他")
    a["指数区分"] = pd.cut(a["指数順"], [0, 3, 5, 99], labels=["指数1-3位", "指数4-5位", "指数6位~"]
                        ).cat.add_categories("データなし").fillna("データなし")
    a["人気帯"] = pd.cut(a["人気"], [5, 7, 9, 99], labels=["6-7番", "8-9番", "10番~"])
    a["全場"] = "全場"
    return h, a


def summary(a, keys):
    g = a.groupby(keys, observed=True)
    t = pd.DataFrame({
        "頭数": g.size(),
        "3着内率": g["top3"].mean() * 100,
        "A/E": g["top3"].sum() / g["期待"].sum(),
        "単勝回収率": g["単勝払戻"].mean(),
        "複勝回収率": g["複勝払戻"].mean(),
        "3着追加回収率": g["3着追加払戻"].sum() / (g.size() * N_ADD * 100) * 100,
        "3着追加的中率": g["3着追加払戻"].apply(lambda s: (s > 0).mean() * 100),
        # 回収率の標準誤差 (配当の裾が長いので目安)
        "単勝SE": g["単勝払戻"].std() / np.sqrt(g.size()),
        "複勝SE": g["複勝払戻"].std() / np.sqrt(g.size()),
        "3着追加SE": g["3着追加払戻率"].std() / np.sqrt(g.size()),
    })
    return t


def three_groups(a):
    """好材料3以上 / その他 / 消し の3区分で, 頭数・3着内馬・3着追加的中の割合と回収率 (帯広は除く)"""
    b = a[a["競馬場"] != "帯広ば"].copy()  # ばんえいは上がり・脚質がなく好材料を数えられない
    b["3区分"] = np.select([b["消し"], b["好材料数"] >= 3], ["消し", "好材料3以上"], "その他")
    rows = []
    for y in ["全期間"] + T.YEARS:
        x = b if y == "全期間" else b[b["年"] == y]
        g = x.groupby("3区分")
        hit = x["3着追加払戻"] > 0
        t = pd.DataFrame({
            "頭数の割合": g.size() / len(x) * 100,
            "3着内馬の割合": g["top3"].sum() / x["top3"].sum() * 100,
            "3着追加的中の割合": hit.groupby(x["3区分"]).sum() / hit.sum() * 100,
            "単勝回収率": g["単勝払戻"].mean(),
            "複勝回収率": g["複勝払戻"].mean(),
            "3着追加回収率": g["3着追加払戻"].sum() / (g.size() * N_ADD * 100) * 100,
        })
        t.index = pd.MultiIndex.from_product([[y], t.index], names=["年", "3区分"])
        rows.append(t)
    return pd.concat(rows)


def save(name, df):
    df.to_csv(T.OUT / f"{name}.csv", encoding="utf-8-sig")
    print(f"\n== {name}\n", df.round(2).to_string())


def main():
    h, a = prepare()
    print("穴馬(6番人気以下)", len(a), "頭")
    h["全場"] = "全場"
    save("ls_by_popularity", summary(h, ["人気区分"]))  # 比較用: 人気別の回収率
    save("ls_overall", summary(a, ["全場"]))
    save("ls_by_class", summary(a, ["区分"]))
    save("ls_by_score", summary(a, ["好材料数"]))
    save("ls_by_index", summary(a, ["指数区分"]))
    save("ls_by_pop_index", summary(a, ["人気帯", "指数区分"]))
    save("ls_by_year_class", summary(a, ["年", "区分"]))
    save("ls_by_course_class", summary(a, ["競馬場", "区分"]).reindex(T.ORDER, level=0))
    save("ls_three_groups", three_groups(a))
    # 期間を分けた検証: 2023-2025 と 2026
    a["期間"] = np.where(a["年"] <= 2025, "2023-2025", "2026(検証)")
    save("ls_by_period_class", summary(a, ["期間", "区分"]))


if __name__ == "__main__":
    main()
