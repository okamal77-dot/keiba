"""3連単の的中馬（1〜3着馬）と各要因の関連性分析（2023〜2026年10月・地方15場）.

要因: 人気 / 走破タイム(持ち時計・前走指数) / 上がり3F / 脚質 / 馬体重増減 /
      騎手との相性(コンビ成績・乗り替わり) / 天候 / 馬場状態 / 転厩・遠征

人気の影響を取り除くため, 各カテゴリの「3着内数」を
「同じ競馬場・同じ人気・同じ頭数帯の平均3着内率」から計算した期待値と比べる (A/E 比).
A/E > 1 なら人気以上に馬券に絡んでいる.

入力: data/ 配下の出走馬・レース一覧・払戻 CSV (FILES), 2026年9月の確定オッズ data/odds/202609_*_odds.csv
出力: analysis/out/tf_*.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
KEY = ["競馬場", "競走年月日", "レース番号"]
ORDER = ["帯広ば", "大井", "川崎", "浦和", "船橋", "名古屋", "笠松", "園田", "姫路", "金沢",
         "高知", "佐賀", "門別", "盛岡", "水沢"]
HOME = {"帯広ば": "ばんえい", "名古屋": "愛知", "園田": "兵庫", "姫路": "兵庫", "門別": "北海道",
        "盛岡": "岩手", "水沢": "岩手"}  # 他は競馬場名 = 所属名
BABA = ["良", "稍重", "重", "不良"]
FILES = {  # 年: (出走馬, レース一覧, 払戻) のファイル名リスト
    2023: (["2023_horselist_全月まとめ.csv"], ["race2023.csv"], ["pay2023.csv"]),
    2024: (["2024_horselist_全月まとめ.csv"], ["race2024.csv"], ["pay2024.csv"]),
    2025: (["2025_horselist_全月まとめ.csv"], ["race2025.csv"], ["pay2025.csv"]),
    2026: (["2026_horselist_1-8月まとめ.csv", "202609_horselist.csv", "202610_horselist.csv"],
           ["race2026.csv", "202609_racelist.csv", "202610_racelist.csv"],
           ["pay2026.csv", "202609_payback.csv", "202610_payback.csv"]),
}
YEARS = list(FILES)
ODDS_MONTH = 202609  # 確定オッズがある月 (オッズで補正した検証に使う)


def read(name, **kw):
    return pd.read_csv(DATA / name, encoding="utf-8-sig", low_memory=False, **kw)


def rec_n(s):
    """'1-2-0-39' -> (出走数, 3着内数)"""
    m = s.astype(str).str.extract(r"^(\d+)-(\d+)-(\d+)-(\d+)$").astype(float)
    return m.sum(axis=1).where(m[0].notna()), m[[0, 1, 2]].sum(axis=1).where(m[0].notna())


def to_sec(t):
    """1309 -> 90.9秒 (m ss d)"""
    return (t // 1000) * 60 + (t % 1000) / 10


def best_sec(s):
    m = s.astype(str).str.extract(r"(\d+):(\d+\.\d)")
    return m[0].astype(float) * 60 + m[1].astype(float)


def parse_corner(s):
    pos, i = {}, 1
    for grp in re.findall(r"\([^)]*\)|\d+", str(s)):
        nums = re.findall(r"\d+", grp)
        for x in nums:
            pos[int(x)] = i
        i += len(nums)
    return pos


def styles(races):
    """最初と最終コーナーの位置から脚質 (manbaken_analysis.py と同じ基準)"""
    rows = []
    cols = [f"コーナー通過順{i}" for i in range(1, 9)]
    for rec in races[KEY + cols].itertuples(index=False):
        cs = [c for c in rec[3:] if isinstance(c, str) and c]
        if not cs:
            continue
        first, last = parse_corner(cs[0]), parse_corner(cs[-1])
        n = max(len(last), 1)
        for b, p in last.items():
            f = first.get(b, p)
            st = "逃げ" if f == 1 else "先行" if p / n <= 0.35 else "差し" if p / n <= 0.7 else "追込"
            rows.append((*rec[:3], b, st, p))
    return pd.DataFrame(rows, columns=KEY + ["馬番", "脚質", "4角位置"])


def baba_group(row):
    b = row["馬場"]
    if row["競馬場"] != "帯広ば":
        return b if b in BABA else np.nan
    v = pd.to_numeric(b, errors="coerce")  # ばんえいは馬場水分(%)
    if pd.isna(v):
        return np.nan
    return "水分~1.4%" if v < 1.5 else "水分1.5-2.4%" if v < 2.5 else "水分2.5%~"


def load():
    h, r, p = (pd.concat([read(f) for y in YEARS for f in FILES[y][i]], ignore_index=True) for i in range(3))
    r, p = r.drop_duplicates(KEY), p.drop_duplicates(KEY)
    h = h.drop_duplicates(KEY + ["馬番"])
    for c in ["着順", "人気", "上がり3F", "タイム", "馬体重増減", "馬番"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["馬体重"] = pd.to_numeric(h["馬体重"], errors="coerce")
    h = h[h["着順"].notna() & h["人気"].notna()].copy()       # 取消・除外・中止を除く
    r["馬場G"] = r.apply(baba_group, axis=1)
    # 2023年の船橋は全レース「良」で記録されており馬場データとして使えない
    r.loc[(r["競馬場"] == "船橋") & (r["競走年月日"] // 10000 == 2023), "馬場G"] = np.nan
    h = h.merge(r[KEY + ["距離", "天候", "馬場", "馬場G", "競走種類名称"]], on=KEY, how="left")
    h["頭数"] = h.groupby(KEY)["馬番"].transform("size")
    h = h[h["頭数"] >= 5]
    h = h[h.set_index(KEY).index.isin(p.set_index(KEY).index)]   # 払戻のあるレース
    h["top3"] = (h["着順"] <= 3).astype(int)
    h["日付"] = pd.to_datetime(h["競走年月日"].astype(str))
    h["年"] = h["日付"].dt.year
    h["馬ID"] = h["馬名"] + "_" + h["生年月日"].astype(str)
    st = styles(r)
    h = h.merge(st, on=KEY + ["馬番"], how="left")
    return h, r, p


def features(h):
    h = h.sort_values(["馬ID", "日付", "レース番号"]).copy()
    # ---- 走破タイム指数: 競馬場×距離×馬場区分ごとの z (小さいほど速い→符号反転) ----
    h["秒"] = to_sec(h["タイム"])
    grp = h.groupby(["競馬場", "距離", "馬場G"])["秒"]
    h["指数"] = -(h["秒"] - grp.transform("mean")) / grp.transform("std")
    h.loc[grp.transform("size") < 30, "指数"] = np.nan
    # ---- 上がり3F 順位 (レース内) ----
    h["上がり順"] = h.groupby(KEY)["上がり3F"].rank(method="min")
    # ---- 前走情報 ----
    g = h.groupby("馬ID")
    for c in ["指数", "上がり順", "脚質", "騎手名", "調教師", "調教師所属", "競馬場", "日付", "着順", "人気"]:
        h["前走" + c] = g[c].shift(1)
    h["近3走ベスト指数"] = g["指数"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())
    h["間隔日"] = (h["日付"] - h["前走日付"]).dt.days
    # ---- レース内順位 (前走指数・持ち時計) ----
    h["持ち時計"] = best_sec(h["最高タイム"])
    h["指数順"] = h.groupby(KEY)["近3走ベスト指数"].rank(ascending=False, method="min")
    h["持ち時計順"] = h.groupby(KEY)["持ち時計"].rank(method="min")
    # ---- 騎手×馬コンビ (騎手成績 = この騎手でのこの馬の成績) ----
    n, t3 = rec_n(h["騎手成績"])
    h["コンビ数"], h["コンビ3着内"] = n, t3
    # ---- 当場成績 / 所属 ----
    h["当場出走数"], _ = rec_n(h["当競馬場成績"])
    h["全出走数"], _ = rec_n(h["全成績"])
    h["所属地"] = h["競馬場"].map(HOME).fillna(h["競馬場"])
    return h


def cat_features(h):
    c = pd.DataFrame(index=h.index)
    c["人気"] = pd.cut(h["人気"], [0, 1, 2, 3, 4, 5, 7, 9, 99],
                     labels=["1番", "2番", "3番", "4番", "5番", "6-7番", "8-9番", "10番~"])
    rk = lambda s, lab: pd.cut(s, [0, 1, 2, 3, 5, 99], labels=lab).cat.add_categories("データなし").fillna("データなし")
    c["近3走ベスト指数順位(レース内)"] = rk(h["指数順"], ["1位", "2位", "3位", "4-5位", "6位~"])
    c["持ち時計順位(レース内)"] = rk(h["持ち時計順"], ["1位", "2位", "3位", "4-5位", "6位~"])
    c["前走上がり3F順位"] = rk(h["前走上がり順"], ["1位", "2位", "3位", "4-5位", "6位~"])
    c["前走脚質"] = h["前走脚質"].fillna("データなし")
    c["馬体重増減"] = pd.cut(h["馬体重増減"], [-99, -10, -4, 3, 9, 99],
                        labels=["-10kg以下", "-9~-4kg", "-3~+3kg", "+4~+9kg", "+10kg以上"]
                        ).cat.add_categories("前走なし/計不").fillna("前走なし/計不")
    # 騎手相性
    combo = np.select(
        [h["コンビ数"].isna(), h["コンビ数"] == 0,
         (h["コンビ数"] >= 3) & (h["コンビ3着内"] / h["コンビ数"] >= 0.5),
         (h["コンビ数"] >= 3) & (h["コンビ3着内"] / h["コンビ数"] < 0.2)],
        ["不明", "初コンビ", "好相性(3戦以上,3着内率50%~)", "低相性(3戦以上,3着内率~20%)"], "その他(1-2戦等)")
    c["騎手コンビ"] = combo
    c["乗り替わり"] = np.where(h["前走騎手名"].isna(), "前走なし",
                          np.where(h["前走騎手名"] == h["騎手名"], "継続騎乗", "乗り替わり"))
    # 転厩・遠征
    tr = np.select(
        [h["調教師所属"] != h["所属地"],
         h["前走調教師所属"].notna() & (h["前走調教師所属"] != h["調教師所属"]),
         h["前走調教師"].notna() & (h["前走調教師"] != h["調教師"]),
         h["前走調教師"].isna() & (h["当場出走数"] == 0) & (h["全出走数"] > 0)],
        ["遠征馬(他地区所属)", "他地区から転入初戦", "同地区内の転厩初戦", "当場初出走(データ内初登場)"], "通常")
    c["転厩・遠征"] = tr
    c["前走から"] = np.select(
        [h["前走競馬場"].isna(), h["前走競馬場"] == h["競馬場"]], ["前走なし", "同場"], "他場から")
    return c


def ae_table(h, cat, by="競馬場"):
    """カテゴリ別 A/E (実際の3着内数 / 人気から見た期待3着内数)"""
    d = pd.DataFrame({by: h[by], "cat": cat, "a": h["top3"], "e": h["期待"]})
    d["v"] = d["e"] * (1 - d["e"])
    g = d.groupby([by, "cat"], observed=True).agg(n=("a", "size"), a=("a", "sum"), e=("e", "sum"), v=("v", "sum"))
    g["3着内率"] = g["a"] / g["n"] * 100
    g["A/E"] = g["a"] / g["e"]
    g["z"] = (g["a"] - g["e"]) / np.sqrt(g["v"])
    return g


def main():
    h, r, p = load()
    h = features(h)
    # 期待3着内率: 競馬場 × 人気(10以上まとめ) × 頭数帯
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待"] = h.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)["top3"].transform("mean")
    c = cat_features(h)
    h["全場"] = "全場"
    print("対象レース数", h.groupby(KEY).ngroups, "出走頭数", len(h))

    # ===== 1. 人気: 3連単の1-2-3着の人気構成 =====
    w = h[h["着順"] <= 3]
    comp = w.pivot_table(index="競馬場", columns="着順", values="人気", aggfunc="mean")
    comp.columns = [f"{int(x)}着平均人気" for x in comp.columns]
    pop_rate = pd.crosstab(h["競馬場"], c["人気"], values=h["top3"], aggfunc="mean") * 100
    rp = w.sort_values(KEY + ["着順"]).groupby(KEY).agg(pops=("人気", list), n=("頭数", "first"))
    rp = rp[rp["pops"].str.len() == 3]
    rp["上位3人気で決着"] = rp["pops"].apply(lambda x: max(x) <= 3)
    rp["上位5人気で決着"] = rp["pops"].apply(lambda x: max(x) <= 5)
    rp["6番人気以下が絡む"] = rp["pops"].apply(lambda x: max(x) >= 6)
    rp["1番人気が絡む"] = rp["pops"].apply(lambda x: 1 in x)
    rp["人気合計"] = rp["pops"].apply(sum)
    rp = rp.reset_index()
    pop_sum = rp.groupby("競馬場")[["上位3人気で決着", "上位5人気で決着", "6番人気以下が絡む", "1番人気が絡む"]].mean() * 100
    pop_sum["人気合計中央値"] = rp.groupby("競馬場")["人気合計"].median()
    pop_sum["レース数"] = rp.groupby("競馬場").size()
    rp["年"] = rp["競走年月日"] // 10000
    for y in YEARS:
        pop_sum[f"6番人気以下が絡む{y}"] = rp[rp["年"] == y].groupby("競馬場")["6番人気以下が絡む"].mean() * 100
    pop = pd.concat([pop_sum, comp], axis=1).reindex(ORDER)
    save("tf_pop_summary", pop)
    save("tf_pop_top3rate", pop_rate.reindex(ORDER))

    # ===== 2. レース内の上がり3F・脚質 (結果としての特徴) =====
    nb = h[h["競馬場"] != "帯広ば"]
    wnb = nb[nb["着順"] <= 3]
    agari = wnb.assign(上がり=pd.cut(wnb["上がり順"], [0, 1, 2, 3, 5, 99], labels=["最速", "2位", "3位", "4-5位", "6位~"]))
    ag = {}
    for k in (1, 2, 3):
        x = agari[agari["着順"] == k]
        ag[f"{k}着_上がり3位内%"] = x.groupby("競馬場")["上がり順"].apply(lambda s: (s <= 3).mean() * 100)
    ag["3着内馬の上がり3位内%"] = wnb.groupby("競馬場")["上がり順"].apply(lambda s: (s <= 3).mean() * 100)
    ag["上がり最速馬の3着内率"] = nb[nb["上がり順"] == 1].groupby("競馬場")["top3"].mean() * 100
    ag["上がり最速馬の勝率"] = nb[nb["上がり順"] == 1].groupby("競馬場")["着順"].apply(lambda s: (s == 1).mean() * 100)
    ag = pd.DataFrame(ag).reindex(ORDER)
    save("tf_agari_inrace", ag)
    sty = nb[nb["脚質"].notna()]
    st_rate = sty.pivot_table(index="競馬場", columns="脚質", values="top3", aggfunc="mean") * 100
    st_share = pd.crosstab(sty[sty["着順"] <= 3]["競馬場"], sty[sty["着順"] <= 3]["脚質"], normalize="index") * 100
    st_win = pd.crosstab(sty[sty["着順"] == 1]["競馬場"], sty[sty["着順"] == 1]["脚質"], normalize="index") * 100
    st_ae = ae_table(sty, sty["脚質"])["A/E"].unstack()
    for n_, d in [("tf_style_top3rate", st_rate), ("tf_style_share_top3", st_share),
                  ("tf_style_share_win", st_win), ("tf_style_ae", st_ae)]:
        save(n_, d.reindex(ORDER)[["逃げ", "先行", "差し", "追込"]])

    # ===== 3. 事前情報 (人気補正後 A/E) =====
    res = {}
    for f in ["近3走ベスト指数順位(レース内)", "持ち時計順位(レース内)", "前走上がり3F順位", "前走脚質",
              "馬体重増減", "騎手コンビ", "乗り替わり", "転厩・遠征", "前走から"]:
        a = ae_table(h, c[f])
        t = ae_table(h, c[f], by="全場")
        out = pd.concat([t, a])
        for y in YEARS:  # 年ごとの A/E (再現性の確認)
            m = h["年"] == y
            yy = pd.concat([ae_table(h[m], c.loc[m, f], by="全場"), ae_table(h[m], c.loc[m, f])])
            out[f"A/E_{y}"] = yy["A/E"]
            out[f"n_{y}"] = yy["n"]
        ye = out[[f"A/E_{y}" for y in YEARS]]
        out["全年同方向"] = ((ye > 1).all(axis=1) & (out["A/E"] > 1)) | ((ye < 1).all(axis=1) & (out["A/E"] < 1))
        res[f] = out
        save(f"tf_ae_{f.split('(')[0]}", out.round(3), show=False)
        print(f"\n== {f}: 全場")
        print(t.round(2).to_string())
        print(a["A/E"].unstack().reindex(ORDER).round(2).to_string())
        print(out.loc[out["全年同方向"] & (out["n"] >= 100)].index.tolist())

    # ===== 4. 天候・馬場状態 =====
    races = h.drop_duplicates(KEY)[KEY + ["天候", "馬場G"]]
    rp2 = rp.merge(races, on=KEY)
    bb = rp2.groupby(["競馬場", "馬場G"]).agg(
        レース数=("pops", "size"), 上位3人気で決着=("上位3人気で決着", "mean"),
        六番人気以下が絡む=("6番人気以下が絡む", "mean"), 一番人気が絡む=("1番人気が絡む", "mean"),
        人気合計中央値=("人気合計", "median"))
    bb[["上位3人気で決着", "六番人気以下が絡む", "一番人気が絡む"]] *= 100
    rp2["年"] = rp2["競走年月日"] // 10000
    for y in YEARS:
        bb[f"六番人気以下が絡む{y}"] = rp2[rp2["年"] == y].groupby(["競馬場", "馬場G"])["6番人気以下が絡む"].mean() * 100
        bb[f"レース数{y}"] = rp2[rp2["年"] == y].groupby(["競馬場", "馬場G"]).size()
    save("tf_baba_pop", bb, show=False)
    rp2["天候G"] = rp2["天候"].replace({"小雨": "雨", "小雪": "雪/雨", "雪": "雪/雨"}).replace({"雨": "雨・雪"}).replace({"雪/雨": "雨・雪"})
    wt = rp2.groupby(["競馬場", "天候G"]).agg(
        レース数=("pops", "size"), 上位3人気で決着=("上位3人気で決着", "mean"),
        六番人気以下が絡む=("6番人気以下が絡む", "mean"), 人気合計中央値=("人気合計", "median"))
    wt[["上位3人気で決着", "六番人気以下が絡む"]] *= 100
    save("tf_weather_pop", wt, show=False)
    # 馬場 × 脚質 (前残り/差し) : 3着内馬に占める逃げ・先行の割合
    sty2 = sty[sty["着順"] <= 3]
    bs = sty2.assign(前=sty2["脚質"].isin(["逃げ", "先行"])).groupby(["競馬場", "馬場G"])["前"].mean().unstack() * 100
    save("tf_baba_front_share", bs.reindex(ORDER)[BABA])
    bae = ae_table(sty, sty["脚質"] + "|" + sty["馬場G"].astype(str))["A/E"].unstack()
    save("tf_baba_style_ae", bae.reindex(ORDER), show=False)
    # 馬場 × 上がり最速馬の3着内率
    ba = nb[nb["上がり順"] == 1].groupby(["競馬場", "馬場G"])["top3"].mean().unstack() * 100
    save("tf_baba_agari1", ba.reindex(ORDER)[BABA])
    # 馬場 × 1番人気3着内率
    fv = h[h["人気"] == 1].groupby(["競馬場", "馬場G"])["top3"].mean().unstack() * 100
    save("tf_baba_fav", fv.reindex(ORDER), show=False)

    # ===== 5. 確定オッズで補正した検証 (オッズがある月のみ, 全場) =====
    odds_check(h, c)

    # ===== 6. 要因の重要度ランキング (競馬場別, |A/E-1| 最大のカテゴリ, n>=100) =====
    rows = []
    for f, t in res.items():
        t = t[(t["n"] >= 100) & (t.index.get_level_values(1) != "データなし")]
        for (k, cat), v in t.iterrows():
            rows.append((k, f, cat, v["n"], v["3着内率"], v["A/E"], v["z"],
                         *[v[f"A/E_{y}"] for y in YEARS], v["全年同方向"]))
    rk = pd.DataFrame(rows, columns=["競馬場", "要因", "カテゴリ", "n", "3着内率", "A/E", "z"]
                      + [f"A/E_{y}" for y in YEARS] + ["全年同方向"])
    sig = rk[rk["z"].abs() >= 2.5].copy()
    sig["強さ"] = (sig["A/E"] - 1).abs()
    sig = sig.sort_values(["競馬場", "強さ"], ascending=[True, False])
    sig.to_csv(OUT / "tf_significant.csv", index=False, encoding="utf-8-sig")
    for k in ["全場"] + ORDER:
        print(f"\n## {k}")
        print(sig[sig["競馬場"] == k].head(14).round(2).to_string(index=False))


def odds_check(h, c):
    """人気順位の代わりに確定単勝オッズで期待3着内率を作り, 要因の A/E を比べる.

    期待値 = 単勝オッズから出した勝率 (1/オッズ をレース内で合計1に正規化) の帯 × 頭数帯 の平均3着内率.
    人気順位より細かく市場の評価を差し引くので, ここでも A/E が 1 から離れる要因は
    「オッズにも織り込まれていない」ことになる.
    """
    files = sorted((DATA / "odds").glob(f"{ODDS_MONTH}_*_odds.csv"))
    if not files:
        return
    o = pd.concat([read(f"odds/{f.name}", usecols=KEY + ["賭式", "番号1", "オッズ"]) for f in files])
    o = o[o["賭式"] == "単勝"].rename(columns={"番号1": "馬番", "オッズ": "単勝オッズ"}).drop(columns="賭式")
    o["馬番"] = pd.to_numeric(o["馬番"], errors="coerce")
    m = h["競走年月日"] // 100 == ODDS_MONTH
    x = h[m].reset_index().merge(o, on=KEY + ["馬番"], how="inner").set_index("index")
    x = x[x["単勝オッズ"] > 0]
    x["勝率"] = 1 / x["単勝オッズ"]
    x["勝率"] /= x.groupby(KEY)["勝率"].transform("sum")
    x["勝率帯"] = pd.qcut(x["勝率"], 25, duplicates="drop")
    x["期待_人気"] = x["期待"]
    x["期待"] = x.groupby(["勝率帯", "頭数帯"], observed=True)["top3"].transform("mean")
    x["全場"] = "全場"
    rows = []
    for f in ["近3走ベスト指数順位(レース内)", "持ち時計順位(レース内)", "前走上がり3F順位", "前走脚質",
              "馬体重増減", "騎手コンビ", "乗り替わり", "転厩・遠征"]:
        a = ae_table(x, c.loc[x.index, f], by="全場")
        b = ae_table(x.assign(期待=x["期待_人気"]), c.loc[x.index, f], by="全場")
        t = a[["n", "3着内率", "A/E", "z"]].rename(columns={"A/E": "A/E_オッズ補正", "z": "z_オッズ補正"})
        t["A/E_人気補正"] = b["A/E"]
        t.index = pd.MultiIndex.from_tuples([(f, k[1]) for k in t.index], names=["要因", "カテゴリ"])
        rows.append(t)
    out = pd.concat(rows)
    print(f"\n== オッズ補正 ({ODDS_MONTH}, {x.groupby(KEY).ngroups}レース, {len(x)}頭)")
    save("tf_odds_check", out.round(3))


def save(name, df, show=True):
    df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")
    if show:
        print(f"\n== {name}\n", df.round(1).to_string())


if __name__ == "__main__":
    main()
