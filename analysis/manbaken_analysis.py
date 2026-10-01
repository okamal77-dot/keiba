"""各競馬場の3連単万馬券（配当10,000円以上）の特徴分析.

入力: data/ 配下の 2023/2026 払戻・出走馬・レース一覧 CSV
出力: analysis/out/*.csv と標準出力のサマリ
"""
import re
import sys
from pathlib import Path

import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
KEY = ["競馬場", "競走年月日", "レース番号"]
TH = 10000


def read(name):
    return pd.read_csv(DATA / name, encoding="utf-8-sig", low_memory=False)


def load_year(year, horse_file, pay_file):
    pay = read(pay_file).drop_duplicates(KEY)  # 同着は先頭行
    pay = pay[KEY + ["３連単払戻金（円）", "３連単人気"]].dropna(subset=["３連単払戻金（円）"])
    pay["万馬券"] = pay["３連単払戻金（円）"] >= TH
    pay["年"] = year
    h = read(horse_file)
    h["着順"] = pd.to_numeric(h["着順"], errors="coerce")
    h["人気"] = pd.to_numeric(h["人気"], errors="coerce")
    h["年"] = year
    n = h.groupby(KEY).size().rename("頭数")
    pay = pay.merge(n, left_on=KEY, right_index=True, how="left")
    return pay, h


def parse_corner(s):
    """'(1,3),7-5=2' -> {馬番: 位置}"""
    pos, i = {}, 1
    for grp in re.findall(r"\([^)]*\)|\d+", str(s)):
        nums = re.findall(r"\d+", grp)
        for x in nums:
            pos[int(x)] = i
        i += len(nums)
    return pos


def running_style(race):
    """最初と最終コーナーの位置から脚質を判定"""
    cs = [race[f"コーナー通過順{i}"] for i in range(1, 9) if pd.notna(race[f"コーナー通過順{i}"])]
    if not cs:
        return {}
    first, last = parse_corner(cs[0]), parse_corner(cs[-1])
    n = max(len(last), 1)
    out = {}
    for b, p in last.items():
        f = first.get(b, p)
        if f == 1:
            st = "逃げ"
        elif p / n <= 0.35:
            st = "先行"
        elif p / n <= 0.7:
            st = "差し"
        else:
            st = "追込"
        out[b] = (st, p)
    return out


def main():
    p23, h23 = load_year(2023, "2023_horselist_全月まとめ.csv", "pay2023.csv")
    p26, h26 = load_year(2026, "2026_horselist_1-8月まとめ.csv", "pay2026.csv")
    pay = pd.concat([p23, p26])
    h = pd.concat([h23, h26]).merge(pay[KEY + ["万馬券", "３連単払戻金（円）"]], on=KEY, how="inner")
    h["top3"] = h["着順"] <= 3
    h["穴top3"] = h["top3"] & (h["人気"] >= 6)

    # ---- 1. 概要 ----
    g = pay.groupby("競馬場")
    summ = pd.DataFrame({
        "レース数": g.size(),
        "万馬券率": g["万馬券"].mean(),
        "万馬券率2023": p23.groupby("競馬場")["万馬券"].mean(),
        "万馬券率2026": p26.groupby("競馬場")["万馬券"].mean(),
        "3連単中央値": g["３連単払戻金（円）"].median(),
        "10万超率": g["３連単払戻金（円）"].apply(lambda x: (x >= 100000).mean()),
        "平均頭数": g["頭数"].mean(),
    })

    # ---- 2. 人気との乖離 ----
    top = h[h["top3"]]
    w = h[h["着順"] == 1]
    fav = h[h["人気"] == 1]
    summ["1人気複勝率_全体"] = fav.groupby("競馬場")["top3"].mean()
    summ["1人気勝率_全体"] = fav.groupby("競馬場")["着順"].apply(lambda x: (x == 1).mean())
    mf = fav[fav["万馬券"]]
    summ["万馬券時_1人気着外率"] = 1 - mf.groupby("競馬場")["top3"].mean()
    mw = w[w["万馬券"]]
    summ["万馬券時_勝馬平均人気"] = mw.groupby("競馬場")["人気"].mean()
    summ["万馬券時_勝馬1-3人気率"] = mw.groupby("競馬場")["人気"].apply(lambda x: (x <= 3).mean())
    mt = top[top["万馬券"]]
    summ["万馬券時_3着内平均人気"] = mt.groupby("競馬場")["人気"].mean()
    summ["万馬券時_穴(6人気~)絡み率"] = mt.groupby(KEY)["人気"].max().ge(6).groupby("競馬場").mean()
    # 万馬券の人気パターン(1着の人気帯)
    pat = mw.assign(帯=pd.cut(mw["人気"], [0, 1, 3, 6, 99], labels=["1人気", "2-3人気", "4-6人気", "7人気~"]))
    pat = pd.crosstab(pat["競馬場"], pat["帯"], normalize="index")
    # 何着に穴が来やすいか
    pos_ana = mt[mt["人気"] >= 6].groupby(["競馬場", "着順"]).size().unstack(fill_value=0)
    pos_ana = pos_ana.div(pos_ana.sum(axis=1), axis=0)

    # 頭数別
    pay["頭数帯"] = pd.cut(pay["頭数"], [0, 8, 10, 12, 99], labels=["~8頭", "9-10頭", "11-12頭", "13頭~"])
    by_n = pay.pivot_table(index="競馬場", columns="頭数帯", values="万馬券", aggfunc="mean", observed=False)

    # ---- 3. 馬場状態 (2026のみ: レース情報あり) ----
    r = read("race2026.csv").drop_duplicates(KEY)
    r = r.merge(p26[KEY + ["万馬券", "３連単払戻金（円）"]], on=KEY, how="inner")
    ban = r["競馬場"] == "帯広ば"
    r["馬場区分"] = r["馬場"]
    m = pd.to_numeric(r.loc[ban, "馬場"], errors="coerce")
    r.loc[ban, "馬場区分"] = pd.cut(m, [-1, 1.4, 2.4, 99], labels=["水分~1.4%(重い)", "1.5-2.4%", "2.5%~(軽い)"]).astype(str)
    baba = r.pivot_table(index="競馬場", columns="馬場区分", values="万馬券", aggfunc="mean")
    baba_n = r.pivot_table(index="競馬場", columns="馬場区分", values="万馬券", aggfunc="size")
    r["距離帯"] = pd.cut(r["距離"], [0, 1200, 1500, 1800, 9999], labels=["~1200", "1201-1500", "1501-1800", "1801~"])
    dist = r[~ban].pivot_table(index="競馬場", columns="距離帯", values="万馬券", aggfunc="mean", observed=False)
    grade = r.pivot_table(index="競馬場", columns="競走種類名称", values="万馬券", aggfunc="mean")
    r["R帯"] = pd.cut(r["レース番号"], [0, 4, 8, 99], labels=["1-4R", "5-8R", "9R~"])
    rno = r.pivot_table(index="競馬場", columns="R帯", values="万馬券", aggfunc="mean", observed=False)

    # ---- 4. 脚質・展開 (2026, ばんえい除く) ----
    rows = []
    for _, race in r[~ban].iterrows():
        for b, (st, p) in running_style(race).items():
            rows.append((race["競馬場"], race["競走年月日"], race["レース番号"], b, st, p))
    st = pd.DataFrame(rows, columns=KEY + ["馬番", "脚質", "最終角位置"])
    hs = h26.merge(st, on=KEY + ["馬番"]).merge(p26[KEY + ["万馬券"]], on=KEY)
    hs["top3"] = hs["着順"] <= 3
    win = hs[hs["着順"] == 1]
    style_win = pd.crosstab([win["競馬場"], win["万馬券"]], win["脚質"], normalize="index")
    # 逃げ馬の3着内率 (万馬券 vs 非万馬券)
    nige = hs[hs["脚質"] == "逃げ"].groupby(["競馬場", "万馬券"])["top3"].mean().unstack()
    # 穴馬(6人気~)が3着内に来た時の脚質
    ana = hs[(hs["人気"] >= 6) & hs["top3"]]
    style_ana = pd.crosstab(ana["競馬場"], ana["脚質"], normalize="index")
    # 上位人気(1-3)が凡走した時の脚質
    flop = hs[(hs["人気"] <= 3) & ~hs["top3"] & hs["万馬券"]]
    style_flop = pd.crosstab(flop["競馬場"], flop["脚質"], normalize="index")
    # 前残り度: 万馬券レースで4角3番手以内から3着内に入った頭数割合
    mt26 = hs[hs["top3"]]
    front = mt26.assign(前=mt26["最終角位置"] <= 3).groupby(["競馬場", "万馬券"])["前"].mean().unstack()

    # ---- 5. 騎手 (2023+2026) ----
    hj = h.copy()
    jg = hj.groupby(["競馬場", "騎手名"])
    jk = pd.DataFrame({
        "騎乗数": jg.size(),
        "6人気~騎乗": jg["人気"].apply(lambda x: (x >= 6).sum()),
        "穴3着内数": jg["穴top3"].sum(),
        "万馬券絡み数": jg.apply(lambda x: (x["top3"] & x["万馬券"]).sum(), include_groups=False),
        "万馬券を穴で演出": jg.apply(lambda x: (x["穴top3"] & x["万馬券"]).sum(), include_groups=False),
        "1人気騎乗": jg["人気"].apply(lambda x: (x == 1).sum()),
        "1人気着外": jg.apply(lambda x: ((x["人気"] == 1) & ~x["top3"]).sum(), include_groups=False),
    }).reset_index()
    jk["穴3着内率"] = jk["穴3着内数"] / jk["6人気~騎乗"]
    jk["1人気着外率"] = jk["1人気着外"] / jk["1人気騎乗"]
    base = hj[hj["人気"] >= 6].groupby("競馬場")["top3"].mean().rename("場平均穴3着内率")
    jk = jk.merge(base, left_on="競馬場", right_index=True)
    jk["穴期待比"] = jk["穴3着内率"] / jk["場平均穴3着内率"]
    jk = jk[jk["6人気~騎乗"] >= 60]
    top_j = (jk.sort_values(["競馬場", "万馬券を穴で演出"], ascending=[True, False])
               .groupby("競馬場").head(5))
    top_ratio = (jk.sort_values(["競馬場", "穴期待比"], ascending=[True, False])
                   .groupby("競馬場").head(3))
    fav_j = jk[jk["1人気騎乗"] >= 30].sort_values(["競馬場", "1人気着外率"], ascending=[True, False]).groupby("競馬場").head(2)

    res = {
        "summary": summ.sort_values("万馬券率", ascending=False),
        "winner_pop_pattern": pat, "ana_finish_pos": pos_ana, "by_fieldsize": by_n,
        "baba_rate": baba, "baba_n": baba_n, "by_distance": dist, "by_grade": grade, "by_raceno": rno,
        "style_winner": style_win, "nige_top3": nige, "style_ana": style_ana,
        "style_fav_flop": style_flop, "front_share": front,
        "jockey_ana_count": top_j, "jockey_ana_ratio": top_ratio, "jockey_fav_flop": fav_j,
    }
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    for k, v in res.items():
        v.to_csv(OUT / f"{k}.csv", encoding="utf-8-sig")
        print(f"\n===== {k} =====")
        print(v.round(3).to_string())


if __name__ == "__main__":
    main()
