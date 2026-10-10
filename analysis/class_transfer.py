"""(1) 競馬場×クラス(A/B/C/D・2歳・3歳・重賞)別の上位4人気決着率など
(2) 転厩・移籍馬の成績 (人気との比較, 回収率, 転入後何戦目か)
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import manbaken_analysis as ma  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
ma.DATA = DATA
OUT = Path(__file__).parent / "out"
KEY = ma.KEY
ORDER = ["帯広ば", "大井", "川崎", "浦和", "船橋", "名古屋", "笠松", "園田", "姫路", "金沢",
         "高知", "佐賀", "門別", "盛岡", "水沢"]
CLS_ORDER = ["重賞", "A", "B", "C", "D", "3歳", "2歳", "その他"]
LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)


def race_class(name, kind):
    if kind in ("重賞", "準重賞"):
        return "重賞"
    s = unicodedata.normalize("NFKC", str(name))
    if "2歳" in s or "新馬" in s:
        return "2歳"
    if "3歳" in s and "3歳以上" not in s:
        return "3歳"
    m = LETTER.search(s)
    return m.group(1) if m else "その他"


def load():
    loaded = {y: ma.load_year(y, f, f"pay{y}.csv") for y, f in ma.YEARS.items()}
    pay = pd.concat([p for p, _ in loaded.values()])
    h = pd.concat([x for _, x in loaded.values()], ignore_index=True)
    names = pd.concat([ma.read(f"pay{y}.csv")[KEY + ["レース名"]] for y in ma.YEARS]).drop_duplicates(KEY)
    kinds = pd.concat([ma.read(f"race{y}.csv")[KEY + ["競走種類名称"]] for y in ma.YEARS
                       if (DATA / f"race{y}.csv").exists()]).drop_duplicates(KEY)
    pay = pay.merge(names, on=KEY, how="left").merge(kinds, on=KEY, how="left")
    pay["クラス"] = [race_class(n, k) for n, k in zip(pay["レース名"], pay["競走種類名称"])]
    return pay, h


# ---------------------------------------------------------------- (1) クラス別
def class_table(pay, h):
    top = h[h["着順"] <= 3].sort_values(KEY + ["着順"]).groupby(KEY)["人気"].agg(list).rename("pops")
    r = pay.merge(top, left_on=KEY, right_index=True)
    r = r[r["pops"].str.len() >= 3]
    r["top4"] = r["pops"].apply(lambda x: max(x[:3]) <= 4)
    r["fav_in"] = r["pops"].apply(lambda x: 1 in x[:3])
    fav = h[h["人気"] == 1][KEY + ["着順", "単勝払戻"]].drop_duplicates(KEY)
    r = r.merge(fav.rename(columns={"着順": "fav着"}), on=KEY, how="left")
    r["fav_odds"] = (r["単勝払戻"] / 100).where(r["fav着"] == 1)
    g = r.groupby(["競馬場", "クラス"])
    t = pd.DataFrame({
        "レース数": g.size(),
        "上位4人気決着率": g["top4"].mean() * 100,
        "万馬券率": g["万馬券"].mean() * 100,
        "1人気勝率": g["fav着"].apply(lambda x: (x == 1).mean() * 100),
        "1人気複勝率": g["fav_in"].mean() * 100,
        "1人気勝時オッズ中央値": g["fav_odds"].median(),
        "平均頭数": g["頭数"].mean(),
        "3連単中央値": g["３連単払戻金（円）"].median(),
    }).reset_index()
    t = t[t["レース数"] >= 30]
    t["競馬場"] = pd.Categorical(t["競馬場"], ORDER, ordered=True)
    t["クラス"] = pd.Categorical(t["クラス"], CLS_ORDER, ordered=True)
    t = t.sort_values(["競馬場", "クラス"])
    t.round(1).to_csv(OUT / "top4_by_class.csv", index=False, encoding="utf-8-sig")
    print("== 競馬場×クラス\n", t.round(1).to_string(index=False))
    allc = r.groupby("クラス").agg(n=("top4", "size"), top4=("top4", "mean"), man=("万馬券", "mean"),
                                  fav=("fav_in", "mean"), odds=("fav_odds", "median"))
    print("\n== クラス計\n", allc.reindex(CLS_ORDER).round(3))


# ---------------------------------------------------------------- (2) 転厩・移籍
def starts(rec):
    try:
        return sum(int(x) for x in str(rec).split("-"))
    except ValueError:
        return 0


def transfer_table(h):
    h = h.sort_values(["馬名", "生年月日", "競走年月日", "レース番号"]).copy()
    hk = ["馬名", "生年月日"]
    grp = h.groupby(hk)
    h["prev所属"] = grp["調教師所属"].shift()
    h["prev調教師"] = grp["調教師"].shift()
    h["初出"] = h["prev所属"].isna()
    h["全出走数"] = h["全成績"].map(starts)

    ev = pd.Series(pd.NA, index=h.index, dtype="object")
    move = ~h["初出"] & (h["prev所属"] != h["調教師所属"])
    ev[move & (h["prev所属"] == "JRA")] = "中央から転入"
    ev[move & (h["prev所属"] != "JRA") & (h["調教師所属"] != "JRA")] = "地方間移籍"
    ev[~h["初出"] & ~move & (h["prev調教師"] != h["調教師"])] = "同地区内の転厩"
    # データに前走がなく, 既に出走歴がある馬 = 中央(JRA)からの転入と推定 (2023年7月以降の初出に限定)
    ev[h["初出"] & (h["全出走数"] > 0) & (h["競走年月日"] >= 20230701)
       & (h["調教師所属"] != "JRA") & (h["競馬場"] != "帯広ば")] = "中央から転入(推定)"
    h["event"] = ev
    h["src"] = h["prev所属"].where(ev.notna())
    # 転入後何戦目か (イベント発生から3戦まで追跡)
    h["blk"] = h["event"].notna().astype(int).groupby([h["馬名"], h["生年月日"]]).cumsum()
    h["戦目"] = h.groupby(hk + ["blk"]).cumcount() + 1
    h["区分"] = h.groupby(hk + ["blk"])["event"].transform("first")
    h.loc[(h["blk"] == 0) | (h["戦目"] > 3), "区分"] = "通常"
    h.loc[h["区分"].isna(), "区分"] = "通常"
    h["区分"] = h["区分"].where(h["区分"] == "通常", h["区分"] + "_" + h["戦目"].astype(str) + "戦目")

    h = h[h["人気"].notna() & h["着順"].notna() & (h["調教師所属"] != "JRA")].copy()
    h["top3"] = h["着順"] <= 3
    h["win"] = h["着順"] == 1
    # 人気から期待される複勝率/勝率 (競馬場×人気の全体平均)
    base = h.groupby(["競馬場", "人気"]).agg(exp3=("top3", "mean"), exp1=("win", "mean"))
    h = h.join(base, on=["競馬場", "人気"])

    def summ(df, by):
        g = df.groupby(by)
        return pd.DataFrame({
            "出走数": g.size(),
            "平均人気": g["人気"].mean(),
            "勝率": g["win"].mean() * 100,
            "複勝率": g["top3"].mean() * 100,
            "人気比_勝率": g["win"].mean() / g["exp1"].mean(),
            "人気比_複勝率": g["top3"].mean() / g["exp3"].mean(),
            "単勝回収率": g["単勝払戻"].mean(),
            "複勝回収率": g["複勝払戻"].mean(),
            "6人気~の複勝率": g.apply(lambda x: x.loc[x["人気"] >= 6, "top3"].mean() * 100, include_groups=False),
        })

    s_all = summ(h, "区分")
    print("\n== 区分別 (全場)\n", s_all.round(2).to_string())
    s_all.round(3).to_csv(OUT / "transfer_summary.csv", encoding="utf-8-sig")

    first = h[h["区分"].str.endswith("_1戦目")].copy()
    first["種別"] = first["区分"].str.replace("_1戦目", "")
    by_track = summ(first[first["種別"].isin(["地方間移籍", "中央から転入(推定)", "同地区内の転厩"])], ["競馬場", "種別"])
    by_track = by_track[by_track["出走数"] >= 30]
    by_track = by_track.reset_index()
    by_track["競馬場"] = pd.Categorical(by_track["競馬場"], ORDER, ordered=True)
    by_track = by_track.sort_values(["競馬場", "種別"])
    print("\n== 転入初戦 競馬場別\n", by_track.round(2).to_string(index=False))
    by_track.round(3).to_csv(OUT / "transfer_first_by_track.csv", index=False, encoding="utf-8-sig")

    # 地方間移籍 1-3戦目の移籍元→移籍先
    mv = h[h["区分"].str.startswith("地方間移籍")].copy()
    mv["元"] = mv.groupby(["馬名", "生年月日", "blk"])["src"].transform("first")
    flow = summ(mv, ["元", "調教師所属"])
    flow = flow[flow["出走数"] >= 80].sort_values("人気比_複勝率", ascending=False)
    print("\n== 地方間移籍 1-3戦目: 移籍元→移籍先 (80走以上)\n", flow.round(2).to_string())
    flow.round(3).to_csv(OUT / "transfer_flow.csv", encoding="utf-8-sig")

    # 人気帯別 (転入初戦 vs 通常)
    h["人気帯"] = pd.cut(h["人気"], [0, 1, 3, 5, 99], labels=["1人気", "2-3人気", "4-5人気", "6人気~"])
    h["種別"] = h["区分"].str.replace(r"_\d戦目", "", regex=True)
    h["何戦目"] = h["区分"].str.extract(r"_(\d)戦目")[0].fillna("-")
    pb = h[h["何戦目"].isin(["1", "-"])].pivot_table(index="種別", columns="人気帯", values="top3", aggfunc="mean", observed=False) * 100
    print("\n== 人気帯別 複勝率 (転入初戦 / 通常)\n", pb.round(1).to_string())
    pb.round(1).to_csv(OUT / "transfer_by_popularity.csv", encoding="utf-8-sig")

    # 万馬券への関与: 万馬券レースで6人気~で3着内に来た馬のうち, 転入1-3戦目の馬の割合
    m = h[h["万馬券"] & h["top3"] & (h["人気"] >= 6)]
    share = (m["種別"] != "通常").mean() * 100
    share_runs = (h["種別"] != "通常").mean() * 100
    print(f"\n万馬券の穴馬(6人気~3着内)に占める転入1-3戦目の馬: {share:.1f}% (全出走に占める割合 {share_runs:.1f}%)")


def main():
    pay, h = load()
    h = h.merge(pay[KEY + ["クラス", "万馬券"]], on=KEY, how="left")
    class_table(pay, h)
    transfer_table(h)


if __name__ == "__main__":
    main()
