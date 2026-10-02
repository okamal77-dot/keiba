"""一般 / 特別 / 重賞(準重賞含む) の比較 (レース一覧がある年のみ)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from manbaken_analysis import KEY, YEARS, load_year, read, running_style  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
import manbaken_analysis as ma  # noqa: E402
ma.DATA = DATA
OUT = Path(__file__).parent / "out"
CLS = {"普通": "一般", "特別": "特別", "重賞": "重賞", "準重賞": "重賞"}
ORDER = ["帯広ば", "大井", "川崎", "浦和", "名古屋", "盛岡", "船橋", "水沢", "高知", "佐賀", "園田", "姫路", "門別", "笠松", "金沢"]


def main():
    ys = [y for y in YEARS if (DATA / f"race{y}.csv").exists()]
    loaded = {y: load_year(y, YEARS[y], f"pay{y}.csv") for y in ys}
    pay = pd.concat([p for p, _ in loaded.values()])
    h = pd.concat([x for _, x in loaded.values()])
    r = pd.concat([read(f"race{y}.csv") for y in ys]).drop_duplicates(KEY)
    r["クラス"] = r["競走種類名称"].map(CLS)
    r["湿"] = r["馬場"].isin(["重", "不良"])
    r = r.merge(pay[KEY + ["万馬券", "３連単払戻金（円）"]], on=KEY)
    h = h.merge(r[KEY + ["クラス", "万馬券"]], on=KEY)
    h["top3"] = h["着順"] <= 3

    g = r.groupby(["競馬場", "クラス"])
    fav = h[h["人気"] == 1].groupby(["競馬場", "クラス"])
    win = h[h["着順"] == 1]
    ana = h[h["人気"] >= 6].groupby(["競馬場", "クラス"])
    t = pd.DataFrame({
        "レース数": g.size(),
        "万馬券率": g["万馬券"].mean() * 100,
        "3連単中央値": g["３連単払戻金（円）"].median(),
        "10万超率": g["３連単払戻金（円）"].apply(lambda x: (x >= 1e5).mean() * 100),
        "平均頭数": g["頭数"].mean(),
        "平均距離": g["距離"].mean(),
        "1人気勝率": fav["着順"].apply(lambda x: (x == 1).mean() * 100),
        "1人気複勝率": fav["top3"].mean() * 100,
        "勝馬平均人気": win.groupby(["競馬場", "クラス"])["人気"].mean(),
        "穴3着内率": ana["top3"].mean() * 100,
    })
    # 頭数を揃えた比較: 一般戦の頭数別万馬券率で期待値を作り, 実績との差を見る
    r["頭数i"] = r["頭数"].clip(upper=14)
    base = r[r["クラス"] == "一般"].groupby(["競馬場", "頭数i"])["万馬券"].mean().rename("期待")
    rr = r.merge(base, left_on=["競馬場", "頭数i"], right_index=True, how="left")
    t["頭数補正後の差"] = (rr.groupby(["競馬場", "クラス"])["万馬券"].mean()
                     - rr.groupby(["競馬場", "クラス"])["期待"].mean()) * 100
    # 馬場: 重・不良 vs 良・稍重 の万馬券率差
    wet = r.pivot_table(index=["競馬場", "クラス"], columns="湿", values="万馬券", aggfunc="mean")
    t["湿-乾"] = (wet.get(True) - wet.get(False)) * 100

    # 脚質
    rows = []
    for _, race in r[r["競馬場"] != "帯広ば"].iterrows():
        for b, (st, p) in running_style(race).items():
            rows.append((race["競馬場"], race["競走年月日"], race["レース番号"], b, st, p))
    st = pd.DataFrame(rows, columns=KEY + ["馬番", "脚質", "最終角位置"])
    hs = h.merge(st, on=KEY + ["馬番"])
    w = hs[hs["着順"] == 1]
    sw = pd.crosstab([w["競馬場"], w["クラス"]], w["脚質"], normalize="index") * 100
    t["勝馬_逃げ"] = sw["逃げ"]
    t["勝馬_差追"] = sw["差し"] + sw["追込"]
    top = hs[hs["top3"]]
    t["前残り率"] = top.groupby(["競馬場", "クラス"])["最終角位置"].apply(lambda x: (x <= 3).mean() * 100)

    # 騎手: リーディング上位5人(一般戦の勝利数)が占める勝利シェア
    lead = (win[win["クラス"] == "一般"].groupby(["競馬場", "騎手名"]).size()
            .groupby(level=0, group_keys=False).nlargest(5).reset_index()[["競馬場", "騎手名"]])
    lead["上位"] = True
    wl = win.merge(lead, on=["競馬場", "騎手名"], how="left")
    t["上位5騎手勝利シェア"] = wl.groupby(["競馬場", "クラス"])["上位"].apply(lambda x: x.notna().mean() * 100)

    # 全体 (競馬場計)
    t = t.reset_index()
    t["競馬場"] = pd.Categorical(t["競馬場"], ORDER, ordered=True)
    t["クラス"] = pd.Categorical(t["クラス"], ["一般", "特別", "重賞"], ordered=True)
    t = t.sort_values(["競馬場", "クラス"])
    t.round(1).to_csv(OUT / "class_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 200)
    print(t.round(1).to_string(index=False))

    # 全場計
    allg = r.groupby("クラス")
    favall = h[h["人気"] == 1].groupby("クラス")
    print(pd.DataFrame({
        "レース数": allg.size(), "万馬券率": allg["万馬券"].mean() * 100,
        "中央値": allg["３連単払戻金（円）"].median(), "頭数": allg["頭数"].mean(),
        "1人気勝率": favall["着順"].apply(lambda x: (x == 1).mean() * 100), "1人気複勝率": favall["top3"].mean() * 100,
        "穴3着内率": h[h["人気"] >= 6].groupby("クラス")["top3"].mean() * 100,
        "補正差": rr.groupby("クラス")["万馬券"].mean() * 100 - rr.groupby("クラス")["期待"].mean() * 100,
        "勝馬逃げ": w.groupby("クラス")["脚質"].apply(lambda x: (x == "逃げ").mean() * 100),
        "勝馬差追": w.groupby("クラス")["脚質"].apply(lambda x: x.isin(["差し", "追込"]).mean() * 100),
        "前残り": top.groupby("クラス")["最終角位置"].apply(lambda x: (x <= 3).mean() * 100),
        "上位5騎手": wl.groupby("クラス")["上位"].apply(lambda x: x.notna().mean() * 100),
    }).round(1).to_string())


if __name__ == "__main__":
    main()
