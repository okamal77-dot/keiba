"""クラス別の回収率（2023〜2026年9月・地方15場）.

longshot_payout.py の穴馬区分（好材料3以上 / その他 / 消し）を, レースのクラス別に集計する.
クラスはレース一覧の 競走種類名称・条件・レース名 から判定する:
  重賞 (重賞・準重賞) > 新馬 > 2歳 > 3歳 (3歳限定) > オープン > A級 > B級 > C級 > その他

入力: trifecta_factors.py と同じ data/ 配下のファイル
出力: analysis/out/cls_*.csv
"""
import re

import numpy as np
import pandas as pd

import longshot_payout as L
import trifecta_factors as T

KEY = T.KEY
# レース名中の格付け記号 (全角). 直後が数字・漢数字・組・－などのときだけ格付けとみなす
GRADE = re.compile(r"([ＡＢＣ])(?=[０-９0-9一二三四五六七八九十上下組級－\-・ＢＣ　 （(]|$)")


def race_class(row):
    kind, cond, name = str(row["競走種類名称"]), str(row["条件"]), str(row["レース名"])
    if kind in ("重賞", "準重賞"):
        return "重賞・準重賞"
    if "新馬" in name:
        return "新馬"
    if re.search(r"[2２]歳", cond) and "以上" not in cond:
        return "2歳"
    if re.search(r"[3３]歳", cond) and "以上" not in cond:
        return "3歳"
    if "オープン" in name or "ＯＰ" in name:
        return "オープン"
    m = GRADE.search(name)
    if m:
        return {"Ａ": "A級", "Ｂ": "B級", "Ｃ": "C級"}[m.group(1)]
    return "その他"


ORDER = ["重賞・準重賞", "オープン", "A級", "B級", "C級", "3歳", "2歳", "新馬", "その他"]


def summary(x, keys):
    x = x.assign(rid=x["競馬場"] + x["競走年月日"].astype(str) + "_" + x["レース番号"].astype(str))
    g = x.groupby(keys, observed=True)
    return pd.DataFrame({
        "レース数": g["rid"].nunique(),
        "頭数": g.size(),
        "3着内率": g["top3"].mean() * 100,
        "単勝回収率": g["単勝払戻"].mean(),
        "複勝回収率": g["複勝払戻"].mean(),
        "3着追加回収率": g["3着追加払戻"].sum() / (g.size() * L.N_ADD * 100) * 100,
        "単勝SE": g["単勝払戻"].std() / np.sqrt(g.size()),
        "複勝SE": g["複勝払戻"].std() / np.sqrt(g.size()),
    })


def save(name, df):
    df.to_csv(T.OUT / f"{name}.csv", encoding="utf-8-sig")
    print(f"\n== {name}\n", df.round(1).to_string())


def main():
    h, a = L.prepare()
    r = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][1]], ignore_index=True).drop_duplicates(KEY)
    r["クラス"] = r.apply(race_class, axis=1)
    cls = r[KEY + ["クラス"]]
    h = h.merge(cls, on=KEY, how="left")
    a = a.merge(cls, on=KEY, how="left").set_index(a.index)
    for d in (h, a):
        d["クラス"] = pd.Categorical(d["クラス"].fillna("その他"), ORDER)
    # 3区分 (帯広は好材料を数えられないので穴馬区分の集計からは除く)
    a["3区分"] = np.select([a["消し"], a["好材料数"] >= 3], ["消し", "好材料3以上"], "その他")
    a.loc[a["競馬場"] == "帯広ば", "3区分"] = "帯広(対象外)"
    h["人気帯"] = pd.cut(h["人気"], [0, 1, 3, 5, 99], labels=["1番人気", "2-3番人気", "4-5番人気", "6番人気以下"])

    save("cls_popularity", summary(h, ["クラス", "人気帯"])[["レース数", "頭数", "3着内率", "単勝回収率", "複勝回収率", "単勝SE", "複勝SE"]])
    b = a[a["3区分"] != "帯広(対象外)"]
    save("cls_longshot_groups", summary(b, ["クラス", "3区分"]))
    # 好材料3以上の穴馬: クラス × 年
    g3 = b[b["3区分"] == "好材料3以上"]
    t = summary(g3, ["クラス", "年"])[["頭数", "単勝回収率", "複勝回収率", "3着追加回収率"]]
    save("cls_good3_by_year", t)
    # 6番人気以下全体: クラス × 年
    save("cls_longshot_by_year", summary(h[h["人気"] >= 6], ["クラス", "年"])[["頭数", "単勝回収率", "複勝回収率", "3着追加回収率"]])
    # クラス × 競馬場 (好材料3以上の穴馬)
    save("cls_good3_by_course", summary(g3, ["クラス", "競馬場"])[["頭数", "単勝回収率", "複勝回収率", "3着追加回収率", "単勝SE", "複勝SE"]])
    # クラス判定の内訳
    print(r.groupby("クラス").size().reindex(ORDER))
    r[["競馬場", "競走年月日", "レース番号", "競走種類名称", "条件", "レース名", "クラス"]].to_csv(
        T.OUT / "cls_race_class.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
