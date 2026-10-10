"""競馬場(所属地区)の規模ランキング: 賞金・開催数・頭数. 移籍の指標用."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import class_transfer as ct  # noqa: E402
import manbaken_analysis as ma  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
ma.DATA = DATA
OUT = Path(__file__).parent / "out"
KEY = ma.KEY
REGION = {"大井": "大井", "川崎": "川崎", "船橋": "船橋", "浦和": "浦和", "園田": "兵庫", "姫路": "兵庫",
          "名古屋": "愛知", "笠松": "笠松", "金沢": "金沢", "高知": "高知", "佐賀": "佐賀",
          "門別": "北海道", "盛岡": "岩手", "水沢": "岩手", "帯広ば": "ばんえい"}
YEARS_R = [2024, 2025, 2026]


def main():
    r = pd.concat([ma.read(f"race{y}.csv") for y in YEARS_R]).drop_duplicates(KEY)
    names = pd.concat([ma.read(f"pay{y}.csv")[KEY + ["レース名"]] for y in YEARS_R]).drop_duplicates(KEY)
    r = r.merge(names, on=KEY, how="left", suffixes=("", "_p"))
    r["クラス"] = [ct.race_class(n, k) for n, k in zip(r["レース名"], r["競走種類名称"])]
    r["地区"] = r["競馬場"].map(REGION)
    r["賞金"] = pd.to_numeric(r["1着賞金(円)"], errors="coerce") / 10000
    r["年"] = r["競走年月日"] // 10000
    days = 365 * 2 + 244  # 2024-01-01〜2026-08-31
    g = r.groupby("地区")
    t = pd.DataFrame({
        "1着賞金中央値(万円)": g["賞金"].median(),
        "C級中央値": r[r["クラス"] == "C"].groupby("地区")["賞金"].median(),
        "B級中央値": r[r["クラス"] == "B"].groupby("地区")["賞金"].median(),
        "A級中央値": r[r["クラス"] == "A"].groupby("地区")["賞金"].median(),
        "重賞中央値": r[r["クラス"] == "重賞"].groupby("地区")["賞金"].median(),
        "1着賞金総額(億円/年)": g["賞金"].sum() / 10000 / (days / 365),
        "年間レース数": g.size() / (days / 365),
    })
    hs = []
    for y in YEARS_R:
        h = ma.read(ma.YEARS[y])[["競馬場", "馬名", "生年月日", "調教師所属", "競走年月日"]]
        hs.append(h)
    h = pd.concat(hs)
    h["地区"] = h["競馬場"].map(REGION)
    h = h[h["調教師所属"] != "JRA"]
    t["在籍頭数(年平均)"] = h.groupby(["地区", h["競走年月日"] // 10000])[["馬名", "生年月日"]].apply(
        lambda x: len(x.drop_duplicates())).groupby("地区").mean()
    t = t.sort_values("C級中央値", ascending=False)
    t.insert(0, "順位", range(1, len(t) + 1))
    t.round(1).to_csv(OUT / "track_scale.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(1).to_string())

    # 移籍成績との関係: 移籍元→先の C級賞金比 と 人気比_複勝率
    f = pd.read_csv(OUT / "transfer_flow.csv", encoding="utf-8-sig")
    lv = t["C級中央値"]
    f["賞金比(先/元)"] = f["調教師所属"].map(lv) / f["元"].map(lv)
    f = f.dropna(subset=["賞金比(先/元)"])
    f["区分"] = pd.cut(f["賞金比(先/元)"], [0, 0.8, 0.95, 1.05, 1.25, 9],
                     labels=["格下へ(~0.8)", "やや格下(0.8-0.95)", "同格", "やや格上(1.05-1.25)", "格上へ(1.25~)"])
    s = f.groupby("区分", observed=False).apply(lambda x: pd.Series({
        "組数": len(x), "出走数": x["出走数"].sum(),
        "人気比_複勝率": (x["人気比_複勝率"] * x["出走数"]).sum() / x["出走数"].sum(),
        "複勝回収率": (x["複勝回収率"] * x["出走数"]).sum() / x["出走数"].sum(),
        "単勝回収率": (x["単勝回収率"] * x["出走数"]).sum() / x["出走数"].sum()}), include_groups=False)
    print("\n== 移籍先/移籍元のC級賞金比 と 移籍後1-3戦の成績\n", s.round(2).to_string())
    print("相関(賞金比 vs 人気比_複勝率):", round(f["賞金比(先/元)"].corr(f["人気比_複勝率"]), 3))
    s.round(3).to_csv(OUT / "transfer_by_scale_gap.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
