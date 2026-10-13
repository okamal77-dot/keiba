"""各競馬場の級(A1, B2, C3 …)を1着賞金で横並びに比較し, 移籍馬の級の移動と成績を集計する.

(1) 級の賞金ラダー: 競馬場×級の1着賞金中央値 (2024〜2026年8月のレース一覧)
(2) 賞金帯ごとの対応表: 同じ賞金帯にどの競馬場のどの級が入るか
(3) 地方間移籍: 前走の級賞金 → 移籍初戦の級賞金 の比率別成績, 主な級の対応
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
TRACKS = ["大井", "船橋", "浦和", "川崎", "園田", "姫路", "高知", "名古屋", "門別", "盛岡", "水沢", "金沢", "佐賀", "笠松"]
TOKEN = re.compile(r"(?<![A-Za-z])([ABC])(\d*)(?![A-Za-z])")
BANDS = [(400, 10**6, "400万〜"), (250, 400, "250〜399万"), (150, 250, "150〜249万"), (100, 150, "100〜149万"),
         (80, 100, "80〜99万"), (60, 80, "60〜79万"), (50, 60, "50〜59万"), (40, 50, "40〜49万"), (0, 40, "〜39万")]


def token(name):
    s = unicodedata.normalize("NFKC", str(name))
    if "2歳" in s or "新馬" in s or ("3歳" in s and "3歳以上" not in s):
        return None
    m = TOKEN.search(s)
    return (m.group(1), int(m.group(2)) if m.group(2) else None) if m else None


def ladder():
    r = pd.concat([ma.read(f"race{y}.csv") for y in (2024, 2025, 2026)]).drop_duplicates(KEY)
    r = r[~r["競走種類名称"].isin(["重賞", "準重賞"])]
    t = r["レース名"].map(token)
    r["級"] = t.map(lambda x: x[0] if x else None)
    r["組"] = t.map(lambda x: x[1] if x else None)
    r["賞金"] = pd.to_numeric(r["1着賞金(円)"], errors="coerce") / 10000
    r = r.dropna(subset=["級", "賞金"])
    g = r.groupby(["競馬場", "級", "組"], dropna=False)["賞金"].agg(["size", "median"]).reset_index()
    g = g[g["size"] >= 15]
    # 番号付きの級がある競馬場では, 番号なし(B など)は除外
    has_num = g.dropna(subset=["組"]).groupby(["競馬場", "級"]).size()
    g = g[g["組"].notna() | ~g.set_index(["競馬場", "級"]).index.isin(has_num.index)]
    rows = []
    for (jo, lv, med), x in g.groupby(["競馬場", "級", "median"]):
        nums = sorted(int(v) for v in x["組"].dropna())
        if not nums:
            label = f"{lv}級"
        elif len(nums) == 1:
            label = f"{lv}{nums[0]}"
        else:
            label = f"{lv}{nums[0]}〜{lv}{nums[-1]}"
        rows.append({"競馬場": jo, "級": label, "級記号": lv, "最小組": nums[0] if nums else 0,
                     "1着賞金中央値(万円)": med, "レース数": int(x["size"].sum())})
    lad = pd.DataFrame(rows)
    lad = lad[lad["競馬場"].isin(TRACKS)]
    lad["競馬場"] = pd.Categorical(lad["競馬場"], TRACKS, ordered=True)
    lad = lad.sort_values(["競馬場", "1着賞金中央値(万円)", "級記号", "最小組"], ascending=[True, False, True, True])
    lad = lad.drop(columns=["級記号", "最小組"])
    lad.to_csv(OUT / "grade_ladder.csv", index=False, encoding="utf-8-sig")

    # 賞金帯 × 競馬場
    band = []
    for lo, hi, lab in BANDS:
        row = {"1着賞金": lab}
        for jo in TRACKS:
            x = g[(g["競馬場"] == jo) & (g["median"] >= lo) & (g["median"] < hi)]
            parts = []
            for lv in ("A", "B", "C"):
                y = x[x["級"] == lv]
                if y.empty:
                    continue
                nums = sorted(int(v) for v in y["組"].dropna())
                if not nums:
                    parts.append(f"{lv}級")
                elif len(nums) <= 2:
                    parts.append("・".join(f"{lv}{n}" for n in nums))
                else:
                    parts.append(f"{lv}{nums[0]}〜{lv}{nums[-1]}")
            row[jo] = "・".join(parts)
        band.append(row)
    band = pd.DataFrame(band)
    band.to_csv(OUT / "grade_band_table.csv", index=False, encoding="utf-8-sig")
    print(band.to_string(index=False))
    return lad, g


def prize_map(g):
    """(競馬場, 級, 組) -> 賞金中央値. 組が無い/未知なら級全体の中央値."""
    exact = {(a, b, c): m for a, b, c, m in zip(g["競馬場"], g["級"], g["組"], g["median"]) if pd.notna(c)}
    lv = g.groupby(["競馬場", "級"])["median"].median().to_dict()
    return exact, lv


def transfers(g):
    exact, lv = prize_map(g)
    hs, ps = [], []
    for y, f in ma.YEARS.items():
        p, h = ma.load_year(y, f, f"pay{y}.csv")
        ps.append(ma.read(f"pay{y}.csv")[KEY + ["レース名"]].drop_duplicates(KEY))
        hs.append(h)
    h = pd.concat(hs, ignore_index=True).merge(pd.concat(ps), on=KEY, how="left")
    tk = h["レース名"].map(token)
    h["級"] = tk.map(lambda x: x[0] if x else None)
    h["組"] = tk.map(lambda x: x[1] if x else None)

    def pz(jo, a, b):
        if a is None:
            return None
        if b is not None and (jo, a, b) in exact:
            return exact[(jo, a, b)]
        return lv.get((jo, a))
    h["級賞金"] = [pz(j, a, b) for j, a, b in zip(h["競馬場"], h["級"], h["組"])]
    h["級名"] = [f"{a}{int(b)}" if a and b is not None and not pd.isna(b) else (f"{a}級" if a else None)
                for a, b in zip(h["級"], h["組"])]

    h = h.sort_values(["馬名", "生年月日", "競走年月日", "レース番号"])
    grp = h.groupby(["馬名", "生年月日"])
    for c in ("調教師所属", "競馬場", "級賞金", "級名", "着順"):
        h[f"前_{c}"] = grp[c].shift()
    mv = h[h["前_調教師所属"].notna() & (h["前_調教師所属"] != h["調教師所属"])
           & (h["前_調教師所属"] != "JRA") & (h["調教師所属"] != "JRA")].copy()
    mv = mv.dropna(subset=["級賞金", "前_級賞金", "人気", "着順"])
    mv["top3"] = mv["着順"] <= 3
    base = h.dropna(subset=["人気", "着順"]).assign(top3=lambda x: x["着順"] <= 3).groupby(["競馬場", "人気"])["top3"].mean()
    mv["exp3"] = [base.get((j, p)) for j, p in zip(mv["競馬場"], mv["人気"])]
    mv["比"] = mv["級賞金"] / mv["前_級賞金"]
    mv["級の移動"] = pd.cut(mv["比"], [0, 0.5, 0.8, 1.2, 2.0, 99],
                         labels=["大きく下の級(〜0.5倍)", "下の級(0.5〜0.8倍)", "同等の級(0.8〜1.2倍)",
                                 "上の級(1.2〜2倍)", "大きく上の級(2倍〜)"])
    s = mv.groupby("級の移動", observed=False).agg(
        出走数=("top3", "size"), 平均人気=("人気", "mean"), 勝率=("着順", lambda x: (x == 1).mean() * 100),
        複勝率=("top3", lambda x: x.mean() * 100), 単勝回収率=("単勝払戻", "mean"), 複勝回収率=("複勝払戻", "mean"))
    s["人気比"] = mv.groupby("級の移動", observed=False).apply(lambda x: x["top3"].mean() / x["exp3"].mean(), include_groups=False)
    s.round(2).to_csv(OUT / "grade_move_summary.csv", encoding="utf-8-sig")
    print("\n== 移籍初戦: 前走の級賞金に対する移籍先の級賞金の比\n", s.round(2).to_string())

    # 前走の着順別 (前走で好走していたか)
    mv["前走"] = pd.cut(mv["前_着順"], [0, 3, 6, 99], labels=["前走3着内", "前走4-6着", "前走7着以下"])
    s2 = mv.pivot_table(index="級の移動", columns="前走", values="top3", aggfunc="mean", observed=False) * 100
    print("\n== 複勝率: 級の移動 × 前走着順\n", s2.round(1).to_string())
    s2.round(1).to_csv(OUT / "grade_move_by_prev.csv", encoding="utf-8-sig")

    # 南関東C級からの移籍先: どの級に入り, どう走るか
    nk = mv[mv["前_競馬場"].isin(["大井", "船橋", "浦和", "川崎"]) & mv["前_級名"].str.startswith("C", na=False)]
    nk = nk[~nk["競馬場"].isin(["大井", "船橋", "浦和", "川崎"])]
    t = nk.groupby("競馬場").agg(
        出走数=("top3", "size"), 主な級=("級名", lambda x: "・".join(x.value_counts().head(3).index)),
        移籍先級賞金中央値=("級賞金", "median"), 勝率=("着順", lambda x: (x == 1).mean() * 100),
        複勝率=("top3", lambda x: x.mean() * 100), 平均人気=("人気", "mean"),
        単勝回収率=("単勝払戻", "mean"), 複勝回収率=("複勝払戻", "mean"))
    t = t[t["出走数"] >= 30].sort_values("複勝率", ascending=False)
    t.round(1).to_csv(OUT / "grade_from_nankan_c.csv", encoding="utf-8-sig")
    print("\n== 南関東C級 → 他地区 移籍初戦\n", t.round(1).to_string())

    # 主要な級の対応 (前走: 競馬場+級 → 移籍先: 競馬場+級), 30件以上
    mv["元"] = mv["前_競馬場"].astype(str) + " " + mv["前_級名"].astype(str)
    mv["先"] = mv["競馬場"].astype(str) + " " + mv["級名"].astype(str)
    pair = mv.groupby(["元", "先"]).agg(件数=("top3", "size"), 複勝率=("top3", lambda x: x.mean() * 100),
                                      人気比=("exp3", "mean"), 複勝回収率=("複勝払戻", "mean"))
    pair["人気比"] = mv.groupby(["元", "先"])["top3"].mean() / pair["人気比"]
    pair = pair[pair["件数"] >= 30].sort_values("件数", ascending=False)
    pair.round(2).to_csv(OUT / "grade_pairs.csv", encoding="utf-8-sig")
    print("\n== 主要な級の対応 (30件以上)\n", pair.round(2).head(40).to_string())


def main():
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    lad, g = ladder()
    transfers(g)


if __name__ == "__main__":
    main()
