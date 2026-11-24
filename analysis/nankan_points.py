"""南関東 (大井・川崎・船橋・浦和) の格付ポイント (2024年1月から) を出走記録から計算し,
「同じクラスの中で, あとどれだけで上のクラスに上がるか (クラス内の位置)」が人気以上に走るかを確かめる.

使い方: python3 analysis/nankan_points.py
ポイント表 (1〜5着): 南関東の格付ポイント表 (令和6年1月1日〜) による. 重賞の格 (SⅠ〜SⅢ) はレース名から読めないので,
  Jpn の格が読めない重賞は SⅢ (牝馬限定は牝SⅡ相当) とみなす. 2着以下は表の比率 (重賞 0.35/0.2/0.1/0.05,
  準重賞 0.36/0.22/0.13/0.09, 特別・普通 0.4/0.25/0.15/0.1) で付ける. 同着は考えない.
クラスの区切り (格付表): 馬齢 (3〜8歳以上) と 上半期/下半期 で決まる.
ポイントを正しく数えられる馬だけを使う: 2024年1月以降に南関東でデビューし (初出走の全成績が 0-0-0-0),
  それより前に他地区の出走記録がない馬. 中央・他地区からの転入馬は初期ポイントが分からないので除く.
クラス内の位置 = (今のポイント − 今のクラスの下限) ÷ (上のクラスの下限 − 今のクラスの下限). 1に近いほど昇級が近い.
評価: 2025年1月〜の一般戦 (2歳・3歳限定戦を除く B・C級) で, 位置の帯ごとの 勝ち A/E (人気から見込む勝ち数との比) と 単勝回収率.
出力: analysis/out/np_horses.csv (出走ごと), np_summary.csv
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from class_change import parse_class

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
OUT = ROOT / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
NANKAN = ["大井", "川崎", "船橋", "浦和"]
HORSE = [f"{y}_horselist_全月まとめ.csv" for y in (2021, 2022, 2023, 2024, 2025)] + \
    ["2026_horselist_1-8月まとめ.csv", "202609_horselist.csv", "202610_horselist.csv", "202611_horselist.csv"]
RACE = [f"race{y}.csv" for y in (2022, 2023, 2024, 2025, 2026)] + \
    ["202609_racelist.csv", "202610_racelist.csv", "202611_racelist.csv"]
PAY = [f"pay{y}.csv" for y in (2022, 2023, 2024, 2025, 2026)] + \
    ["202609_payback.csv", "202610_payback.csv", "202611_payback.csv"]

SPECIAL = {"A1": 600, "A2": 500, "B1": 320, "B2": 280, "B3": 240, "C1": 200, "C2": 180, "C3": 140}
NORMAL = {"A1": 0, "A2": 320, "B1": 280, "B2": 240, "B3": 200, "C1": 150, "C2": 100, "C3": 80}
RATIO = {"重賞": [1, .35, .2, .1, .05], "準重賞": [1, .36, .22, .13, .09], "他": [1, .4, .25, .15, .1]}
CLASSES = ["A1", "A2", "B1", "B2", "B3", "C1", "C2", "C3"]
# 格付表: クラスの下限 [3歳, 4歳, 5歳, 6歳, 7歳, 8歳以上]
FIRST = {"A1": [2800, 3400, 4300, 5000, 5500, 5900], "A2": [2000, 2200, 2500, 3200, 3700, 4100],
         "B1": [1500, 1700, 1900, 2200, 2700, 3000], "B2": [1100, 1200, 1400, 1700, 2000, 2200],
         "B3": [700, 800, 1000, 1300, 1600, 1800], "C1": [np.nan, 500, 700, 1000, 1300, 1500],
         "C2": [np.nan, 200, 400, 700, 1000, 1200], "C3": [np.nan, 0, 80, 160, 240, 240]}
SECOND = {"A1": [3000, 3600, 4400, 5200, 5700, 5900], "A2": [2200, 2300, 2600, 3400, 3900, 4100],
          "B1": [1700, 1800, 2000, 2500, 2900, 3000], "B2": [1200, 1300, 1500, 1800, 2100, 2200],
          "B3": [800, 900, 1100, 1400, 1700, 1800], "C1": [500, 600, 800, 1100, 1400, 1500],
          "C2": [200, 300, 500, 800, 1100, 1200], "C3": [np.nan, 20, 80, 160, 240, 240]}


def read(f, **kw):
    return pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, **kw)


def race_cls(name, course):
    c = parse_class(name, course)
    if not isinstance(c, tuple) or c[1] == 0:
        return None
    s = f"{'ABC'[c[0]]}{c[1]}"
    return s if s in SPECIAL else None


def win_points(kind, name, cond, month, cls):
    t = unicodedata.normalize("NFKC", str(name))
    age = "2歳" if "2歳" in str(cond) else ("3歳" if "3歳" in str(cond) and "3歳以上" not in str(cond) else "一般")
    if kind == "重賞":
        g = 1 if re.search(r"(JpnI|GI)(?!I)", t) else 2 if "JpnII" in t and "JpnIII" not in t else 3 if "JpnIII" in t else 4
        if age == "2歳":
            return [2000, 1500, 1200, 1000][g - 1], "重賞"
        if age == "3歳":
            return [2400, 1800, 1400, 1200][g - 1], "重賞"
        return [5000, 3100, 2600, 2100][g - 1], "重賞"
    if kind == "準重賞":
        return (600 if age != "一般" else 700), "準重賞"
    if kind == "特別":
        if age == "2歳":
            return 320, "他"
        if age == "3歳":
            return 280, "他"
        return (SPECIAL.get(cls, np.nan)), "他"
    if age == "2歳":
        return (280 if "新馬" in t else 250), "他"
    if age == "3歳":
        return (220 if month <= 6 else 160), "他"
    return NORMAL.get(cls, np.nan), "他"


def bounds(cls, age, month):
    tb = FIRST if month <= 6 else SECOND
    j = min(max(int(age), 3), 8) - 3
    lo = tb[cls][j]
    i = CLASSES.index(cls)
    hi = tb[CLASSES[i - 1]][j] if i > 0 else np.nan
    return lo, hi


def main():
    cols = KEY + ["馬番", "馬名", "生年月日", "齢", "全成績", "着順", "人気"]
    h = pd.concat([read(f, usecols=cols) for f in HORSE if (DATA / f).exists()], ignore_index=True)
    h = h.drop_duplicates(KEY + ["馬番"], keep="last")
    for c in ["着順", "人気", "齢", "馬番"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["ID"] = h["馬名"].astype(str) + "|" + h["生年月日"].astype(str)
    rl = pd.concat([read(f, usecols=KEY + ["競走種類名称", "レース名", "条件", "頭数"]) for f in RACE if (DATA / f).exists()])
    rl = rl.drop_duplicates(KEY, keep="last")
    h = h.merge(rl, on=KEY, how="left").sort_values(["ID", "競走年月日", "レース番号"])
    # ポイントを数えられる馬: 最初の記録が 2024年以降の南関東でのデビュー
    first = h.groupby("ID").head(1)
    ok = first[(first["競馬場"].isin(NANKAN)) & (first["競走年月日"] >= 20240101) &
               (first["全成績"].astype(str).str.replace(" ", "").str.match(r"^0-0-0-0"))]["ID"]
    d = h[h["ID"].isin(set(ok))].copy()
    d["月"] = d["競走年月日"] // 100 % 100
    d["クラス"] = [race_cls(n, c) if c in NANKAN else None for n, c in zip(d["レース名"], d["競馬場"])]
    pts = []
    for r in d.itertuples(index=False):
        if r.競馬場 not in NANKAN or not (1 <= (r.着順 if pd.notna(r.着順) else 0) <= 5):
            pts.append(0.0)
            continue
        w, kind = win_points(r.競走種類名称, r.レース名, r.条件, r.月, r.クラス)
        pts.append(0.0 if pd.isna(w) else w * RATIO[kind][int(r.着順) - 1])
    d["獲得P"] = pts
    d["持ちP"] = d.groupby("ID")["獲得P"].cumsum() - d["獲得P"]  # そのレースの前までの合計
    # 他地区に出たことがある馬は除く (ポイントが変わらないのでそのままでもよいが, 移籍の影響を避ける)
    other = d[~d["競馬場"].isin(NANKAN)]["ID"].unique()
    d = d[~d["ID"].isin(other)]
    # 評価: 2025年〜の一般戦 B・C級
    e = d[(d["競走年月日"] >= 20250101) & d["クラス"].isin(["B1", "B2", "B3", "C1", "C2", "C3"]) &
          ~d["条件"].astype(str).str.contains("2歳|3歳(?!以上)", regex=True)].copy()
    lo_hi = [bounds(c, a, m) for c, a, m in zip(e["クラス"], e["齢"], e["月"])]
    e["下限"], e["上限"] = [x[0] for x in lo_hi], [x[1] for x in lo_hi]
    e["位置"] = (e["持ちP"] - e["下限"]) / (e["上限"] - e["下限"])
    e = e[e["位置"].notna() & e["人気"].notna() & e["着順"].notna()]
    # 人気から見込む勝率: 南関東 2025年〜 の全出走 (同じ人気・頭数帯) の勝率
    allr = h[h["競馬場"].isin(NANKAN) & (h["競走年月日"] >= 20250101) & h["人気"].notna()].copy()
    for x in (allr, e):
        x["頭数帯"] = pd.cut(pd.to_numeric(x["頭数"], errors="coerce"), [0, 9, 12, 20], labels=["〜9", "10〜12", "13〜"])
        x["人気c"] = x["人気"].clip(upper=10)
    base = allr.assign(勝=allr["着順"] == 1).groupby(["人気c", "頭数帯"], observed=True)["勝"].mean().rename("期待")
    e = e.merge(base, on=["人気c", "頭数帯"], how="left")
    e["勝"] = (e["着順"] == 1).astype(int)
    e["3着内"] = (e["着順"] <= 3).astype(int)
    pay = pd.concat([read(f, usecols=KEY + ["単勝組番", "単勝払戻金（円）"]) for f in PAY if (DATA / f).exists()])
    pay = pay.drop_duplicates(KEY, keep="last")
    e = e.merge(pay, on=KEY, how="left")
    e["単勝払"] = np.where(pd.to_numeric(e["単勝組番"], errors="coerce") == e["馬番"], e["単勝払戻金（円）"], 0)
    # レース内で持ちPの順位 (数えられる馬の中で)
    e["位置帯"] = pd.cut(e["位置"], [-np.inf, 0, 0.25, 0.5, 0.75, 1.0, np.inf],
                       labels=["下限未満(降級圏)", "0〜25%", "25〜50%", "50〜75%", "75〜100%", "上限超え(昇級待ち)"])
    rng = np.random.default_rng(0)

    def summ(g):
        n = len(g)
        idx = rng.integers(0, n, (1000, n))
        w, ex = g["勝"].to_numpy(), g["期待"].to_numpy()
        ae = w[idx].sum(1) / ex[idx].sum(1)
        return pd.Series({"出走": n, "勝率%": w.mean() * 100, "期待勝率%": ex.mean() * 100, "勝ちA/E": w.sum() / ex.sum(),
                          "A/Eの90%区間": "%.2f〜%.2f" % tuple(np.percentile(ae, [5, 95])),
                          "3着内率%": g["3着内"].mean() * 100, "単勝回収率%": g["単勝払"].sum() / n, "平均人気": g["人気"].mean()})

    print("ポイントを数えられる馬:", d["ID"].nunique(), "頭 / 評価に使う出走:", len(e), "(", e["競走年月日"].min(), "〜", e["競走年月日"].max(), ")")
    t = e.groupby("位置帯", observed=True).apply(summ, include_groups=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\n== クラス内の位置ごと (2025年〜, 南関東の一般戦 B・C級)\n", t.round(2).to_string())
    t2 = e.assign(級=e["クラス"].str[0]).groupby(["級", "位置帯"], observed=True).apply(summ, include_groups=False)
    print("\n== 級ごと\n", t2.round(2).to_string())
    e["人気帯"] = np.select([e["人気"] <= 3, e["人気"] <= 6], ["1〜3番人気", "4〜6番人気"], "7番人気以下")
    t3 = e.groupby(["人気帯", "位置帯"], observed=True).apply(summ, include_groups=False)
    print("\n== 人気帯ごと\n", t3.round(2).to_string())
    pd.concat([t.reset_index().assign(区分="全体"), t2.reset_index().assign(区分="級"), t3.reset_index().assign(区分="人気帯")]) \
        .round(3).to_csv(OUT / "np_summary.csv", index=False, encoding="utf-8-sig")
    e[KEY + ["馬番", "馬名", "齢", "クラス", "持ちP", "下限", "上限", "位置", "人気", "着順", "期待", "単勝払"]] \
        .to_csv(OUT / "np_horses.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
