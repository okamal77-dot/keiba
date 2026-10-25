"""前残りの日に, 上位人気の相手を「逃げ・先行できる馬」に入れ替えると回収率が上がるかを確かめる.

使い方: python3 analysis/pace_bias_bets.py <data>
対象: 2021年〜2026年10月, ばんえいを除く14場, 5頭以上.
当日の前残り度: 同じ日・同じ競馬場の, それより前のレースで 3着内の馬の4角の前目度 − 全馬の平均 (extra_features.add_bias と同じ).
  前のレースが3つ以上あるレースだけを使い, 前残り度で3つに分ける (前残りの日 / ふつう / 差しの日. 全期間の3分位).
先行力: その馬の近5走の 4角位置/頭数 の平均 (小さいほど前). 近走が無い馬は 0.5.
買い方 (人気は確定人気. 100円ずつ):
  馬単  : 人気順 1→2〜5番人気 (4点)       / 1→ 2〜7番人気のうち先行力の上位4頭 (4点)
  ３連単: 人気順 1→2,3,4→2,3,4 (6点)     / 1→ 2〜7番人気のうち先行力の上位3頭 → 同 (6点)
  ３連複: 人気1〜5番の5頭BOX (10点)      / 1番人気 + 2〜7番人気のうち先行力の上位4頭の5頭BOX (10点)
  ほかに 1番人気を「先行力の上位」に限る場合 (1番人気が先行馬のときだけ買う) も見る.
3期間 (2021-22, 2023-24, 2025-26) で同じ向きかを確かめ, 人気順との差の90%区間はレース単位のブートストラップ.
出力: analysis/out/pbb_*.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import trifecta_factors as T  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
FILES = [(f"{y}_horselist_全月まとめ.csv", f"race{y}.csv", f"pay{y}.csv") for y in range(2021, 2026)] + [
    ("2026_horselist_1-8月まとめ.csv", "race2026.csv", "pay2026.csv"),
    ("202609_horselist.csv", "202609_racelist.csv", "202609_payback.csv"),
    ("202610_horselist.csv", "202610_racelist.csv", "202610_payback.csv")]
CORNERS = [f"コーナー通過順{i}" for i in range(1, 9)]
P3 = ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"]
F3 = ["３連複組番馬番1", "３連複組番馬番2", "３連複組番馬番3", "３連複払戻金（円）"]
P2 = ["馬単組番1", "馬単組番2", "馬単払戻金（円）"]


def read(f, **kw):
    return pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, **kw)


def load():
    hs, ps = [], []
    for hf, rf, pf in FILES:
        if not all((DATA / f).exists() for f in (hf, rf, pf)):
            continue
        h = read(hf, usecols=KEY + ["馬番", "馬名", "生年月日", "人気", "着順"])
        for c in ["馬番", "人気", "着順"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h = h[(h["競馬場"] != "帯広ば") & h["着順"].notna() & h["人気"].notna()]
        r = read(rf, usecols=KEY + CORNERS).drop_duplicates(KEY)
        st = T.styles(r[r["競馬場"] != "帯広ば"])[KEY + ["馬番", "4角位置"]]
        hs.append(h.merge(st, on=KEY + ["馬番"], how="left"))
        ps.append(read(pf, usecols=KEY + P3 + F3 + P2))
    h = pd.concat(hs, ignore_index=True).drop_duplicates(KEY + ["馬番"])
    p = pd.concat(ps).drop_duplicates(KEY).dropna(subset=["３連単組番馬番1", "馬単組番1"]).set_index(KEY)
    h["頭数"] = h.groupby(KEY)["馬番"].transform("size")
    h = h[h["頭数"] >= 5]
    h = h[h.set_index(KEY).index.isin(p.index)].copy()
    h["日付"] = pd.to_datetime(h["競走年月日"].astype(str))
    h["馬ID"] = h["馬名"] + "_" + h["生年月日"].astype(str)
    h = h.sort_values(["馬ID", "日付", "レース番号"])
    h["位置率"] = h["4角位置"] / h["頭数"]
    h["先行力"] = h.groupby("馬ID")["位置率"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    # 当日の前残り度 (extra_features.add_bias と同じ作り方)
    h["前目度"] = 1 - (h["4角位置"] - 1) / (h["頭数"] - 1)
    t3 = h[h["着順"] <= 3].groupby(KEY)["前目度"].mean()
    al = h.groupby(KEY)["前目度"].mean()
    rk = (t3 - al).rename("前有利").reset_index().sort_values(KEY)
    g = rk.groupby(["競馬場", "競走年月日"])["前有利"]
    rk["当日前残り"] = g.transform(lambda s: s.shift(1).expanding().mean())
    rk["前のR数"] = g.transform(lambda s: s.shift(1).notna().astype(float).expanding().sum()).fillna(0)
    return h, p, rk.set_index(KEY)


def perms(a, mates):
    return {(a, x, y) for x in mates for y in mates if x != y}


def races(h, p, rk):
    rows = []
    for k, x in h.groupby(KEY):
        if k not in rk.index or rk.loc[k, "前のR数"] < 3 or pd.isna(rk.loc[k, "当日前残り"]):
            continue
        x = x.sort_values(["人気", "馬番"])
        nos = x["馬番"].astype(int).tolist()
        if len(nos) < 7:
            continue
        sen = dict(zip(x["馬番"].astype(int), x["先行力"]))
        fav = nos[0]
        cand = sorted(nos[1:7], key=lambda n: sen[n])          # 2〜7番人気を先行力の順に
        pr = p.loc[k]
        r3 = tuple(int(pr[c]) for c in P3[:3])
        f3 = frozenset(int(pr[c]) for c in F3[:3])
        r2 = (int(pr[P2[0]]), int(pr[P2[1]]))
        fav_front = sen[fav] <= np.median(list(sen.values()))
        bets = {
            "馬単 人気順": ({(fav, m) for m in nos[1:5]}, "馬単"),
            "馬単 先行4頭": ({(fav, m) for m in cand[:4]}, "馬単"),
            "３連単 人気順": (perms(fav, nos[1:4]), "３連単"),
            "３連単 先行3頭": (perms(fav, cand[:3]), "３連単"),
            "３連複 人気1〜5": ({frozenset(c) for c in combinations(nos[:5], 3)}, "３連複"),
            "３連複 1番人気+先行4頭": ({frozenset(c) for c in combinations([fav] + cand[:4], 3)}, "３連複"),
        }
        row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "当日前残り": rk.loc[k, "当日前残り"],
               "1番人気が先行馬": fav_front, "入れ替わった頭数": len(set(cand[:4]) - set(nos[1:5]))}
        for name, (s, kind) in bets.items():
            hit = {"馬単": r2, "３連単": r3, "３連複": f3}[kind]
            pay = {"馬単": pr[P2[2]], "３連単": pr[P3[3]], "３連複": pr[F3[3]]}[kind]
            row[name + "_点"] = len(s)
            row[name + "_払"] = pay if hit in s else 0
        rows.append(row)
    return pd.DataFrame(rows)


def roi(g, b):
    return g[b + "_払"].sum() / max(g[b + "_点"].sum() * 100, 1) * 100


def main():
    h, p, rk = load()
    d = races(h, p, rk)
    d["期"] = np.select([d["競走年月日"] < 20230101, d["競走年月日"] < 20250101], ["2021-22", "2023-24"], "2025-26")
    q = d["当日前残り"].quantile([1 / 3, 2 / 3]).values
    d["当日"] = np.select([d["当日前残り"] >= q[1], d["当日前残り"] <= q[0]], ["前残りの日", "差しの日"], "ふつう")
    d.to_csv(OUT / "pbb_races.csv", index=False, encoding="utf-8-sig")
    print(f"対象 {len(d)}レース ({d['競走年月日'].min()}〜{d['競走年月日'].max()}). 前残り度の境目 {q.round(3)}")
    pairs = [("馬単 人気順", "馬単 先行4頭"), ("３連単 人気順", "３連単 先行3頭"), ("３連複 人気1〜5", "３連複 1番人気+先行4頭")]
    rng = np.random.default_rng(0)
    rows = []
    for day in ["前残りの日", "ふつう", "差しの日", "全体"]:
        for sub_name, sub in [("全レース", None), ("1番人気が先行馬", True)]:
            g = d if day == "全体" else d[d["当日"] == day]
            if sub is not None:
                g = g[g["1番人気が先行馬"]]
            for a, b in pairs:
                diff = ((g[b + "_払"] - g[a + "_払"]) / (g[a + "_点"] * 100) * 100).to_numpy()
                bs = diff[rng.integers(0, len(diff), (1000, len(diff)))].mean(1)
                row = {"当日": day, "対象": sub_name, "買い方": b.split()[0], "レース数": len(g),
                       "人気順%": roi(g, a), "先行馬に入れ替え%": roi(g, b), "差": diff.mean(),
                       "差の90%区間": f"{np.percentile(bs, 5):+.1f}〜{np.percentile(bs, 95):+.1f}",
                       "人気順の的中%": (g[a + "_払"] > 0).mean() * 100, "入れ替えの的中%": (g[b + "_払"] > 0).mean() * 100}
                for per, gg in g.groupby("期"):
                    row[f"差 {per}"] = roi(gg, b) - roi(gg, a)
                rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(1).to_string(index=False))
    t.round(2).to_csv(OUT / "pbb_summary.csv", index=False, encoding="utf-8-sig")
    # 前残りの日の競馬場別 (馬単・３連複)
    g = d[d["当日"] == "前残りの日"]
    c = g.groupby("競馬場").apply(lambda x: pd.Series({
        "レース数": len(x), "馬単 人気順%": roi(x, "馬単 人気順"), "馬単 先行4頭%": roi(x, "馬単 先行4頭"),
        "３連複 人気%": roi(x, "３連複 人気1〜5"), "３連複 先行%": roi(x, "３連複 1番人気+先行4頭")}))
    print("\n== 前残りの日の競馬場別\n", c.round(0).to_string())
    c.round(1).to_csv(OUT / "pbb_course.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
