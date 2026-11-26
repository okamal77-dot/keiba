"""1強のレースで, 1〜3番人気の脚質によって ３連単 1→4,5,6→4,5,6 (B1) や1番人気の勝率が変わるか.

使い方: python3 analysis/ashi_1kyo.py
対象: 確定オッズのある 2026年3月・6〜11月, ばんえいを除く, 6頭以上, オッズの形が 1強 のレース (b1_features.py と同じ).
脚質 (レース前に分かるもの): その馬の近5走 (当該レースは含まない) の脚質で最も多いもの. 同数なら前の脚質.
  1走ごとの脚質は trifecta_factors.styles (最初のコーナー1番手=逃げ, 最終コーナー 35%以内=先行, 70%以内=差し, それより後ろ=追込).
  近走がない馬は「不明」.
見るもの: 1番人気の勝率, 1番人気が3着以内の率, 4〜6番人気が2頭とも2・3着に入る率,
  B1 (6点) の的中率と回収率, 参考に ３連単 1→2,3→全 (C1) の的中率と回収率.
出力: analysis/out/ashi_1kyo.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], str(Path(__file__).parent.parent / "data")]
import odds_bets as OB  # noqa: E402
import trifecta_factors as T  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
KEY = ["競馬場", "競走年月日", "レース番号"]
ORDER = ["逃げ", "先行", "差し", "追込"]


def pre_style(h):
    h = h.sort_values(["馬ID", "日付", "レース番号"]).copy()
    code = h["脚質"].map({s: i for i, s in enumerate(ORDER)})
    out = []
    for _, s in code.groupby(h["馬ID"]):
        v = s.to_numpy()
        res = []
        for i in range(len(v)):
            past = v[max(0, i - 5):i]
            past = past[~np.isnan(past)]
            if len(past) == 0:
                res.append("不明")
            else:
                cnt = np.bincount(past.astype(int), minlength=4)
                res.append(ORDER[int(np.argmax(cnt))])  # 同数なら前の脚質 (argmax は最初の最大)
        out.append(pd.Series(res, index=s.index))
    h["事前脚質"] = pd.concat(out)
    return h


def main():
    h, _, _ = T.load()
    h = h[h["競馬場"] != "帯広ば"]
    h = pre_style(h)
    st = h.set_index(KEY + ["馬番"])["事前脚質"]
    fs = sorted((DATA / "odds").glob("2026*_odds.csv"), key=lambda f: "final" in f.name)
    od = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["賭式", "番号1", "オッズ"]).query("賭式 == '単勝'")
                    for f in fs]).drop_duplicates(KEY + ["番号1"], keep="last")
    od["o"] = pd.to_numeric(od["オッズ"], errors="coerce")
    od = od[(od["o"] > 0) & (od["競馬場"] != "帯広ば")]
    pf = [DATA / "pay2026.csv"] + [DATA / f"2026{m:02d}_payback.csv" for m in (9, 10, 11)]
    pay = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in pf if f.exists()]).drop_duplicates(KEY, keep="last")
    pay = pay.dropna(subset=["３連単組番馬番1"]).set_index(KEY)
    rows = []
    for k, g in od.groupby(KEY):
        if k not in pay.index or len(g) < 6:
            continue
        g = g.sort_values(["o", "番号1"])
        nos, oo = g["番号1"].astype(int).tolist(), g["o"].tolist()
        if OB.shape_of(oo)[0] != "1強":
            continue
        pr = pay.loc[k]
        r3 = [int(pr[f"３連単組番馬番{i}"]) for i in (1, 2, 3)]
        n = len(nos)
        b1 = r3[0] == nos[0] and r3[1] in nos[3:6] and r3[2] in nos[3:6]
        c1 = r3[0] == nos[0] and r3[1] in nos[1:3]
        sty = [st.get(k + (x,), "不明") for x in nos[:3]]
        rows.append({"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "頭数": n,
                     "1番人気脚質": sty[0], "2番人気脚質": sty[1], "3番人気脚質": sty[2],
                     "逃げ型の数(1-3人気)": sum(s == "逃げ" for s in sty),
                     "前(逃げ先行)の数(1-3人気)": sum(s in ("逃げ", "先行") for s in sty),
                     "1人気1着": r3[0] == nos[0], "1人気3着内": nos[0] in r3,
                     "2,3人気が3着外": nos[1] not in r3 and nos[2] not in r3,
                     "B1的中": b1, "B1払": pr["３連単払戻金（円）"] if b1 else 0,
                     "C1的中": c1, "C1払": pr["３連単払戻金（円）"] if c1 else 0, "C1点": 2 * (n - 2)})
    d = pd.DataFrame(rows)
    d["期"] = np.where(d["競走年月日"] < 20260901, "前半", "後半")
    rng = np.random.default_rng(0)

    def summ(x):
        m = len(x)
        ret = x["B1払"].to_numpy()
        r = {"レース": m, "1人気勝率%": x["1人気1着"].mean() * 100, "1人気3着内%": x["1人気3着内"].mean() * 100,
             "2,3人気とも3着外%": x["2,3人気が3着外"].mean() * 100,
             "B1的中率%": x["B1的中"].mean() * 100, "B1回収率%": ret.sum() / (m * 600) * 100,
             "C1的中率%": x["C1的中"].mean() * 100, "C1回収率%": x["C1払"].sum() / (x["C1点"].sum() * 100) * 100}
        if m >= 60:
            idx = rng.integers(0, m, (1000, m))
            r["B1の90%区間"] = "%.0f〜%.0f" % tuple(np.percentile(ret[idx].sum(1) / (m * 600) * 100, [5, 95]))
        for p, gp in x.groupby("期"):
            r[f"B1回収率%({p})"] = gp["B1払"].sum() / (len(gp) * 600) * 100
        return r

    out = [{"分け方": "全体", "値": "1強すべて", **summ(d)}]
    for col in ["1番人気脚質", "2番人気脚質", "3番人気脚質", "逃げ型の数(1-3人気)", "前(逃げ先行)の数(1-3人気)"]:
        for v, g in d.groupby(col):
            if len(g) >= 60:
                out.append({"分け方": col, "値": str(v), **summ(g)})
    d["2,3番人気"] = d["2番人気脚質"].where(d["2番人気脚質"] != "不明") + "・" + d["3番人気脚質"].where(d["3番人気脚質"] != "不明")
    for v, g in d.groupby("2,3番人気"):
        if len(g) >= 60:
            out.append({"分け方": "2,3番人気の脚質", "値": v, **summ(g)})
    d["1人気前×2,3人気前の数"] = np.where(d["1番人気脚質"].isin(["逃げ", "先行"]), "1人気が前", "1人気が後ろ") + "/2,3人気の前" + \
        ((d["2番人気脚質"].isin(["逃げ", "先行"])).astype(int) + (d["3番人気脚質"].isin(["逃げ", "先行"])).astype(int)).astype(str) + "頭"
    for v, g in d.groupby("1人気前×2,3人気前の数"):
        if len(g) >= 60:
            out.append({"分け方": "1人気の位置×2,3人気の前の数", "値": v, **summ(g)})
    t = pd.DataFrame(out)
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.max_columns", 20)
    print("対象: 1強", len(d), "レース")
    print(t.round(1).to_string(index=False))
    t.round(2).to_csv(ROOT / "out" / "ashi_1kyo.csv", index=False, encoding="utf-8-sig")
    d.to_csv(ROOT / "out" / "ashi_1kyo_races.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
