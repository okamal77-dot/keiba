"""３連単の流し方を確定人気で比べる (的中率・回収率, 競馬場別・馬場別).

使い方: python3 analysis/tri_flow.py
対象: 2022年1月〜2026年11月, ばんえいを除く, 6頭以上, ３連単の払戻があるレース. 人気は確定人気 (同じ人気は馬番の小さい方を先).
買い方 (各100円. P=人気):
  A1 「1番人気を捨てて 2,3番人気から」: 1着 P2・P3 → 2着 P2・P3 → 3着 1番人気以外の全部   (2×(頭数-3) 点)
  A2 「1番人気を捨てて 2,3番人気から」: 1着 P2・P3 → 2着・3着 1番人気以外の全部            (2×(頭数-2)×(頭数-3) 点)
  B1 「1番人気から 4,5,6番人気へ」   : 1着 P1 → 2着・3着 P4・P5・P6                       (6点)
  B2 「1番人気から 4,5,6番人気へ」   : 1着 P1 → 2着 P4・P5・P6 → 3着 全部                   (3×(頭数-2) 点)
  C1 「1番人気→2,3番人気→全部」       : 1着 P1 → 2着 P2・P3 → 3着 全部                     (2×(頭数-2) 点)
  C2 「1番人気→2,3番人気→全部」(裏も) : 1着 P1 → 2着 P2・P3 → 3着 全部 + 1着 P2・P3 → 2着 P1 → 3着 全部 (4×(頭数-2) 点)
  参考 今のルール                     : 1着 P1 → 2着・3着 P2・P3・P4                       (6点)
回収率の90%区間はレース単位のブートストラップ.
出力: analysis/out/tri_flow_summary.csv (全体・競馬場・馬場・競馬場×馬場)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
KEY = ["競馬場", "競走年月日", "レース番号"]
HORSE = [f"{y}_horselist_全月まとめ.csv" for y in (2022, 2023, 2024, 2025)] + \
    ["2026_horselist_1-8月まとめ.csv", "202609_horselist.csv", "202610_horselist.csv", "202611_horselist.csv"]
PAY = [f"pay{y}.csv" for y in (2022, 2023, 2024, 2025, 2026)] + ["202609_payback.csv", "202610_payback.csv", "202611_payback.csv"]
RACE = [f"race{y}.csv" for y in (2022, 2023, 2024, 2025, 2026)] + ["202609_racelist.csv", "202610_racelist.csv", "202611_racelist.csv"]
PLANS = ["A1 2,3→2,3→全(1番人気なし)", "A2 2,3→全→全(1番人気なし)", "B1 1→4,5,6→4,5,6", "B2 1→4,5,6→全",
         "C1 1→2,3→全", "C2 1→2,3→全 +裏(2,3→1→全)", "参考 今のルール 1→2,3,4→2,3,4"]
BABA = ["良", "稍重", "重", "不良"]


def read(f, **kw):
    return pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, **kw)


def plan(code, p1, p2, p3, n):
    """人気順位 (1始まり) の着順 (p1,p2,p3) が当たりか と 点数"""
    if code[0:2] == "A1":
        return (p1 in (2, 3) and p2 in (2, 3) and p3 != 1), 2 * (n - 3)
    if code[0:2] == "A2":
        return (p1 in (2, 3) and p2 != 1 and p3 != 1), 2 * (n - 2) * (n - 3)
    if code[0:2] == "B1":
        return (p1 == 1 and p2 in (4, 5, 6) and p3 in (4, 5, 6)), 6
    if code[0:2] == "B2":
        return (p1 == 1 and p2 in (4, 5, 6)), 3 * (n - 2)
    if code[0:2] == "C1":
        return (p1 == 1 and p2 in (2, 3)), 2 * (n - 2)
    if code[0:2] == "C2":
        return ((p1 == 1 and p2 in (2, 3)) or (p1 in (2, 3) and p2 == 1)), 4 * (n - 2)
    return (p1 == 1 and p2 in (2, 3, 4) and p3 in (2, 3, 4)), 6


def main():
    h = pd.concat([read(f, usecols=KEY + ["馬番", "人気"]) for f in HORSE if (DATA / f).exists()])
    h = h.drop_duplicates(KEY + ["馬番"], keep="last")
    for c in ["馬番", "人気"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h[h["人気"].notna() & (h["競馬場"] != "帯広ば")]
    pay = pd.concat([read(f, usecols=KEY + ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3", "３連単払戻金（円）"])
                     for f in PAY if (DATA / f).exists()]).drop_duplicates(KEY, keep="last").dropna()
    rl = pd.concat([read(f, usecols=KEY + ["馬場"]) for f in RACE if (DATA / f).exists()]).drop_duplicates(KEY, keep="last")
    rows = []
    for k, g in h.groupby(KEY):
        rows.append((*k, len(g), dict(zip(g.sort_values(["人気", "馬番"])["馬番"].astype(int), range(1, len(g) + 1)))))
    r = pd.DataFrame(rows, columns=KEY + ["頭数", "順位"])
    r = r[r["頭数"] >= 6].merge(pay, on=KEY).merge(rl, on=KEY, how="left")
    r["馬場"] = r["馬場"].astype(str).str.strip()
    out = {"競馬場": r["競馬場"], "競走年月日": r["競走年月日"], "馬場": r["馬場"], "頭数": r["頭数"]}
    for code in PLANS:
        hit, pts = [], []
        for rk, n, a, b, c in zip(r["順位"], r["頭数"], r["３連単組番馬番1"], r["３連単組番馬番2"], r["３連単組番馬番3"]):
            p = [rk.get(int(x), 99) for x in (a, b, c)]
            hh, pp = plan(code, *p, n)
            hit.append(hh)
            pts.append(pp)
        out[code + "_的中"] = np.array(hit)
        out[code + "_点"] = np.array(pts)
        out[code + "_払"] = np.where(hit, r["３連単払戻金（円）"], 0)
    d = pd.DataFrame(out)
    rng = np.random.default_rng(0)

    def summ(x, code, ci=False):
        inv = x[code + "_点"].to_numpy() * 100
        ret = x[code + "_払"].to_numpy()
        res = {"レース": len(x), "的中率%": x[code + "_的中"].mean() * 100, "平均点数": x[code + "_点"].mean(),
               "回収率%": ret.sum() / max(inv.sum(), 1) * 100,
               "的中時の平均配当": ret[ret > 0].mean() if (ret > 0).any() else np.nan}
        if ci and len(x) >= 50:
            idx = rng.integers(0, len(x), (1000, len(x)))
            rr = ret[idx].sum(1) / np.maximum(inv[idx].sum(1), 1) * 100
            res["回収率90%区間"] = "%.0f〜%.0f" % tuple(np.percentile(rr, [5, 95]))
        return res

    rows = []
    for code in PLANS:
        rows.append({"区分": "全体", "競馬場": "全体", "馬場": "全体", "買い方": code, **summ(d, code, ci=True)})
        for y, g in d.groupby(d["競走年月日"] // 10000):
            rows.append({"区分": "年", "競馬場": str(y), "馬場": "全体", "買い方": code, **summ(g, code)})
        for c, g in d.groupby("競馬場"):
            rows.append({"区分": "競馬場", "競馬場": c, "馬場": "全体", "買い方": code, **summ(g, code, ci=True)})
        for b in BABA:
            g = d[d["馬場"] == b]
            rows.append({"区分": "馬場", "競馬場": "全体", "馬場": b, "買い方": code, **summ(g, code, ci=True)})
        for (c, b), g in d[d["馬場"].isin(BABA)].groupby(["競馬場", "馬場"]):
            rows.append({"区分": "競馬場×馬場", "競馬場": c, "馬場": b, "買い方": code, **summ(g, code, ci=True)})
    t = pd.DataFrame(rows)
    t.round(2).to_csv(ROOT / "out" / "tri_flow_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 20)
    print("対象:", len(d), "レース", d["競走年月日"].min(), "〜", d["競走年月日"].max())
    print(t[t["区分"] == "全体"].drop(columns=["区分", "競馬場", "馬場"]).round(1).to_string(index=False))
    for kind, col in [("年", "競馬場"), ("競馬場", "競馬場"), ("馬場", "馬場")]:
        x = t[t["区分"] == kind]
        print(f"\n== {kind}ごとの回収率%")
        print(x.pivot(index=col, columns="買い方", values="回収率%").round(0).to_string())
        print(f"== {kind}ごとの的中率%")
        print(x.pivot(index=col, columns="買い方", values="的中率%").round(1).to_string())


if __name__ == "__main__":
    main()
