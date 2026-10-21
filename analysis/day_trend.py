"""1日の中で, 前半のレースの傾向 (1番人気が勝つ・2番人気が勝つ・荒れる・内枠・前残り) が後半のレースでも続くかを確かめる.

使い方: python3 analysis/day_trend.py <data>
対象: 2021年〜2026年10月, ばんえいを除く14場. 1日8レース以上の開催日.
前半 = その日の 1R〜(レース数の半分)R, 後半 = 残り.
各レースの値から「競馬場×年×レース番号」の平均を引いた差 (その日の偏り) を使い,
前半の平均と後半の平均の相関 (開催日単位) を見る. 偶然の幅は開催日をシャッフルした場合と比べる.
さらに, 前半の傾向で開催日を3つに分け, 後半の成績 (と 1番人気・2番人気の単勝回収率) を比べる.
出力: analysis/out/dt_*.csv
"""
import sys
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
METRICS = {
    "1番人気が勝つ": "fav1",
    "2番人気が勝つ": "fav2",
    "3番人気が勝つ": "fav3",
    "4番人気以下が勝つ": "fav4p",
    "1番人気が3着内": "fav1_t3",
    "３連単が万馬券": "manba",
    "勝ち馬の人気(対数)": "logpop",
    "3着内の内枠度": "inner",
    "3着内の前目度": "front",
}


def read(f, **kw):
    return pd.read_csv(DATA / f, encoding="utf-8-sig", low_memory=False, **kw)


def load():
    rows = []
    for hf, rf, pf in FILES:
        if not all((DATA / f).exists() for f in (hf, rf, pf)):
            continue
        h = read(hf, usecols=KEY + ["馬番", "人気", "着順"])
        r = read(rf, usecols=KEY + CORNERS)
        p = read(pf, usecols=KEY + ["単勝払戻金（円）", "３連単払戻金（円）"])
        for c in ["馬番", "人気", "着順"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h = h[h["競馬場"] != "帯広ば"].dropna().drop_duplicates(KEY + ["馬番"])
        h["頭数"] = h.groupby(KEY)["馬番"].transform("size")
        h = h[h["頭数"] >= 5]
        st = T.styles(r.drop_duplicates(KEY))[KEY + ["馬番", "4角位置"]]
        h = h.merge(st, on=KEY + ["馬番"], how="left")
        h["内枠度"] = 1 - (h["馬番"] - 1) / (h["頭数"] - 1)
        h["前目度"] = 1 - (h["4角位置"] - 1) / (h["頭数"] - 1)
        g = h.groupby(KEY)
        win = h[h["着順"] == 1].drop_duplicates(KEY).set_index(KEY)
        t3 = h[h["着順"] <= 3].groupby(KEY)[["内枠度", "前目度"]].mean()
        al = g[["内枠度", "前目度"]].mean()
        x = pd.DataFrame({"頭数": g["頭数"].first()})
        wp = win["人気"].reindex(x.index)
        x["fav1"], x["fav2"], x["fav3"] = (wp == 1) * 1.0, (wp == 2) * 1.0, (wp == 3) * 1.0
        x["fav4p"] = (wp >= 4) * 1.0
        x["logpop"] = np.log(wp)
        f1 = h[h["人気"] == 1].drop_duplicates(KEY).set_index(KEY)["着順"].reindex(x.index)
        x["fav1_t3"] = (f1 <= 3) * 1.0
        x["inner"] = (t3["内枠度"] - al["内枠度"]).reindex(x.index)
        x["front"] = (t3["前目度"] - al["前目度"]).reindex(x.index)
        pp = p.drop_duplicates(KEY).set_index(KEY).reindex(x.index)
        x["manba"] = (pd.to_numeric(pp["３連単払戻金（円）"], errors="coerce") >= 10000) * 1.0
        tan = pd.to_numeric(pp["単勝払戻金（円）"], errors="coerce")
        x["単勝1人気"] = np.where(wp == 1, tan, 0)
        x["単勝2人気"] = np.where(wp == 2, tan, 0)
        x = x[wp.notna() & tan.notna()]
        rows.append(x.reset_index())
    d = pd.concat(rows).drop_duplicates(KEY)
    d["年"] = d["競走年月日"] // 10000
    d["R数"] = d.groupby(["競馬場", "競走年月日"])["レース番号"].transform("size")
    d = d[d["R数"] >= 8].copy()
    d["順"] = d.groupby(["競馬場", "競走年月日"])["レース番号"].rank()
    d["後半"] = d["順"] > d["R数"] // 2
    for m in METRICS.values():
        d[m + "_偏り"] = d[m] - d.groupby(["競馬場", "年", "レース番号"])[m].transform("mean")
    return d


def day_table(d):
    agg = {m + "_偏り": "mean" for m in METRICS.values()}
    a = d.groupby(["競馬場", "競走年月日", "後半"]).agg(agg).unstack("後半")
    return a


def corr_with_null(a, m, rng, n_perm=500):
    x, y = a[(m + "_偏り", False)], a[(m + "_偏り", True)]
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    idx = rng.integers(0, len(x), (500, len(x)))
    bs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return r, np.percentile(bs, [5, 95]), np.mean(np.abs(null) >= abs(r)), len(x)


def main():
    d = load()
    print(f"対象: {d['競走年月日'].min()}〜{d['競走年月日'].max()}, {d.groupby(['競馬場', '競走年月日']).ngroups}開催日, {len(d)}レース")
    a = day_table(d)
    rng = np.random.default_rng(0)
    rows = []
    for name, m in METRICS.items():
        r, ci, pv, n = corr_with_null(a, m, rng)
        rows.append({"項目": name, "開催日": n, "前半と後半の相関": r, "90%区間": f"{ci[0]:+.3f}〜{ci[1]:+.3f}",
                     "シャッフルでこれ以上の割合": pv})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== (1) 前半の偏りと後半の偏りの相関 (開催日単位)\n", t.round(3).to_string(index=False))
    t.round(4).to_csv(OUT / "dt_corr.csv", index=False, encoding="utf-8-sig")

    # (2) 前半の回数で分けて, 後半の率を見る
    first = d[~d["後半"]].groupby(["競馬場", "競走年月日"])
    second = d[d["後半"]].groupby(["競馬場", "競走年月日"])
    out = []
    for name, m in [("1番人気が勝つ", "fav1"), ("2番人気が勝つ", "fav2"), ("4番人気以下が勝つ", "fav4p"), ("３連単が万馬券", "manba")]:
        f = first[m].mean()
        s = second[[m, m + "_偏り", "単勝1人気", "単勝2人気"]].agg(["mean", "size"])
        s = pd.DataFrame({"後半の率": s[(m, "mean")], "後半の偏り": s[(m + "_偏り", "mean")],
                          "後半の単勝1番人気": s[("単勝1人気", "mean")], "後半の単勝2番人気": s[("単勝2人気", "mean")],
                          "後半R数": s[(m, "size")]})
        s["前半の率"] = f.reindex(s.index)
        s["前半"] = pd.qcut(s["前半の率"].rank(method="first"), 3, labels=["前半で少ない日", "中くらい", "前半で多い日"])
        g = s.groupby("前半", observed=True)
        res = pd.DataFrame({"開催日": g.size(), "前半の率%": g["前半の率"].mean() * 100,
                            "後半の率%": g.apply(lambda x: np.average(x["後半の率"], weights=x["後半R数"])) * 100,
                            "後半の偏り(ポイント)": g.apply(lambda x: np.average(x["後半の偏り"], weights=x["後半R数"])) * 100,
                            "後半 1番人気単勝回収率%": g.apply(lambda x: np.average(x["後半の単勝1番人気"], weights=x["後半R数"])),
                            "後半 2番人気単勝回収率%": g.apply(lambda x: np.average(x["後半の単勝2番人気"], weights=x["後半R数"]))})
        res.insert(0, "項目", name)
        out.append(res.reset_index())
    t2 = pd.concat(out)
    print("\n== (2) 前半の結果で開催日を3つに分けたときの後半の成績\n", t2.round(1).to_string(index=False))
    t2.round(2).to_csv(OUT / "dt_tercile.csv", index=False, encoding="utf-8-sig")

    # (3) 競馬場ごとの相関 (1番人気が勝つ・万馬券・内枠度・前目度)
    rows = []
    for c, ac in a.groupby(level="競馬場"):
        row = {"競馬場": c, "開催日": len(ac)}
        for name, m in [("1番人気が勝つ", "fav1"), ("2番人気が勝つ", "fav2"), ("３連単が万馬券", "manba"),
                        ("3着内の内枠度", "inner"), ("3着内の前目度", "front")]:
            x, y = ac[(m + "_偏り", False)], ac[(m + "_偏り", True)]
            ok = x.notna() & y.notna()
            row[name] = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 20 else np.nan
        rows.append(row)
    t3 = pd.DataFrame(rows)
    print("\n== (3) 競馬場ごとの相関 (開催日単位. 0.06程度までは偶然でも出る)\n", t3.round(3).to_string(index=False))
    t3.round(4).to_csv(OUT / "dt_course.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
