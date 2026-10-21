"""オッズの形のルール (odds_bets.py) の買い目で, 1レースの投資を最大1,000円 (３連単+馬単の合計) としたときの
配分の仕方 (券種の割合・買い目ごとの金額) で回収率がどう変わるかを比べる.

使い方: python3 analysis/stake_alloc.py <data>
  対象: 確定オッズ (data/odds/*_odds.csv) と払戻 (data/2026MM_payback.csv, data/entries/*_payback*.csv) がある 2026年9月〜のレース.
  買い目・見送りは odds_bets.bets と同じ (1番人気の騎手の補正・A級の6点を含む). 人気は確定単勝オッズの順.
配分 (100円単位. 端数は大きい順に配る):
  均等       : 買い目ごとに同じ金額
  払戻均等   : どれが当たっても払戻がほぼ同じになるよう, 金額をその組の確定オッズに反比例させる (確定オッズを使うので先読みあり)
  確率比例   : 単勝オッズから Harville 式で出した組の確率に比例 (単勝オッズだけで決まる. 払戻均等の近似)
  1/確率比例 : Harville の確率に反比例 (当たりにくい組ほど厚く. 単勝オッズだけで決まる. 人気どうしの組は0円になることがある)
  割安だけ   : Harville の確率 × その組のオッズ が 1 以上の組だけを均等に買う (確定オッズを使うので先読みあり)
券種の割合: ３連単と馬単の両方を買うレースで, 1,000円のうち ３連単に回す割合. 片方だけ買うレースはその券種に1,000円.
9月で傾向を見て, 10月で同じ向きかを確かめる. 回収率の90%区間はレース単位のブートストラップ.
出力: analysis/out/sa_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import odds_bets as OB  # noqa: E402
from race_profile import race_class  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = OB.KEY
BUDGET = 1000
SPLITS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
METHODS = ["均等", "払戻均等", "確率比例", "1/確率比例", "割安だけ"]


def load():
    of = sorted((DATA / "odds").glob("2026*_odds.csv")) + sorted((DATA / "entries").glob("*_odds_*.csv"))
    od = []
    for f in of:
        x = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        od.append(x[x["賭式"].isin(["単勝", "３連単", "馬単"])])
    od = pd.concat(od).drop_duplicates(KEY + ["賭式", "番号1", "番号2", "番号3"], keep="last")
    pf = [DATA / "202609_payback.csv", DATA / "202610_payback.csv"] + sorted((DATA / "entries").glob("*_payback*.csv"))
    pay = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in pf if f.exists()])
    pay = pay.drop_duplicates(KEY, keep="last").set_index(KEY)
    hf = [DATA / "202609_horselist.csv", DATA / "202610_horselist.csv"] + sorted((DATA / "entries").glob("*_horselist.csv"))
    hl = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["馬番", "騎手名"]) for f in hf if f.exists()])
    jkn = hl.drop_duplicates(KEY + ["馬番"], keep="last").set_index(KEY + ["馬番"])["騎手名"]
    rf = [DATA / "202609_racelist.csv", DATA / "202610_racelist.csv"] + sorted((DATA / "entries").glob("*_racelist.csv"))
    rl = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["レース名"]) for f in rf if f.exists()])
    names = rl.drop_duplicates(KEY, keep="last").set_index(KEY)["レース名"]
    return od, pay, jkn, names


def units(w, budget):
    """重み w に比例して budget を100円単位で配る (最大剰余法). 戻り値: 円"""
    w = np.asarray(w, float)
    n = budget // 100
    if w.sum() <= 0 or n == 0:
        return np.zeros(len(w))
    x = w / w.sum() * n
    u = np.floor(x)
    for i in np.argsort(-(x - u))[: int(n - u.sum())]:
        u[i] += 1
    return u * 100


def alloc(method, odds, prob, budget):
    if method == "均等":
        return units(np.ones(len(odds)), budget)
    if method == "払戻均等":
        return units(1 / odds, budget)
    if method == "確率比例":
        return units(prob, budget)
    if method == "1/確率比例":
        return units(1 / prob, budget)
    ok = prob * odds >= 1
    return units(ok.astype(float), budget)


def harville3(p, a, b, c):
    return p[a] * p[b] / (1 - p[a]) * p[c] / (1 - p[a] - p[b])


def harville2(p, a, b):
    return p[a] * p[b] / (1 - p[a])


def races(od, pay, jkn, names):
    jk = OB.load_jockey()
    tri_o = od[od["賭式"] == "３連単"].set_index(KEY + ["番号1", "番号2", "番号3"])["オッズ"]
    ex_o = od[od["賭式"] == "馬単"].set_index(KEY + ["番号1", "番号2"])["オッズ"]
    win = od[od["賭式"] == "単勝"]
    out = []
    for k, g in win.groupby(KEY):
        if k not in pay.index:
            continue
        g = g.assign(o=pd.to_numeric(g["オッズ"], errors="coerce"))
        g = g[g["o"] > 0].sort_values(["o", "番号1"])
        if len(g) < 5:
            continue
        nos = g["番号1"].astype(int).tolist()
        oo = g["o"].tolist()
        fav_ae = OB.jk_ae(jk, jkn.get(k + (nos[0],)), k[0], "1番人気")
        b = OB.bets(nos, oo, fav_ae, race_class(names.get(k)) if k in names.index else None)
        if not b["点数3"] and not b["点数2"]:
            continue
        q = 1 / np.array(oo)
        p = dict(zip(nos, q / q.sum()))
        n1, n2, n3, n4, n5 = nos[:5]
        mates = {6: (n2, n3, n4), 2: (n2, n3)}.get(b["点数3"], ())
        tri = [(n1, x, y) for x in mates for y in mates if x != y]
        ex = [(n1, x) for x in (n2, n3, n4, n5)] if b["点数2"] else []
        pr = pay.loc[k]
        r3 = tuple(int(pr[c]) for c in ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"]) if pd.notna(pr["３連単組番馬番1"]) else None
        r2 = (int(pr["馬単組番1"]), int(pr["馬単組番2"])) if pd.notna(pr["馬単組番1"]) else None
        t_odds = np.array([tri_o.get(k + c, np.nan) for c in tri], float)
        e_odds = np.array([ex_o.get(k + c, np.nan) for c in ex], float)
        if np.isnan(t_odds).any() or np.isnan(e_odds).any():
            continue
        out.append({"key": k, "形": b["形"],
                    "tri": tri, "t_odds": t_odds, "t_prob": np.array([harville3(p, *c) for c in tri]),
                    "t_hit": [c == r3 for c in tri], "t_pay": pr["３連単払戻金（円）"],
                    "ex": ex, "e_odds": e_odds, "e_prob": np.array([harville2(p, *c) for c in ex]),
                    "e_hit": [c == r2 for c in ex], "e_pay": pr["馬単払戻金（円）"]})
    return out


def simulate(rs, method, split):
    """split: ３連単の割合 (数値), または 形ごとの割合 {形: 割合}"""
    rows = []
    for r in rs:
        sp = split[r["形"]] if isinstance(split, dict) else split
        has3, has2 = len(r["tri"]) > 0, len(r["ex"]) > 0
        b3 = BUDGET if not has2 else (0 if not has3 else int(round(BUDGET * sp / 100)) * 100)
        b2 = BUDGET - b3 if has2 else 0
        s3 = alloc(method, r["t_odds"], r["t_prob"], b3) if has3 else np.zeros(0)
        s2 = alloc(method, r["e_odds"], r["e_prob"], b2) if has2 else np.zeros(0)
        ret3 = sum(s / 100 * r["t_pay"] for s, h in zip(s3, r["t_hit"]) if h)
        ret2 = sum(s / 100 * r["e_pay"] for s, h in zip(s2, r["e_hit"]) if h)
        rows.append({"競馬場": r["key"][0], "競走年月日": r["key"][1], "レース番号": r["key"][2], "形": r["形"],
                     "投資3": s3.sum(), "払戻3": ret3, "投資2": s2.sum(), "払戻2": ret2,
                     "投資": s3.sum() + s2.sum(), "払戻": ret3 + ret2})
    return pd.DataFrame(rows)


def ci(d, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    inv, ret = d["投資"].to_numpy(), d["払戻"].to_numpy()
    idx = rng.integers(0, len(d), (n, len(d)))
    r = ret[idx].sum(1) / np.maximum(inv[idx].sum(1), 1) * 100
    return np.percentile(r, [5, 95])


def summary(d):
    roi = lambda x: x["払戻"].sum() / max(x["投資"].sum(), 1) * 100  # noqa: E731
    lo, hi = ci(d)
    m9 = d[d["競走年月日"] < 20261001]
    m10 = d[d["競走年月日"] >= 20261001]
    return {"レース数": len(d), "投資": d["投資"].sum(), "払戻": d["払戻"].sum(), "回収率%": roi(d),
            "90%区間": f"{lo:.0f}〜{hi:.0f}", "9月%": roi(m9), "10月%": roi(m10),
            "的中レース%": (d["払戻"] > 0).mean() * 100,
            "３連単 回収率%": d["払戻3"].sum() / max(d["投資3"].sum(), 1) * 100,
            "馬単 回収率%": d["払戻2"].sum() / max(d["投資2"].sum(), 1) * 100,
            "最大の1レース払戻": d["払戻"].max()}


def main():
    od, pay, jkn, names = load()
    rs = races(od, pay, jkn, names)
    print(f"対象 {len(rs)}レース (買い目のあるレース, {min(r['key'][1] for r in rs)}〜{max(r['key'][1] for r in rs)})")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    rows = []
    for m in METHODS:
        for s in SPLITS:
            rows.append({"配分": m, "３連単の割合": s, **summary(simulate(rs, m, s))})
    t = pd.DataFrame(rows)
    t.round(1).to_csv(OUT / "sa_grid.csv", index=False, encoding="utf-8-sig")
    print(t.round(1).to_string(index=False))

    # 採用した配分 (odds_bets.allocate): 1強は馬単に全額, ほかは半分ずつ, 組ごとは 1/確率比例. 均等・半々との差 (対応のあるブートストラップ)
    base = simulate(rs, "均等", 0.5)
    rng = np.random.default_rng(2)
    rows = []
    for name, m, sp in [("均等・半々", "均等", 0.5), ("均等・1強は馬単だけ", "均等", {"1強": 0.0, "2強": 0.5, "3強": 0.5, "4強": 0.5}),
                        ("1/確率比例・半々", "1/確率比例", 0.5),
                        ("1/確率比例・1強は馬単だけ (採用)", "1/確率比例", {"1強": 0.0, "2強": 0.5, "3強": 0.5, "4強": 0.5})]:
        d = simulate(rs, m, sp)
        diff = (d["払戻"] - base["払戻"]).to_numpy()
        bs = diff[rng.integers(0, len(d), (2000, len(d)))].sum(1) / base["投資"].sum() * 100
        rows.append({"配分": name, **summary(d), "均等・半々との差": diff.sum() / base["投資"].sum() * 100,
                     "差の90%区間": f"{np.percentile(bs, 5):+.1f}〜{np.percentile(bs, 95):+.1f}"})
    t0 = pd.DataFrame(rows)
    t0.round(1).to_csv(OUT / "sa_adopted.csv", index=False, encoding="utf-8-sig")
    print("\n== 採用した配分と均等の比較\n", t0.round(1).to_string(index=False))

    # 形ごと (均等・確率比例, 割合0.5)
    rows = []
    for m in ["均等", "払戻均等", "確率比例"]:
        d = simulate(rs, m, 0.5)
        for shape, g in d.groupby("形"):
            rows.append({"配分": m, "形": shape, **summary(g)})
    t2 = pd.DataFrame(rows)
    t2.round(1).to_csv(OUT / "sa_shape.csv", index=False, encoding="utf-8-sig")
    print("\n== 形ごと (３連単の割合0.5)\n", t2.round(1).to_string(index=False))

    # 本命寄り/穴寄り: 確率比例 と 逆 (人気薄の組に厚く) の差を見るための, 買い目ごとの回収率 (オッズ帯別)
    rows = []
    for r in rs:
        for kind, c, o, h, pv in [("３連単", r["tri"], r["t_odds"], r["t_hit"], r["t_pay"]), ("馬単", r["ex"], r["e_odds"], r["e_hit"], r["e_pay"])]:
            for oi, hi in zip(o, h):
                rows.append({"券種": kind, "オッズ": oi, "払戻": pv if hi else 0})
    b = pd.DataFrame(rows)
    b["オッズ帯"] = pd.cut(b["オッズ"], [0, 5, 10, 20, 50, 100, 1e9], labels=["~5倍", "5-10", "10-20", "20-50", "50-100", "100倍~"])
    t3 = b.groupby(["券種", "オッズ帯"], observed=True).agg(点数=("払戻", "size"), 的中=("払戻", lambda s: (s > 0).sum()), 払戻=("払戻", "sum"))
    t3["回収率%"] = t3["払戻"] / (t3["点数"] * 100) * 100
    t3.round(1).to_csv(OUT / "sa_oddsband.csv", encoding="utf-8-sig")
    print("\n== 買い目1点ごとのオッズ帯別回収率 (100円ずつ)\n", t3.round(1).to_string())


if __name__ == "__main__":
    main()
