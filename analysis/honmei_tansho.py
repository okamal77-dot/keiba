"""能力モデルの◎の単勝を, 締切時の単勝オッズで買った場合の成績と, オッズの形のルール (1,000円の配分) との組み合わせを確かめる.

使い方: python3 analysis/honmei_tansho.py <data>
◎と勝率: analysis/out/pred2_underrated_rows_*.csv (predict_v2.py の検証区間. 学習に使っていない直近3か月の予想) の
  勝率順1位の馬. 勝率は ev_calibrate と同じ区分 (モデル勝率の帯 × 想定人気の帯) で実際の勝率に寄せた「補正勝率」を使う.
  補正は偶数日・奇数日の2分割で, 片方で求めたものをもう片方に当てる (先読みなし).
  加えて, 11/5 は当日の予想 (pred2_20261105_*.csv の 補正勝率%) をそのまま使う.
オッズ: data/odds の締切時のオッズ (2026年9月〜. 同じレースは *_final_odds.csv を優先).
期待値 = 補正勝率 × 締切時の単勝オッズ.
組み合わせ (1レースの上限は ルール1,000円 + 単勝):
  A ルールだけ             : odds_bets.allocate の1,000円
  B ルール + ◎単勝100〜300円: ◎の期待値が 1.2〜2.0 のときだけ ◎の単勝を足す
  C 置き換え               : ◎の期待値が 1.2〜2.0 で ◎が1番人気でないときは, ルールを買わずに ◎の単勝1,000円
  D ◎単勝だけ (期待値1.2〜2.0) 1,000円
出力: analysis/out/ht_*.csv
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ev_calibrate as EC  # noqa: E402
import odds_bets as OB  # noqa: E402
from race_profile import race_class  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = OB.KEY
EV_LO, EV_HI = 1.2, 2.0


def load_odds():
    fs = sorted((DATA / "odds").glob("2026*_odds.csv"), key=lambda f: "final" in f.name)
    od = []
    for f in fs:
        x = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        if x.empty:
            continue
        x = x[x["賭式"] == "単勝"]
        od.append(x[KEY + ["番号1", "オッズ"]])
    od = pd.concat(od).drop_duplicates(KEY + ["番号1"], keep="last")
    od = od.rename(columns={"番号1": "馬番", "オッズ": "単勝オッズ"})
    od["単勝オッズ"] = pd.to_numeric(od["単勝オッズ"], errors="coerce")
    od = od[od["単勝オッズ"] > 0]
    od["馬番"] = od["馬番"].astype(int)
    od["締切人気"] = od.groupby(KEY)["単勝オッズ"].rank(method="first")
    return od


def load_pay():
    fs = [DATA / f"2026{m:02d}_payback.csv" for m in (9, 10, 11)]
    p = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in fs if f.exists()])
    return p.drop_duplicates(KEY, keep="last").set_index(KEY)


def load_side():
    fs = [DATA / f"2026{m:02d}_horselist.csv" for m in (9, 10, 11)]
    h = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["馬番", "騎手名"]) for f in fs if f.exists()])
    jkn = h.drop_duplicates(KEY + ["馬番"], keep="last").set_index(KEY + ["馬番"])["騎手名"]
    fs = [DATA / f"2026{m:02d}_racelist.csv" for m in (9, 10, 11)]
    r = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False, usecols=KEY + ["レース名"]) for f in fs if f.exists()])
    return jkn, r.drop_duplicates(KEY, keep="last").set_index(KEY)["レース名"]


def honmei():
    """レースごとの ◎ (馬番, 補正勝率)"""
    m = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob(str(OUT / "pred2_underrated_rows_*.csv"))])
    m = m.drop_duplicates(KEY + ["馬番"], keep="last").reset_index(drop=True)
    m["補正勝率"] = np.nan
    even = (m["競走年月日"] % 2 == 0).values
    for tr, te in [(even, ~even), (~even, even)]:
        pb, qb = EC.bands(m[tr])
        g = m[tr].groupby([pb, qb], observed=True)
        win = (g["着順"].apply(lambda s: (s == 1).sum()) + EC.SHRINK * g["勝率"].mean()) / (g.size() + EC.SHRINK)
        pcal = win / g["勝率"].mean()
        pb, qb = EC.bands(m[te])
        r = pcal.reindex(pd.MultiIndex.from_arrays([pb, qb])).fillna(1.0).values
        m.loc[te, "補正勝率"] = m.loc[te, "勝率"].values * r
    top = m[m["勝率順"] == 1][KEY + ["馬番", "補正勝率"]]
    live = []
    for f in glob.glob(str(OUT / "pred2_20261105_*.csv")):
        if f.endswith("_valid.csv"):
            continue
        x = pd.read_csv(f, encoding="utf-8-sig")
        x = x[x["印"] == "◎"]
        live.append(pd.DataFrame({"競馬場": Path(f).stem.split("_")[2], "競走年月日": 20261105, "レース番号": x["レース"],
                                  "馬番": x["馬番"], "補正勝率": x["補正勝率%"] / 100}))
    top = pd.concat([top] + live).drop_duplicates(KEY, keep="last")
    return top.rename(columns={"馬番": "◎"})


def rule_bets(od, pay, jkn, names):
    jk = OB.load_jockey()
    rows = []
    for k, g in od.groupby(KEY):
        if k not in pay.index:
            continue
        g = g.sort_values(["単勝オッズ", "馬番"])
        nos, oo = g["馬番"].tolist(), g["単勝オッズ"].tolist()
        if len(nos) < 5:
            continue
        pr = pay.loc[k]
        r3 = tuple(int(pr[c]) for c in ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"]) if pd.notna(pr["３連単組番馬番1"]) else None
        r2 = (int(pr["馬単組番1"]), int(pr["馬単組番2"])) if pd.notna(pr["馬単組番1"]) else None
        if r3 is None and r2 is None:
            continue
        fav_ae = OB.jk_ae(jk, jkn.get(k + (nos[0],)), k[0], "1番人気")
        b = OB.bets(nos, oo, fav_ae, race_class(names.get(k)) if k in names.index else None)
        al = OB.allocate(nos, oo, b) if (b["点数3"] or b["点数2"]) else []
        inv = sum(y for _, _, y in al)
        ret = sum(y / 100 * (pr["３連単払戻金（円）"] if kd == "３連単" else pr["馬単払戻金（円）"])
                  for kd, t, y in al if (kd == "３連単" and t == r3) or (kd == "馬単" and t == r2))
        rows.append({"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "形": b["形"], "ルール投資": inv, "ルール払戻": ret,
                     "勝ち馬": int(pr["単勝組番"]) if pd.notna(pr["単勝組番"]) else -1,
                     "単勝配当": pr["単勝払戻金（円）"]})
    return pd.DataFrame(rows)


def roi(inv, ret):
    return ret.sum() / max(inv.sum(), 1) * 100


def boot(inv, ret, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    inv, ret = np.asarray(inv, float), np.asarray(ret, float)
    idx = rng.integers(0, len(inv), (n, len(inv)))
    r = ret[idx].sum(1) / np.maximum(inv[idx].sum(1), 1) * 100
    return f"{np.percentile(r, 5):.0f}〜{np.percentile(r, 95):.0f}"


def main():
    od = load_odds()
    pay = load_pay()
    jkn, names = load_side()
    hm = honmei()
    d = rule_bets(od, pay, jkn, names).merge(hm, on=KEY, how="inner")
    d = d.merge(od.rename(columns={"馬番": "◎"}), on=KEY + ["◎"], how="left").dropna(subset=["単勝オッズ"])
    d["期待値"] = d["補正勝率"] * d["単勝オッズ"]
    d["◎勝"] = d["勝ち馬"] == d["◎"]
    d["◎払戻100"] = np.where(d["◎勝"], d["単勝配当"], 0)
    d["期"] = np.select([d["競走年月日"] < 20261001, d["競走年月日"] < 20261101], ["9月", "10月"], "11/5")
    d.to_csv(OUT / "ht_races.csv", index=False, encoding="utf-8-sig")
    print(f"対象 {len(d)}レース ({d['競走年月日'].min()}〜{d['競走年月日'].max()}, {d['競馬場'].nunique()}場)")
    pd.set_option("display.width", 250, "display.max_columns", 30)

    # (1) ◎の単勝 100円ずつ: 期待値の帯・◎が1番人気かどうか
    d["期待値帯"] = pd.cut(d["期待値"], [0, 0.8, 1.0, 1.2, 2.0, 99], labels=["0.8未満", "0.8-1.0", "1.0-1.2", "1.2-2.0", "2.0超"], right=False)
    d["◎の人気"] = np.where(d["締切人気"] == 1, "1番人気", np.where(d["締切人気"] <= 3, "2-3番人気", "4番人気以下"))
    rows = []
    for name, col in [("全体", None), ("期待値帯", "期待値帯"), ("◎の人気", "◎の人気"), ("期", "期")]:
        groups = [("全体", d)] if col is None else list(d.groupby(col, observed=True))
        for gk, g in groups:
            rows.append({"区分": name, "値": gk, "レース数": len(g), "◎勝率%": g["◎勝"].mean() * 100,
                         "平均オッズ": g["単勝オッズ"].mean(), "単勝回収率%": roi(g["◎勝"] * 0 + 100, g["◎払戻100"]),
                         "90%区間": boot(np.full(len(g), 100), g["◎払戻100"])})
    t1 = pd.DataFrame(rows)
    print("\n== (1) ◎の単勝 (締切オッズ, 100円ずつ)\n", t1.round(1).to_string(index=False))
    t1.round(2).to_csv(OUT / "ht_honmei.csv", index=False, encoding="utf-8-sig")
    band = d[(d["期待値"] >= EV_LO) & (d["期待値"] < EV_HI)]
    x = band.groupby(["期", "◎の人気"]).apply(lambda g: pd.Series({"レース数": len(g), "単勝回収率%": roi(np.full(len(g), 100), g["◎払戻100"])}))
    print("\n== 期待値1.2〜2.0 の◎: 期 × ◎の人気\n", x.round(0).to_string())

    # (2) 組み合わせ
    ok = (d["期待値"] >= EV_LO) & (d["期待値"] < EV_HI)
    ok_nf = ok & (d["締切人気"] > 1)
    plans = {}
    plans["A ルールだけ (1,000円)"] = (d["ルール投資"], d["ルール払戻"])
    for s in (100, 200, 300):
        plans[f"B ルール + ◎単勝{s}円 (期待値1.2〜2.0)"] = (d["ルール投資"] + ok * s, d["ルール払戻"] + ok * d["◎払戻100"] * s / 100)
    plans["C 置き換え (期待値1.2〜2.0 で◎が1番人気でない → ◎単勝1,000円)"] = (
        np.where(ok_nf, 1000, d["ルール投資"]), np.where(ok_nf, d["◎払戻100"] * 10, d["ルール払戻"]))
    plans["D ◎単勝だけ (期待値1.2〜2.0, 1,000円)"] = (ok * 1000, ok * d["◎払戻100"] * 10)
    plans["E ◎単勝だけ (全レース, 1,000円)"] = (np.full(len(d), 1000), d["◎払戻100"] * 10)
    rows = []
    for name, (inv, ret) in plans.items():
        inv, ret = pd.Series(np.asarray(inv, float)), pd.Series(np.asarray(ret, float))
        r = {"買い方": name, "買ったレース": int((inv > 0).sum()), "投資": inv.sum(), "払戻": ret.sum(),
             "回収率%": roi(inv, ret), "90%区間": boot(inv, ret)}
        for per in ["9月", "10月", "11/5"]:
            mk = (d["期"] == per).values
            r[per + "%"] = roi(inv[mk], ret[mk])
        rows.append(r)
    t2 = pd.DataFrame(rows)
    print("\n== (2) ルールとの組み合わせ\n", t2.round(1).to_string(index=False))
    t2.round(2).to_csv(OUT / "ht_plans.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
