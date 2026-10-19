"""荒れ判定 (race_profile.py) の検証.

使い方: python3 analysis/race_profile_backtest.py <data>
  (0) 2021年・2022年 (集計表より前の年なので, 学習に使っていない期間): 判定と実際の比較, 判定別の回収率.
  (1) 2026年1〜8月: 推定万馬券率・上位4人気決着率と実際の比較, 判定別の買い方の回収率.
      集計表 (analysis/out/*.csv) はこの期間を含む 2023年〜2026年8月で作っているので, 学習期間内の確認になる.
  (2) 2026年9月 (学習に使っていない期間): 確定単勝オッズを入れて判定し, 買い目の回収率を
      「同じ点数を３連単の人気順 (確定オッズの低い順) に買った場合」と比べる. あわせて
      predict_v2.py に足した「荒判定の3着押さえ」を, 人気順の2頭軸に足した場合で検証する.
  ※ 確定オッズは締切後の値なので, 実際に買える時点のオッズより少し有利な検証になる.
出力: analysis/out/rpb_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import manbaken_analysis as MA  # noqa: E402
import race_profile as RP  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(__file__).parent / "out"
KEY = ["競馬場", "競走年月日", "レース番号"]
PERIODS = {"2021年": ("2021_horselist_全月まとめ.csv", "race2021.csv", "pay2021.csv"),
           "2022年": ("2022_horselist_全月まとめ.csv", "race2022.csv", "pay2022.csv"),
           "1-8月": ("2026_horselist_1-8月まとめ.csv", "race2026.csv", "pay2026.csv"),
           "9月": ("202609_horselist.csv", "202609_racelist.csv", "202609_payback.csv")}


def read(name, **kw):
    return pd.read_csv(DATA / name, encoding="utf-8-sig", low_memory=False, **kw)


def load():
    hs, rs, ps = [], [], []
    for per, (hf, rf, pf) in PERIODS.items():
        if not all((DATA / f).exists() for f in (hf, rf, pf)):
            print(f"※ {per} のデータが無いので飛ばす", file=sys.stderr)
            continue
        h, r, p = read(hf), read(rf).drop_duplicates(KEY), read(pf).drop_duplicates(KEY)
        for x in (h, r, p):
            x["期間"] = per
        hs.append(h), rs.append(r), ps.append(p)
    h, r, p = pd.concat(hs), pd.concat(rs), pd.concat(ps)
    for c in ["着順", "人気", "馬番"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h[h["人気"].notna() & h["馬番"].notna()]
    # 前走の脚質 (コーナー通過順から. 事前に分かる情報として使う)
    st = []
    for _, race in r.iterrows():
        for b, (s, _) in MA.running_style(race).items():
            st.append((race["競馬場"], race["競走年月日"], race["レース番号"], b, s))
    st = pd.DataFrame(st, columns=KEY + ["馬番", "脚質"])
    h = h.merge(st, on=KEY + ["馬番"], how="left")
    h["馬ID"] = h["馬名"].astype(str) + "_" + h["生年月日"].astype(str)
    h = h.sort_values(["馬ID", "競走年月日", "レース番号"])
    h["前走脚質"] = h.groupby("馬ID")["脚質"].shift(1)
    return h, r, p


def load_odds():
    files = sorted((DATA / "odds").glob("202609_*_odds.csv"))
    o = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False,
                               usecols=KEY + ["賭式", "番号1", "番号2", "番号3", "オッズ"]) for f in files])
    o["オッズ"] = pd.to_numeric(o["オッズ"], errors="coerce")
    win = o[o["賭式"] == "単勝"].rename(columns={"番号1": "馬番", "オッズ": "単勝オッズ"})[KEY + ["馬番", "単勝オッズ"]]
    tri = o[(o["賭式"] == "３連単") & (o["オッズ"] > 0)].copy()
    tri["組"] = list(zip(tri["番号1"].astype(int), tri["番号2"].astype(int), tri["番号3"].astype(int)))
    tri = {k: dict(zip(g["組"], g["オッズ"])) for k, g in tri.groupby(KEY)}
    return win, tri


def by_pop(x, A, B, C):
    """人気の組 (A→B→C) を馬番の３連単の組に"""
    m = dict(zip(x["人気"].astype(int), x["馬番"].astype(int)))
    return [(m[a], m[b], m[c]) for a in A for b in B for c in C
            if len({a, b, c}) == 3 and a in m and b in m and c in m]


def main():
    h, r, p = load()
    win, tri_odds = load_odds()
    h = h.merge(win, on=KEY + ["馬番"], how="left")
    t = RP.load()
    rows = []
    rinfo = r.set_index(KEY)
    pinfo = p.set_index(KEY)
    for k, x in h.groupby(KEY, sort=False):
        if k not in rinfo.index or len(x) < 5 or x["着順"].notna().sum() < 3:
            continue
        course, _, rno = k
        if course not in t["summary"].index:
            continue
        if k not in pinfo.index:
            continue
        pr = pinfo.loc[k]
        tri_cols = ["３連単払戻金（円）", "３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"]
        if pr[tri_cols].isna().any():   # ３連単が不成立などで組番が無いレース
            continue
        res = (int(pr["３連単組番馬番1"]), int(pr["３連単組番馬番2"]), int(pr["３連単組番馬番3"]))
        payout = float(pr["３連単払戻金（円）"])
        ri = rinfo.loc[k]
        per = ri["期間"]
        x = x.sort_values("人気")
        odds_ok = per == "9月" and x["単勝オッズ"].notna().all() and (x["単勝オッズ"] > 0).all()
        fav_odds = float(x["単勝オッズ"].iloc[0]) if odds_ok else None
        rr = {"競馬場": course, "R": int(rno), "距離": pd.to_numeric(ri["距離"], errors="coerce"), "馬場": ri["馬場"],
              "頭数": len(x), "クラス": ri["レース名"], "種別": ri["競走種類名称"], "1人気オッズ": fav_odds}
        if pd.isna(rr["距離"]):
            rr["距離"] = None
        runners = [{"人気": int(a.人気), "馬番": int(a.馬番), "単勝オッズ": float(a.単勝オッズ) if odds_ok else None,
                    "脚質": a.前走脚質 if isinstance(a.前走脚質, str) else "", "騎手": a.騎手名, "移籍": ""}
                   for a in x.itertuples()]
        out = RP.predict_race(t, rr, runners)
        pops = x.set_index("馬番")["人気"]
        row = {"期間": per, "競馬場": course, "R": int(rno), "頭数": len(x), "判定": out["verdict"],
               "推定万馬券率": out["pm"], "推定上位4人気": out["t4"], "1人気オッズ": fav_odds,
               "万馬券": payout >= 10000, "上位4人気決着": max(pops.get(b, 99) for b in res) <= 4,
               "３連単配当": payout, "モデル的中確率": out["probs"].get(res, 0.0),
               "点数": len(out["tickets"]), "払戻": payout if res in out["tickets"] else 0}
        # 比較用の固定の買い方 (人気順)
        for name, combos in [("1人気1着固定9点", by_pop(x, [1], [2, 3, 4], [2, 3, 4, 5])),
                             ("2頭軸18点", by_pop(x, [1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5])),
                             ("上位4人気BOX24点", by_pop(x, [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]))]:
            row[name + "_点数"] = len(combos)
            row[name + "_払戻"] = payout if res in combos else 0
        # 荒判定の3着押さえ (predict_v2 に足したもの) を人気順の2頭軸に足した場合. 比較: 6・7番人気を足す
        if out["verdict"] == "荒":
            n = len(x)
            order = list(range(n))  # 人気順 = 印の順の代わり
            nos = x["馬番"].astype(int).tolist()
            o = x["単勝オッズ"].astype(float).tolist() if odds_ok else RP.default_odds(n)
            strength = (1 / np.array(o)) ** 0.8
            ana = RP.ana_addon(course, "荒", order, nos, x["人気"].astype(int).tolist(), [False] * n, strength,
                               x["騎手名"].tolist(), x["前走脚質"].fillna("").tolist())
            for name, picks in [("押さえ", [nos[i] for i, _ in ana]),
                                ("押さえ比較_6-7人気", [nos[i] for i in range(n) if x["人気"].iloc[i] in (6, 7)])]:
                m = dict(zip(x["人気"].astype(int), nos))
                combos = [(m[a], m[b], c) for a in [1, 2] for b in [1, 2, 3, 4] for c in picks
                          if a in m and b in m and len({m[a], m[b], c}) == 3]
                row[name + "_点数"] = len(combos)
                row[name + "_払戻"] = payout if res in combos else 0
        # 9月: 同じ点数を３連単の人気順 (確定オッズの低い順) に買った場合
        if k in tri_odds:
            mk = sorted(tri_odds[k].items(), key=lambda kv: kv[1])[:row["点数"]]
            row["市場順_払戻"] = payout if res in {c for c, _ in mk} else 0
            ev = [(c, out["probs"].get(c, 0) * od) for c, od in tri_odds[k].items()]
            row["_ev"] = ev
            row["_res"] = res
        rows.append(row)
    d = pd.DataFrame(rows)
    ev_rows = []
    for _, rw in d[d["期間"] == "9月"].iterrows():
        if isinstance(rw.get("_ev"), list):
            for c, e in rw["_ev"]:
                ev_rows.append((e, rw["３連単配当"] if c == rw["_res"] else 0))
    d = d.drop(columns=[c for c in ["_ev", "_res"] if c in d])
    d.to_csv(OUT / "rpb_races.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 40)

    # ---- (1) 判定ごとの実績
    def summ(g):
        s = {"レース数": len(g), "推定万馬券率%": g["推定万馬券率"].mean() * 100, "実際の万馬券率%": g["万馬券"].mean() * 100,
             "推定上位4人気%": g["推定上位4人気"].mean() * 100, "実際の上位4人気%": g["上位4人気決着"].mean() * 100,
             "モデル買い目 的中%": (g["払戻"] > 0).mean() * 100,
             "モデル買い目 回収率%": g["払戻"].sum() / max(g["点数"].sum() * 100, 1) * 100}
        for name in ["1人気1着固定9点", "2頭軸18点", "上位4人気BOX24点"]:
            s[name + " 回収率%"] = g[name + "_払戻"].sum() / max(g[name + "_点数"].sum() * 100, 1) * 100
        if "市場順_払戻" in g and g["市場順_払戻"].notna().any():
            gg = g[g["市場順_払戻"].notna()]
            s["同点数を市場人気順 的中%"] = (gg["市場順_払戻"] > 0).mean() * 100
            s["同点数を市場人気順 回収率%"] = gg["市場順_払戻"].sum() / max(gg["点数"].sum() * 100, 1) * 100
        return pd.Series(s)

    t1 = d.groupby(["期間", "判定"]).apply(summ)
    print("== 判定ごとの実績\n", t1.round(1).to_string())
    t1.round(2).to_csv(OUT / "rpb_by_verdict.csv", encoding="utf-8-sig")

    # 推定万馬券率の帯ごとの実際 (較正)
    d["推定帯"] = pd.cut(d["推定万馬券率"], [0, .3, .4, .5, .6, .7, 1])
    t2 = d.groupby(["期間", "推定帯"], observed=True).agg(レース数=("万馬券", "size"), 推定=("推定万馬券率", "mean"),
                                                       実際=("万馬券", "mean"))
    t2[["推定", "実際"]] *= 100
    print("\n== 推定万馬券率の較正\n", t2.round(1).to_string())
    t2.round(2).to_csv(OUT / "rpb_calibration.csv", encoding="utf-8-sig")

    # 判定の当てはまり: 荒判定と堅判定で, 実際の万馬券率に差があるか (競馬場別)
    t3 = d.pivot_table(index="競馬場", columns="判定", values="万馬券", aggfunc="mean") * 100
    t3["レース数"] = d.groupby("競馬場").size()
    print("\n== 競馬場×判定 の実際の万馬券率(%)\n", t3.round(1).to_string())
    t3.round(2).to_csv(OUT / "rpb_track_verdict.csv", encoding="utf-8-sig")

    # ---- (2) 荒判定の3着押さえ
    a = d[d["判定"] == "荒"]
    t4 = pd.DataFrame({name: {"レース数": int((a[name + "_点数"] > 0).sum()), "点数": int(a[name + "_点数"].sum()),
                              "的中": int((a[name + "_払戻"] > 0).sum()),
                              "回収率%": a[name + "_払戻"].sum() / max(a[name + "_点数"].sum() * 100, 1) * 100}
                       for name in ["押さえ", "押さえ比較_6-7人気", "2頭軸18点"]}).T
    for per in PERIODS:
        g = a[a["期間"] == per]
        for name in ["押さえ", "押さえ比較_6-7人気", "2頭軸18点"]:
            t4.loc[name, f"回収率%_{per}"] = g[name + "_払戻"].sum() / max(g[name + "_点数"].sum() * 100, 1) * 100
    print("\n== 荒判定の3着押さえ (人気順の2頭軸 1,2→1〜4→穴 に足した分)\n", t4.round(1).to_string())
    t4.round(2).to_csv(OUT / "rpb_addon.csv", encoding="utf-8-sig")

    # ---- (3) 9月: モデルの確率 × 確定３連単オッズ の帯ごとの回収率
    if ev_rows:
        e = pd.DataFrame(ev_rows, columns=["期待値", "払戻"])
        e["帯"] = pd.cut(e["期待値"], [0, .5, .8, 1.0, 1.2, 1.5, 2, 3, 1e9])
        t5 = e.groupby("帯", observed=True).agg(組数=("払戻", "size"), 的中=("払戻", lambda s: (s > 0).sum()),
                                               払戻=("払戻", "sum"))
        t5["回収率%"] = t5["払戻"] / (t5["組数"] * 100) * 100
        print("\n== 9月: モデルの確率×確定３連単オッズ (期待値) の帯ごとの回収率\n", t5.round(1).to_string())
        t5.round(2).to_csv(OUT / "rpb_ev.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
