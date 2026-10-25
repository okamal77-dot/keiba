"""odds_bets.py の買い目 (オッズの形のルール) の答え合わせ.

使い方: python3 analysis/odds_bets_review.py <オッズCSV> <払戻CSV> [--races <racelist.csv>] [--entries <horselist.csv>]
  オッズCSV: odds_bets.py と同じ形式 (確定オッズ・snapshots.csv など). 払戻CSV: payback 形式.
  払戻のあるレースだけを集計する. オッズが締切前の値なら, 当日買ったときの判定と同じになる.
1レース1,000円の配分 (odds_bets.allocate) の成績も出す.
出力: 標準出力とanalysis/out/obr_<日付>.csv
"""
import argparse
from pathlib import Path

import pandas as pd

import odds_bets as OB
from race_profile import race_class

KEY = OB.KEY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("odds")
    ap.add_argument("pay")
    ap.add_argument("--races")
    ap.add_argument("--entries")
    a = ap.parse_args()
    d = OB.read_odds([a.odds])
    if a.entries:
        e = pd.read_csv(a.entries, encoding="utf-8-sig", low_memory=False)[KEY + ["馬番", "騎手名"]]
        d = d.drop(columns=["騎手"], errors="ignore").merge(e, on=KEY + ["馬番"], how="left").rename(columns={"騎手名": "騎手"})
    names = {}
    if a.races:
        r = pd.read_csv(a.races, encoding="utf-8-sig", low_memory=False)
        names = dict(zip(zip(r["競馬場"], r["競走年月日"], r["レース番号"]), r["レース名"]))
    pay = pd.read_csv(a.pay, encoding="utf-8-sig", low_memory=False).drop_duplicates(KEY).set_index(KEY)
    jk = OB.load_jockey()
    rows = []
    for k, g in d.groupby(KEY):
        if k not in pay.index:
            continue
        pr = pay.loc[k]
        g = g.sort_values(["単勝オッズ", "馬番"])
        nos = g["馬番"].astype(int).tolist()
        if len(nos) < 5:
            continue
        fav_ae = OB.jk_ae(jk, g["騎手"].iloc[0], k[0], "1番人気") if "騎手" in g else None
        b = OB.bets(nos, g["単勝オッズ"].tolist(), fav_ae, race_class(names.get(k)) if names.get(k) else None)
        n1, n2, n3, n4, n5 = nos[:5]
        tri = {6: {(n1, x, y) for x in (n2, n3, n4) for y in (n2, n3, n4) if x != y},
               2: {(n1, n2, n3), (n1, n3, n2)}}.get(b["点数3"], set())
        ex = {(n1, x) for x in (n2, n3, n4, n5)} if b["点数2"] else set()
        r3 = tuple(int(pr[c]) for c in ["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"]) \
            if pd.notna(pr["３連単組番馬番1"]) else None
        r2 = (int(pr["馬単組番1"]), int(pr["馬単組番2"])) if pd.notna(pr["馬単組番1"]) else None
        if r3 is None and r2 is None:      # 取り止め・不成立 (返還) のレースは数えない
            continue
        # 1レース1,000円の配分 (第8-8節, odds_bets.allocate)
        al = OB.allocate(nos, g["単勝オッズ"].tolist(), b) if (b["点数3"] or b["点数2"]) else []
        a_inv = sum(y for _, _, y in al)
        a_ret = sum(y / 100 * (pr["３連単払戻金（円）"] if kd == "３連単" else pr["馬単払戻金（円）"])
                    for kd, t, y in al if (kd == "３連単" and t == r3) or (kd == "馬単" and t == r2))
        rows.append({"競馬場": k[0], "R": k[2], "形": b["形"], "クラス": b.get("クラス", ""),
                     "配分投資": a_inv, "配分払戻": a_ret,
                     "結果": "-".join(map(str, r3)) if r3 else "",
                     "結果の人気": "-".join(str(nos.index(x) + 1) if x in nos else "?" for x in r3) if r3 else "",
                     "３連単": b["３連単"], "３連単点数": len(tri),
                     "３連単払戻": pr["３連単払戻金（円）"] if r3 in tri else 0,
                     "馬単": b["馬単"], "馬単点数": len(ex), "馬単払戻": pr["馬単払戻金（円）"] if r2 in ex else 0,
                     "３連単配当": pr["３連単払戻金（円）"], "馬単配当": pr["馬単払戻金（円）"]})
    t = pd.DataFrame(rows).sort_values(["競馬場", "R"])
    pd.set_option("display.width", 250, "display.max_colwidth", 40)
    print(t[["競馬場", "R", "形", "クラス", "結果", "結果の人気", "３連単点数", "３連単払戻", "馬単点数", "馬単払戻", "配分投資", "配分払戻", "３連単配当", "馬単配当"]]
          .to_string(index=False))
    for name in ["３連単", "馬単"]:
        pts, ret = t[f"{name}点数"].sum(), t[f"{name}払戻"].sum()
        print(f"{name}: {(t[f'{name}点数'] > 0).sum()}レースで購入 {pts}点 ({pts * 100:,}円) / 的中 {(t[f'{name}払戻'] > 0).sum()} / "
              f"払戻 {ret:,.0f}円 / 回収率 {ret / max(pts * 100, 1) * 100:.0f}%")
    pts = t["３連単点数"].sum() + t["馬単点数"].sum()
    ret = t["３連単払戻"].sum() + t["馬単払戻"].sum()
    print(f"合計: {pts * 100:,}円 → {ret:,.0f}円 (回収率 {ret / max(pts * 100, 1) * 100:.0f}%)")
    ai, ar = t["配分投資"].sum(), t["配分払戻"].sum()
    print(f"1,000円の配分: {(t['配分投資'] > 0).sum()}レース {ai:,.0f}円 → {ar:,.0f}円 (回収率 {ar / max(ai, 1) * 100:.0f}%)")
    date = int(d["競走年月日"].iloc[0])
    t.to_csv(Path(__file__).parent / "out" / f"obr_{date}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
