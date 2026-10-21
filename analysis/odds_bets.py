"""締切前の単勝オッズから, オッズの形 (オッズの崖の位置) を判定して ３連単・馬単 の買い目を出す.

根拠: REPORT_trifecta_strategy.md 第8章 (2026年9〜10月, 確定オッズの2,198レース. odds_gap_trifecta.py)
  - 3強・4強 (2〜4番人気のオッズが接近): ３連単 1→2,3,4→2,3,4 (6点) 回収率93% / 馬単 1→2〜5 (4点) 90%
  - 1強 (1番人気が抜けている)      : ３連単 1→2,3→2,3 (2点) 79% / 馬単 1→2〜5 (4点) 83%
  - 2強                           : ３連単は見送り (どの買い方も70%前後) / 馬単 1→2〜5 (4点) 81%
  - 1番人気が3倍以上               : ３連単は見送り (68%) / 馬単 1→2〜5 (4点) 90%
  どれも90%区間に100%を含み, 100%を安定して超えるとは言えない.
騎手の補正 (jockey_adjust.py, analysis/out/jk_table.csv): 1番人気の騎手が, その競馬場の1番人気で人気ほど勝たない
  (縮小A/E 0.95未満, 1番人気30回以上) ときは, 馬単 1→2〜5 を見送り扱いにする (9〜10月の検証で回収率77%, ほかは86〜87%).
  ３連単は同じ分け方で逆の結果 (117%) が出たが, 高配当数本によるもので変えない.
クラスの補正 (class_bets.py, 第8-6節): A級は 1強 でも ３連単を6点 (1→2,3,4→2,3,4) にする
  (2021〜2026年の人気順で, A級の2点は6点より13.6ポイント低く, 3期間とも同じ向き). クラスはレース名から判定する.

使い方:
  python3 analysis/odds_bets.py <オッズCSV> [<オッズCSV> ...]
    オッズCSV: 競馬場,競走年月日,レース番号,馬番,単勝オッズ の列を持つもの. 次のどれでもよい
      - analysis/odds_log/snapshots.csv (odds_log.py card で出馬表を貼り付けて記録したもの. 同じレースは最新の記録を使う)
      - data/entries/<日付>_<場>_<R>_odds.csv (odds_log.py card が作るもの)
      - data/odds/2026MM_*_odds.csv (確定オッズ. 賭式=単勝 の行を使う. 答え合わせ用)
  python3 analysis/odds_bets.py --date 20261020 analysis/odds_log/snapshots.csv   (日付で絞る)
  python3 analysis/odds_bets.py --entries data/entries/20261020_horselist.csv ...   (騎手名を出走表から補う)
  python3 analysis/odds_bets.py --jockey-sheet data/entries/20261020_horselist.csv  (オッズ前の騎手メモだけ出す)
  配分: 1レース1,000円 (３連単+馬単) の配分も出す (第8-8節). 1強は馬単に全額, ほかは半分ずつ, 組ごとは当たりにくい組ほど厚く.
  レース名 (クラスの判定用) は, オッズCSVにあればそれを, 無ければ --races のレース一覧 (racelist 形式) を使う.
  --date を付けると data/entries/<日付>_racelist.csv があれば自動で読む.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from race_profile import race_class

KEY = ["競馬場", "競走年月日", "レース番号"]
SHAPES = ["1強", "2強", "3強", "4強"]
JK_PATH = Path(__file__).parent / "out" / "jk_table.csv"
JK_MIN = 30          # この騎乗数未満の騎手×競馬場×人気帯は補正に使わない
FAV_WEAK = 0.95      # 1番人気でこれ未満なら馬単1着固定を見送り扱い
BUDGET = 1000        # 配分の例: 1レースの上限 (３連単+馬単)


def load_jockey():
    if not JK_PATH.exists():
        return None
    t = pd.read_csv(JK_PATH, encoding="utf-8-sig")
    t = t[t["騎乗"] >= JK_MIN]
    return t.set_index(["騎手名", "競馬場", "人気帯"])["縮小A/E"]


def jk_ae(jk, name, course, band):
    if jk is None or not isinstance(name, str):
        return None
    return jk.get((name.replace(" ", "").replace("　", ""), course, band))


def read_odds(paths):
    out = []
    for p in paths:
        x = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        if "賭式" in x.columns:                      # 確定オッズ (全賭式) の形式
            x = x[x["賭式"] == "単勝"].rename(columns={"番号1": "馬番", "オッズ": "単勝オッズ"})
        if "記録時刻" in x.columns:                  # snapshots.csv: レースごとに最新の記録
            last = x.groupby(KEY)["記録時刻"].transform("max")
            x = x[x["記録時刻"] == last]
        out.append(x[[c for c in KEY + ["馬番", "単勝オッズ", "騎手", "レース名"] if c in x.columns]])
    d = pd.concat(out, ignore_index=True)
    d["単勝オッズ"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d["馬番"] = pd.to_numeric(d["馬番"], errors="coerce")
    return d[(d["単勝オッズ"] > 0) & d["馬番"].notna()].drop_duplicates(KEY + ["馬番"], keep="last")


def shape_of(odds):
    """オッズ (昇順) の上位5頭から, 隣り合う人気のオッズ比が最も大きい位置を返す"""
    o = odds[:5]
    ratios = [o[i + 1] / o[i] for i in range(4)]
    return SHAPES[int(np.argmax(ratios))], ratios


def bets(nos, odds, fav_ae=None, cls=None):
    """nos, odds: 人気順 (オッズの低い順). fav_ae: 1番人気の騎手のその場の1番人気での縮小A/E.
    cls: クラス (race_class の値. "A" なら 1強 でも6点). 戻り値: dict"""
    if len(nos) < 5:
        return {"形": "-", "３連単": "見送り (5頭未満)", "馬単": "見送り (5頭未満)", "点数3": 0, "点数2": 0, "注記": ""}
    shape, ratios = shape_of(odds)
    n1, n2, n3, n4, n5 = nos[:5]
    exacta = f"{n1} → {n2},{n3},{n4},{n5} (4点)"
    pts2 = 4
    notes = []
    if fav_ae is not None and fav_ae < FAV_WEAK:
        exacta = f"見送り (1番人気の騎手が1番人気で勝ち切れない: A/E {fav_ae:.2f}) / 買うなら {exacta}"
        pts2 = 0
    if odds[0] >= 3.0:
        tri, pts3 = "見送り (1番人気が3倍以上)", 0
    elif shape in ("3強", "4強"):
        tri, pts3 = f"{n1} → {n2},{n3},{n4} → {n2},{n3},{n4} (6点)", 6
    elif shape == "1強" and cls == "A":
        tri, pts3 = f"{n1} → {n2},{n3},{n4} → {n2},{n3},{n4} (6点)", 6
        notes.append("A級のため1強でも6点 (A級の2点は6点より約14ポイント低い)")
    elif shape == "1強":
        tri, pts3 = f"{n1} → {n2},{n3} → {n2},{n3} (2点)", 2
    else:
        tri, pts3 = "見送り (2強: 安定した買い方がない)", 0
    if odds[3] / odds[1] <= 1.6 and shape in ("3強", "4強") and pts3 == 6:
        notes.append("2〜4番人気がとくに接近 (検証で回収率100%前後の区間)")
    return {"形": shape, "クラス": cls or "", "1〜5番人気のオッズ": " / ".join(f"{o:.1f}" for o in odds[:5]),
            "３連単": tri, "馬単": exacta, "点数3": pts3, "点数2": pts2, "注記": " / ".join(notes)}


def harville(odds_by_no):
    """単勝オッズ {馬番: オッズ} から勝率 (1/オッズを合計1に揃えたもの) を返す"""
    q = {n: 1 / o for n, o in odds_by_no.items()}
    t = sum(q.values())
    return {n: v / t for n, v in q.items()}


def units(w, budget):
    """重み w に比例して budget を100円単位で配る (最大剰余法). 戻り値: 円のリスト"""
    w = np.asarray(w, float)
    n = budget // 100
    if w.sum() <= 0 or n == 0:
        return [0] * len(w)
    x = w / w.sum() * n
    u = np.floor(x)
    for i in np.argsort(-(x - u))[: int(n - u.sum())]:
        u[i] += 1
    return [int(v) * 100 for v in u]


def allocate(nos, odds, b, budget=BUDGET):
    """1レース budget 円 (３連単+馬単の合計) の配分 (第8-8節, stake_alloc.py).
    1強 は馬単だけに全額, ほかは ３連単・馬単 に半分ずつ (片方だけ買うならその券種に全額).
    買い目ごとの金額は Harville 式の確率に反比例 (当たりにくい組ほど厚く. 人気どうしの組は0円になることがある).
    戻り値: [(券種, 組, 金額)]"""
    p = harville(dict(zip(nos, odds)))
    n1, n2, n3, n4, n5 = nos[:5]
    mates = {6: (n2, n3, n4), 2: (n2, n3)}.get(b["点数3"], ())
    tri = [(n1, x, y) for x in mates for y in mates if x != y]
    ex = [(n1, x) for x in (n2, n3, n4, n5)] if b["点数2"] else []
    if b["形"] == "1強" and ex:
        tri = []
    b3 = budget if not ex else (budget // 2 if tri else 0)
    b2 = budget - b3 if ex else 0
    w3 = [1 / (p[a] * p[c] / (1 - p[a]) * p[d] / (1 - p[a] - p[c])) for a, c, d in tri]
    w2 = [1 / (p[a] * p[c] / (1 - p[a])) for a, c in ex]
    return [("３連単", t, y) for t, y in zip(tri, units(w3, b3))] + [("馬単", t, y) for t, y in zip(ex, units(w2, b2))]


def fmt_alloc(al):
    out = []
    for kind in ["３連単", "馬単"]:
        xs = [f"{'-'.join(map(str, t))} {y}円" for k, t, y in al if k == kind and y > 0]
        if xs:
            out.append(f"{kind}: " + " / ".join(xs))
    return "  ".join(out)


def jockey_sheet(entries, jk):
    """オッズが出る前の騎手メモ: 1番人気になったときの強弱 (縮小A/E 1.10以上 / 0.90以下) と, 人気薄で人気以上に勝つ騎手 (1.15以上)"""
    h = pd.read_csv(entries, encoding="utf-8-sig", low_memory=False)
    for k, g in h.groupby(KEY):
        lines = []
        for _, r in g.sort_values("馬番").iterrows():
            tags = []
            # 再現性を確かめた向きだけ出す (jockey_adjust.py の (1)): 1番人気での強弱, 人気薄で人気以上に勝つ騎手
            v = jk_ae(jk, r["騎手名"], k[0], "1番人気")
            if v is not None and (v >= 1.10 or v <= 0.90):
                tags.append(f"1番人気なら{'◎' if v >= 1.10 else '×'}{v:.2f}")
            for band in ["4〜6番人気", "7番人気以下"]:
                v = jk_ae(jk, r["騎手名"], k[0], band)
                if v is not None and v >= 1.15:
                    tags.append(f"{band}なら◎{v:.2f}")
            if tags:
                lines.append(f"    {int(r['馬番'])}番 {r['騎手名']}: " + " ".join(tags))
        print(f"{k[0]} {k[2]}R")
        print("\n".join(lines) if lines else "    (目立つ騎手なし)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--date", type=int)
    ap.add_argument("--entries", help="騎手名を補う出走表 (horselist 形式)")
    ap.add_argument("--races", nargs="*", default=[], help="レース名を補うレース一覧 (racelist 形式, クラスの判定用)")
    ap.add_argument("--budget", type=int, default=BUDGET, help="1レースの上限 (３連単+馬単, 円. 既定1000)")
    ap.add_argument("--jockey-sheet", action="store_true", help="files を出走表とみなし, 騎手メモだけ出す")
    a = ap.parse_args()
    jk = load_jockey()
    if a.jockey_sheet:
        print("騎手メモ (◎: その人気帯で人気以上に勝つ, ×: 人気ほど勝たない. 縮小A/E = 実際の勝ち数÷人気から見込む勝ち数, 2021年〜, その場で30騎乗以上)")
        for f in a.files:
            jockey_sheet(f, jk)
        return
    d = read_odds(a.files)
    if a.entries:
        e = pd.read_csv(a.entries, encoding="utf-8-sig", low_memory=False)[KEY + ["馬番", "騎手名"]]
        d = d.drop(columns=["騎手"], errors="ignore").merge(e, on=KEY + ["馬番"], how="left").rename(columns={"騎手名": "騎手"})
    if a.date:
        d = d[d["競走年月日"] == a.date]
        auto = Path(__file__).parent.parent / "data" / "entries" / f"{a.date}_racelist.csv"
        if auto.exists() and str(auto) not in a.races:
            a.races.append(str(auto))
    if a.races:
        rl = pd.concat([pd.read_csv(f, encoding="utf-8-sig", low_memory=False)[KEY + ["レース名"]] for f in a.races])
        rl = rl.drop_duplicates(KEY).rename(columns={"レース名": "レース名_一覧"})
        d = d.merge(rl, on=KEY, how="left")
        d["レース名"] = d["レース名"].fillna(d["レース名_一覧"]) if "レース名" in d else d["レース名_一覧"]
    rows = []
    for k, g in d.groupby(KEY):
        g = g.sort_values(["単勝オッズ", "馬番"])
        fav_ae = jk_ae(jk, g["騎手"].iloc[0], k[0], "1番人気") if "騎手" in g else None
        name = g["レース名"].dropna().iloc[0] if "レース名" in g and g["レース名"].notna().any() else None
        b = bets(g["馬番"].astype(int).tolist(), g["単勝オッズ"].tolist(), fav_ae, race_class(name) if name else None)
        if b["点数3"] or b["点数2"]:
            b["配分"] = fmt_alloc(allocate(g["馬番"].astype(int).tolist(), g["単勝オッズ"].tolist(), b, a.budget))
        rows.append({"競馬場": k[0], "日付": k[1], "R": k[2], **b})
    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        print(f"{r['競馬場']} {r['R']}R  形:{r['形']}  クラス:{r.get('クラス') or '不明'}  オッズ(1〜5番人気): {r.get('1〜5番人気のオッズ', '')}")
        print(f"  ３連単: {r['３連単']}")
        print(f"  馬単  : {r['馬単']}" + (f"\n  注記  : {r['注記']}" if r["注記"] else ""))
        if isinstance(r.get("配分"), str):
            print(f"  配分  : {r['配分']}")
    out_path = Path(__file__).parent / "out" / f"odds_bets_{a.date or 'all'}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n{len(out)}レース, ３連単 {int(out['点数3'].sum())}点・馬単 {int(out['点数2'].sum())}点 → {out_path}")


if __name__ == "__main__":
    main()
