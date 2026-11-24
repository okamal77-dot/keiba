"""今の予想 (オッズの形のルール + 配分) を作っている「予想家」(部品) ごとの貢献度を測る.

使い方: python3 analysis/contrib.py   (data/ に 202609〜11 の payback・horselist・racelist と data/odds の確定オッズが要る)
予想家 (部品):
  市場     : 確定オッズの人気順 (1番人気を1着に固定し, 2〜4番人気・2〜5番人気を相手にする). これが土台.
  形       : 2強は３連単を見送り, 1強は３連単を2点に絞る.
  3倍      : 1番人気が3倍以上なら３連単を見送り.
  騎手     : 1番人気の騎手がその場の1番人気で勝ち切れない (縮小A/E 0.95未満) なら馬単を見送り.
  クラス   : A級は1強でも３連単6点.
  配分     : 1,000円を, 1強は馬単に全額, ほかは半々, 組ごとは当たりにくい組ほど厚く (1/確率比例).
  能力モデル: 印は買い目に入っていない. ◎が1番人気と同じか違うかでルールの成績が変わるかを見る (ht_races.csv).
貢献度 = 全部そろったルールの回収率 − その部品だけを外したルールの回収率 (各100円の買い方で比べる).
差の90%区間はレース単位のブートストラップ.
出力: analysis/out/contrib_summary.csv, contrib_races.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], str(Path(__file__).parent.parent / "data")]
import odds_bets as OB  # noqa: E402
import stake_alloc as SA  # noqa: E402
from race_profile import race_class  # noqa: E402

KEY = OB.KEY
OUT = Path(__file__).parent / "out"
PARTS = ["形", "3倍", "騎手", "クラス"]


def plan(nos, odds, fav_ae, cls, use):
    """use: 使う部品の集合. 戻り値: (３連単の組, 馬単の組)"""
    shape, _ = OB.shape_of(odds)
    n1, n2, n3, n4, n5 = nos[:5]
    six = [(n1, x, y) for x in (n2, n3, n4) for y in (n2, n3, n4) if x != y]
    two = [(n1, n2, n3), (n1, n3, n2)]
    ex = [(n1, x) for x in (n2, n3, n4, n5)]
    tri = six
    if "形" in use:
        if shape == "2強":
            tri = []
        elif shape == "1強":
            tri = six if ("クラス" in use and cls == "A") else two
    if "3倍" in use and odds[0] >= 3.0:
        tri = []
    if "騎手" in use and fav_ae is not None and fav_ae < OB.FAV_WEAK:
        ex = []
    return tri, ex, shape


def main():
    od, pay, jkn, names = SA.load()
    jk = OB.load_jockey()
    tri_o = od[od["賭式"] == "３連単"].set_index(KEY + ["番号1", "番号2", "番号3"])["オッズ"]
    ex_o = od[od["賭式"] == "馬単"].set_index(KEY + ["番号1", "番号2"])["オッズ"]
    win = od[od["賭式"] == "単勝"]
    hon = pd.read_csv(OUT / "ht_races.csv", encoding="utf-8-sig").set_index(KEY)["◎"]
    variants = {"全部": set(PARTS), "市場だけ": set()}
    variants.update({f"−{p}": set(PARTS) - {p} for p in PARTS})
    rows = []
    for k, g in win.groupby(KEY):
        if k not in pay.index or k[0] == "帯広ば":
            continue
        g = g.assign(o=pd.to_numeric(g["オッズ"], errors="coerce"))
        g = g[g["o"] > 0].sort_values(["o", "番号1"])
        if len(g) < 5:
            continue
        nos, oo = g["番号1"].astype(int).tolist(), g["o"].tolist()
        pr = pay.loc[k]
        if pd.isna(pr["３連単組番馬番1"]) or pd.isna(pr["馬単組番1"]):
            continue
        r3 = tuple(int(pr[f"３連単組番馬番{i}"]) for i in (1, 2, 3))
        r2 = (int(pr["馬単組番1"]), int(pr["馬単組番2"]))
        fav_ae = OB.jk_ae(jk, jkn.get(k + (nos[0],)), k[0], "1番人気")
        cls = race_class(names.get(k)) if k in names.index else None
        row = {"競馬場": k[0], "競走年月日": k[1], "レース番号": k[2], "1番人気": nos[0], "1番人気オッズ": oo[0],
               "◎": hon.get(k, np.nan)}
        for vn, use in variants.items():
            tri, ex, shape = plan(nos, oo, fav_ae, cls, use)
            row["形"] = shape
            row[vn + "_投"] = 100 * (len(tri) + len(ex))
            row[vn + "_払"] = (pr["３連単払戻金（円）"] if r3 in tri else 0) + (pr["馬単払戻金（円）"] if r2 in ex else 0)
            if vn == "全部":
                # 配分 (今のやり方). オッズが欠けた組は配分の計算から外す
                t_odds = np.array([tri_o.get(k + c, np.nan) for c in tri], float)
                e_odds = np.array([ex_o.get(k + c, np.nan) for c in ex], float)
                if (tri or ex) and not (np.isnan(t_odds).any() or np.isnan(e_odds).any()):
                    q = 1 / np.array(oo)
                    p = dict(zip(nos, q / q.sum()))
                    b3 = 0 if not tri else (1000 if not ex else (0 if shape == "1強" else 500))
                    b2 = 1000 - b3 if ex else 0
                    s3 = SA.alloc("1/確率比例", t_odds, np.array([SA.harville3(p, *c) for c in tri]), b3) if tri else []
                    s2 = SA.alloc("1/確率比例", e_odds, np.array([SA.harville2(p, *c) for c in ex]), b2) if ex else []
                    row["配分_投"] = float(np.sum(s3) + np.sum(s2))
                    row["配分_払"] = sum(s / 100 * pr["３連単払戻金（円）"] for s, c in zip(s3, tri) if c == r3) + \
                        sum(s / 100 * pr["馬単払戻金（円）"] for s, c in zip(s2, ex) if c == r2)
                    row["均等1000_投"] = 1000.0
                    n3, n2 = len(tri), len(ex)
                    e3 = SA.units(np.ones(n3), b3) if tri else []
                    e2 = SA.units(np.ones(n2), b2) if ex else []
                    row["均等1000_払"] = sum(s / 100 * pr["３連単払戻金（円）"] for s, c in zip(e3, tri) if c == r3) + \
                        sum(s / 100 * pr["馬単払戻金（円）"] for s, c in zip(e2, ex) if c == r2)
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "contrib_races.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), (2000, len(d)))

    def roi(v, x=d):
        return x[v + "_払"].sum() / max(x[v + "_投"].sum(), 1) * 100

    def boot(v, x=d, ii=idx):
        i, p = x[v + "_投"].fillna(0).to_numpy(), x[v + "_払"].fillna(0).to_numpy()
        return p[ii].sum(1) / np.maximum(i[ii].sum(1), 1) * 100

    def ci(a):
        return "%.0f〜%.0f" % tuple(np.percentile(a, [5, 95]))

    full = boot("全部")
    m = d["競走年月日"] // 100
    print("対象:", len(d), "レース", d["競走年月日"].min(), "〜", d["競走年月日"].max())
    out = []
    names_ = {"市場だけ": "市場 (土台: 人気順で毎回 ３連単6点+馬単4点)", "−形": "形", "−3倍": "3倍", "−騎手": "騎手", "−クラス": "クラス"}
    for v in ["市場だけ", "−形", "−3倍", "−騎手", "−クラス"]:
        b = boot(v)
        r = {"予想家": names_[v], "外したときの回収率": roi(v), "全部の回収率": roi("全部"),
             "貢献 (ポイント)": roi("全部") - roi(v), "貢献の90%区間": ci(full - b),
             "投資の変化(全部−外す, 円/100R)": (d["全部_投"].sum() - d[v + "_投"].sum()) / len(d) * 100,
             "収支の変化(円/100R)": ((d["全部_払"] - d["全部_投"]).sum() - (d[v + "_払"] - d[v + "_投"]).sum()) / len(d) * 100}
        for lab, mk in [("9月", m == 202609), ("10月", m == 202610), ("11月", m == 202611)]:
            r[lab + "の貢献"] = roi("全部", d[mk]) - roi(v, d[mk])
        out.append(r)
    dd = d.dropna(subset=["配分_投"])
    ii2 = rng.integers(0, len(dd), (2000, len(dd)))
    r = {"予想家": "配分 (1/確率比例. 比べる相手は同じ買い目の1,000円均等)", "外したときの回収率": roi("均等1000", dd),
         "全部の回収率": roi("配分", dd), "貢献 (ポイント)": roi("配分", dd) - roi("均等1000", dd),
         "貢献の90%区間": ci(boot("配分", dd, ii2) - boot("均等1000", dd, ii2))}
    mm = dd["競走年月日"] // 100
    for lab, mk in [("9月", mm == 202609), ("10月", mm == 202610), ("11月", mm == 202611)]:
        r[lab + "の貢献"] = roi("配分", dd[mk]) - roi("均等1000", dd[mk])
    out.append(r)
    t = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(t.round(1).to_string(index=False))
    t.round(1).to_csv(OUT / "contrib_summary.csv", index=False, encoding="utf-8-sig")

    # 能力モデル: ◎ が 1番人気と同じか
    h = d.dropna(subset=["◎"])
    same = h["◎"].astype(int) == h["1番人気"]
    print(f"\n== 能力モデル (◎がわかる {len(h)} レース)")
    for lab, x in [("◎=1番人気", h[same]), ("◎≠1番人気", h[~same])]:
        print(f"{lab}: {len(x)}レース  ルール各100円 {roi('全部', x):.0f}%  配分 {roi('配分', x.dropna(subset=['配分_投'])):.0f}%  "
              f"市場だけ {roi('市場だけ', x):.0f}%")
    hs = h.assign(同=same)
    b_same = []
    for _ in range(2000):
        s = hs.sample(len(hs), replace=True, random_state=rng.integers(1e9))
        b_same.append(roi("全部", s[s["同"]]) - roi("全部", s[~s["同"]]))
    print("差 (◎=1番人気 − ◎≠1番人気) の90%区間:", ci(np.array(b_same)))


if __name__ == "__main__":
    main()
