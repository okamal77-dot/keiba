"""出走表から各馬の着順確率を推定する（人気・オッズを使わない事前情報モデル）.

使い方: python3 analysis/predict_race.py <dataディレクトリ> <出走表horselist.csv> <出走表racelist.csv> <競馬場>

モデル:
  上位3着の着順を使った順位ロジット (Plackett-Luce の上位3段階, いわゆる exploded logit).
  特徴量は trifecta_factors.py の分析で効いていた事前情報のみ (人気・オッズ・当日馬体重は使わない):
    近3走ベスト指数・前走指数・持ち時計 (いずれもレース内の相対値), 前走着順・前走人気, 前走上がり3位内,
    前走脚質, 騎手×馬のコンビ成績, 全成績・当場成績の3着内率, 騎手の前年(直近1年)3着内率,
    他地区からの転入初戦・遠征, 前走からの間隔, 年齢, 牝馬
  学習: 対象競馬場の 2024年1月〜検証開始前, 検証: 直近3か月, 予想: 検証後に全期間で再学習

出力: analysis/out/pred_<日付>_<競馬場>.csv と標準出力の予想表
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY
VALID_MONTHS = 3


def load_entries(horse_file, race_file, course):
    e = pd.read_csv(horse_file, encoding="utf-8-sig", low_memory=False)
    r = pd.read_csv(race_file, encoding="utf-8-sig", low_memory=False)
    e, r = e[e["競馬場"] == course].copy(), r[r["競馬場"] == course].drop_duplicates(KEY)
    for c in ["着順", "人気", "上がり3F", "タイム", "馬体重増減", "馬番", "馬体重"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    r["馬場G"] = np.nan
    e = e.merge(r[KEY + ["距離", "天候", "馬場", "馬場G", "競走種類名称", "レース名"]], on=KEY, how="left")
    e["頭数"] = e.groupby(KEY)["馬番"].transform("size")
    e["top3"] = np.nan
    e["日付"] = pd.to_datetime(e["競走年月日"].astype(str))
    e["年"] = e["日付"].dt.year
    e["馬ID"] = e["馬名"] + "_" + e["生年月日"].astype(str)
    e["予想対象"] = True
    return e, r


def jockey_rate(h, course):
    """騎手の3着内率 (対象場, 前年1年分). 30騎乗分を場平均で縮約"""
    x = h[(h["競馬場"] == course) & h["top3"].notna()]
    base = x["top3"].mean()
    out = {}
    for y in sorted(x["年"].unique()) + [None]:
        if y is None:  # 予想用: 直近365日
            last = x["日付"].max()
            w = x[x["日付"] > last - pd.Timedelta(days=365)]
        else:
            w = x[x["年"] == y - 1]
        g = w.groupby("騎手名")["top3"].agg(["sum", "size"])
        out[y] = ((g["sum"] + 30 * base) / (g["size"] + 30)).to_dict(), base
    return out


def build_features(d, jr):
    """レース内の相対値を含む特徴量行列"""
    f = pd.DataFrame(index=d.index)
    g = d.groupby(KEY)

    def rel(col):
        # 競走中止に近い大敗などの極端値で全体が引きずられないよう ±3 に抑える
        v = d[col].clip(-3, 3)
        return (v - v.groupby([d[k] for k in KEY]).transform("mean")).fillna(0), v.isna().astype(float)

    f["指数"], f["指数なし"] = rel("近3走ベスト指数")
    f["前走指数"], _ = rel("前走指数")
    t = -d["持ち時計"]
    f["持ち時計"] = (t - t.groupby([d[k] for k in KEY]).transform("median")).fillna(0).clip(-3, 3)
    f["持ち時計なし"] = d["持ち時計"].isna().astype(float)
    f["前走着順"] = np.log(d["前走着順"].fillna(6))
    f["前走人気"] = np.log(d["前走人気"].fillna(6))
    f["前走なし"] = d["前走着順"].isna().astype(float)
    f["前走上がり3位内"] = (d["前走上がり順"] <= 3).astype(float)
    for s in ["逃げ", "先行", "差し"]:
        f["前走" + s] = (d["前走脚質"] == s).astype(float)
    f["コンビ3着内率"] = (d["コンビ3着内"].fillna(0) + 1) / (d["コンビ数"].fillna(0) + 4)
    f["初コンビ"] = (d["コンビ数"].fillna(0) == 0).astype(float)
    n, t3 = T.rec_n(d["全成績"])
    f["全成績3着内率"] = (t3.fillna(0) + 1) / (n.fillna(0) + 4)
    n, t3 = T.rec_n(d["当競馬場成績"])
    f["当場3着内率"] = (t3.fillna(0) + 1) / (n.fillna(0) + 4)
    keys = [None if t else y for t, y in zip(d["予想対象"], d["年"])]
    f["騎手3着内率"] = [jr.get(k, jr[None])[0].get(j, jr.get(k, jr[None])[1]) for k, j in zip(keys, d["騎手名"])]
    f["転入初戦"] = (d["前走調教師所属"].notna() & (d["前走調教師所属"] != d["調教師所属"])).astype(float)
    f["遠征"] = (d["調教師所属"] != d["所属地"]).astype(float)
    f["間隔"] = np.log1p(d["間隔日"].fillna(60).clip(0, 365))
    f["年齢"] = d["齢"].clip(2, 9)
    f["牝馬"] = (d["性"] == "牝").astype(float)
    return f


def pad(d, f, with_order=True):
    """レース単位の配列 X[R,H,F], mask[R,H], 1-3着の位置 order[R,3]"""
    races = list(d.groupby(KEY, sort=False).groups.values())
    H = max(len(ix) for ix in races)
    X = np.zeros((len(races), H, f.shape[1]))
    M = np.zeros((len(races), H), bool)
    O = -np.ones((len(races), 3), int)
    for i, ix in enumerate(races):
        X[i, :len(ix)] = f.loc[ix].values
        M[i, :len(ix)] = True
        if with_order:
            pos = d.loc[ix, "着順"].values
            for k in (1, 2, 3):
                w = np.where(pos == k)[0]
                if len(w):
                    O[i, k - 1] = w[0]
    return X, M, O, races


def fit(X, M, O, l2=1.0, iters=30):
    """上位3段階の順位ロジットを Newton 法で推定"""
    R, H, F = X.shape
    mu, sd = X[M].mean(0), X[M].std(0) + 1e-9
    Z = (X - mu) / sd
    b = np.zeros(F)
    ok = (O >= 0).all(1)
    Z, M2, O2 = Z[ok], M[ok], O[ok]
    for _ in range(iters):
        g = -l2 * b
        Hs = -l2 * np.eye(F)
        avail = M2.copy()
        for s in range(3):
            u = np.where(avail, Z @ b, -np.inf)
            p = np.exp(u - u.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            ch = Z[np.arange(len(Z)), O2[:, s]]
            ex = (p[..., None] * Z).sum(1)
            g += (ch - ex).sum(0)
            Hs -= np.einsum("rh,rhi,rhj->ij", p, Z, Z) - ex.T @ ex
            avail[np.arange(len(Z)), O2[:, s]] = False
        step = np.linalg.solve(Hs, g)
        b -= step
        if np.abs(step).max() < 1e-6:
            break
    return b, mu, sd


def win_probs(X, M, b, mu, sd):
    u = np.where(M, ((X - mu) / sd) @ b, -np.inf)
    p = np.exp(u - u.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


STAGE_POW = (1.0, 0.81, 0.65)  # 2・3着の段階では勝率を平らにする (Benter の補正). 検証で較正を確認


def pl_probs(w, pows=STAGE_POW):
    """補正つき Plackett-Luce で 3着内確率と ３連単の全組の確率"""
    n = len(w)
    w2, w3 = w ** pows[1], w ** pows[2]
    top3 = np.zeros(n)
    tri = {}
    for i, j, k in itertools.permutations(range(n), 3):
        pr = w[i] * w2[j] / (w2.sum() - w2[i]) * w3[k] / (w3.sum() - w3[i] - w3[j])
        tri[(i, j, k)] = pr
        top3[[i, j, k]] += pr
    return top3, tri


def suggest(w, tri, nos):
    """勝率の形から買い目を作り, モデル上の的中確率を返す.
    本命の勝率 40%以上: ３連単 ◎ → 2〜5位 → 2〜5位 (12点) + 2位の勝率20%以上なら 2位 → ◎ → 3〜5位 (3点)
    25〜40%: ３連単 1・2位 → 1〜4位 → 1〜5位
    25%未満 (混戦): ３連複 1〜5位 BOX (10点)"""
    o = list((-w).argsort())
    n = nos
    if w[o[0]] >= 0.40:
        combos = [(o[0], j, k) for j in o[1:5] for k in o[1:5] if j != k]
        text = f"３連単 {n[o[0]]} → {','.join(map(str, n[o[1:5]]))} → {','.join(map(str, n[o[1:5]]))}"
        if w[o[1]] >= 0.20:
            combos += [(o[1], o[0], k) for k in o[2:5]]
            text += f" ＋ {n[o[1]]} → {n[o[0]]} → {','.join(map(str, n[o[2:5]]))}"
        return "軸1頭", text, len(combos), sum(tri[c] for c in combos)
    if w[o[0]] >= 0.25:
        combos = [(i, j, k) for i in o[:2] for j in o[:4] for k in o[:5] if len({i, j, k}) == 3]
        text = f"３連単 {','.join(map(str, n[o[:2]]))} → {','.join(map(str, n[o[:4]]))} → {','.join(map(str, n[o[:5]]))}"
        return "2頭軸", text, len(combos), sum(tri[c] for c in combos)
    box = set(o[:5])
    pr = sum(v for c, v in tri.items() if set(c) <= box)
    return "混戦", f"３連複 {','.join(map(str, sorted(n[o[:5]])))} BOX", 10, pr


def evaluate(d, f, b, mu, sd, label):
    X, M, O, _ = pad(d, f)
    P = win_probs(X, M, b, mu, sd)
    ok = O[:, 0] >= 0
    top = P.argmax(1)
    win_hit = (top == O[:, 0])[ok].mean()
    in3 = np.array([t in o for t, o in zip(top, O)])[ok].mean()
    ll = np.log(P[np.arange(len(P)), O[:, 0]][ok]).mean()
    fav = d[d["人気"] == 1].groupby(KEY)["着順"].first()
    print(f"[{label}] {ok.sum()}レース: 本命の勝率 {win_hit:.1%}, 本命の3着内率 {in3:.1%}, 勝ち馬の対数尤度 {ll:.3f}"
          f" / 参考: 1番人気の勝率 {(fav == 1).mean():.1%}, 3着内率 {(fav <= 3).mean():.1%}")


def main():
    data, hfile, rfile, course = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    h, _, _ = T.load()
    e, r = load_entries(hfile, rfile, course)
    allh = pd.concat([h, e], ignore_index=True)
    allh["予想対象"] = allh["予想対象"].astype("boolean").fillna(False).astype(bool)
    allh = T.features(allh)
    jr = jockey_rate(allh, course)
    d = allh[(allh["競馬場"] == course) & (allh["年"] >= 2024)].copy()
    f = build_features(d, jr)

    hist = d[~d["予想対象"]]
    cut = hist["日付"].max() - pd.DateOffset(months=VALID_MONTHS)
    tr, va = hist[hist["日付"] <= cut], hist[hist["日付"] > cut]
    X, M, O, _ = pad(tr, f.loc[tr.index])
    b, mu, sd = fit(X, M, O)
    evaluate(va, f.loc[va.index], b, mu, sd, f"検証 {va['日付'].min():%Y-%m}〜{va['日付'].max():%Y-%m}")

    X, M, O, _ = pad(hist, f.loc[hist.index])
    b, mu, sd = fit(X, M, O)
    coef = pd.Series(b, index=f.columns).sort_values()
    print("\n係数 (標準化後, +ほど上位に来やすい)\n", coef.round(3).to_string())

    tgt = d[d["予想対象"]].sort_values(KEY + ["馬番"])
    X, M, _, races = pad(tgt, f.loc[tgt.index], with_order=False)
    P = win_probs(X, M, b, mu, sd)
    rows = []
    for i, ix in enumerate(races):
        x = tgt.loc[ix]
        w = P[i, :len(ix)]
        top3, tri = pl_probs(w)
        rank = (-w).argsort().argsort()
        marks = np.array(["◎", "○", "▲", "△", "△"] + [""] * 20)[rank]
        good = ((x["指数順"] <= 3).astype(int) + (x["前走上がり順"] <= 3).astype(int)
                + x["前走脚質"].isin(["先行", "差し"]).astype(int)
                + (~((x["コンビ数"] >= 3) & (x["コンビ3着内"] / x["コンビ数"] <= 0.2))).astype(int))
        kesi = (x["指数順"] >= 6) & (x["前走脚質"] == "追込")
        best = sorted(tri.items(), key=lambda kv: -kv[1])[:5]
        nos = x["馬番"].astype(int).values
        kind, buy, npts, hit = suggest(w, tri, nos)
        for j, (_, row) in enumerate(x.iterrows()):
            rows.append({
                "レース": int(row["レース番号"]), "レース名": row["レース名"], "距離": row["距離"],
                "印": marks[j], "馬番": int(row["馬番"]), "馬名": row["馬名"], "騎手": row["騎手名"],
                "勝率%": round(w[j] * 100, 1), "3着内率%": round(top3[j] * 100, 1),
                "前走": f"{row['前走競馬場'] if pd.notna(row['前走競馬場']) else '-'}"
                        f"{'' if pd.isna(row['前走着順']) else int(row['前走着順'])}着"
                        f"({'' if pd.isna(row['前走人気']) else int(row['前走人気'])}人気)",
                "前走日": "" if pd.isna(row["前走日付"]) else f"{row['前走日付']:%m/%d}",
                "指数順": "" if pd.isna(row["指数順"]) else int(row["指数順"]),
                "前走上がり順": "" if pd.isna(row["前走上がり順"]) else int(row["前走上がり順"]),
                "前走脚質": row["前走脚質"] if pd.notna(row["前走脚質"]) else "",
                "好材料(体重除く4点)": int(good.iloc[j]), "消し": bool(kesi.iloc[j]),
                "上位組(３連単)": " / ".join(f"{nos[a]}-{nos[bb]}-{nos[c]} {p*100:.1f}%" for (a, bb, c), p in best) if j == 0 else "",
                "買い目": f"[{kind}] {buy} ({npts}点, モデル上の的中確率 {hit*100:.0f}%)" if j == 0 else "",
            })
    out = pd.DataFrame(rows)
    date = int(tgt["競走年月日"].iloc[0])
    path = T.OUT / f"pred_{date}_{course}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for k, g in out.groupby("レース"):
        print(f"\n## {k}R {g['レース名'].iloc[0]} {g['距離'].iloc[0]}m")
        print(g.sort_values("勝率%", ascending=False).drop(columns=["レース", "レース名", "距離", "上位組(３連単)", "買い目"]).to_string(index=False))
        print("３連単 上位:", g["上位組(３連単)"].iloc[0])
        print("買い目:", g["買い目"].iloc[0])


if __name__ == "__main__":
    main()
