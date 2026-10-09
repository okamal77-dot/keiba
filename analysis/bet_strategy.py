"""買い方の比較バックテスト（ばんえい除く14場）.

印: 人気・オッズを使わない能力値モデル (ability_index.py の6要素) の順位で ◎○▲△△☆ (上位6頭).
    前年までで学習し翌年を予想する (2024→2025年, 2024-2025→2026年1〜9月).
買い方 (1点100円):
  1) 点数を絞る        : ３連単 ◎→○→▲ / ◎→○▲→○▲ / ◎○▲BOX, 馬単 ◎→○, 馬連 ◎-○, 単勝◎
  2) 人気薄の印から馬連: ◎○▲のうち最も人気のない馬を軸に, 残りの◎○▲へ2点 / ◎○▲△△☆の残りへ5点
  3) 人気薄の印から馬単: 同じ軸を1着に, ◎○▲△△☆の残り5頭へ5点
  2)3) は軸が 何番人気以下 のときだけ買う条件も比べる.
出力: analysis/out/bet_*.csv
"""
import numpy as np
import pandas as pd

import ability_index as A
import predict_race as P
import trifecta_factors as T

KEY = T.KEY
MARKS = ["◎", "○", "▲", "△1", "△2", "☆"]


def payouts():
    raw = pd.concat([T.read(f) for y in T.YEARS for f in T.FILES[y][2]], ignore_index=True)
    out = {}
    spec = {
        "単勝": (["単勝組番"], "単勝払戻金（円）"),
        "馬連": (["馬複組番1", "馬複組番2"], "馬複払戻金（円）"),
        "馬単": (["馬単組番1", "馬単組番2"], "馬単払戻金（円）"),
        "３連複": (["３連複組番馬番1", "３連複組番馬番2", "３連複組番馬番3"], "３連複払戻金（円）"),
        "３連単": (["３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"], "３連単払戻金（円）"),
    }
    for k, (cols, amt) in spec.items():
        x = raw[KEY + cols + [amt]].dropna()
        d = {}
        for r in x.itertuples(index=False):
            combo = tuple(int(v) for v in r[3:3 + len(cols)])
            if k in ("馬連", "３連複"):
                combo = tuple(sorted(combo))
            d.setdefault(tuple(r[:3]), {})[combo] = d.get(tuple(r[:3]), {}).get(combo, 0) + r[-1]
        out[k] = d
    fk = {}
    for i in (1, 2, 3):
        x = raw[KEY + [f"複勝組番{i}", f"複勝払戻金{i}（円）"]].dropna()
        for r in x.itertuples(index=False):
            fk.setdefault(tuple(r[:3]), {})[(int(r[3]),)] = r[4]
    out["複勝"] = fk
    return out


def predict_marks(d, f):
    """前年までで学習 → 翌年の各レースで能力値順位"""
    res = []
    for test_year, train_years in [(2025, [2024]), (2026, [2024, 2025])]:
        tr = d[d["年"].isin(train_years)]
        te = d[d["年"] == test_year]
        X, M, O, _ = P.pad(tr, f.loc[tr.index])
        b, mu, sd = P.fit(X, M, O)
        s = ((f.loc[te.index] - mu) / sd).values @ b
        te = te.assign(能力値=s)
        te["印順"] = te.groupby(KEY)["能力値"].rank(ascending=False, method="first")
        res.append(te)
    return pd.concat(res)


def main():
    h, _, _ = T.load()
    h["予想対象"] = False
    h = T.features(h)
    h = h[h["競馬場"] != "帯広ば"]
    h = A.add_runs(h)
    jr = A.jockey_rates(h)
    d = h[(h["年"] >= 2024) & h["人気"].notna()].copy()
    f = A.features(d, jr)
    t = predict_marks(d, f)
    pay = payouts()

    rows = []
    for k, g in t.groupby(KEY, sort=False):
        if len(g) < 6 or k not in pay["３連単"]:
            continue
        g = g.sort_values("印順")
        m = g["馬番"].astype(int).tolist()[:6]
        pop = dict(zip(g["馬番"].astype(int), g["人気"]))
        top3 = m[:3]
        axis = max(top3, key=lambda x: pop[x])  # ◎○▲のうち最も人気のない馬
        others3 = [x for x in top3 if x != axis]
        others6 = [x for x in m if x != axis]
        P_ = {kind: pay[kind].get(k, {}) for kind in pay}

        def ret(kind, combos):
            if kind in ("馬連", "３連複"):
                combos = [tuple(sorted(c)) for c in combos]
            return len(combos) * 100, sum(P_[kind].get(c, 0) for c in combos), any(c in P_[kind] for c in combos)

        bets = {
            "1-a 単勝◎ (1点)": ret("単勝", [(m[0],)]),
            "1-b 馬連◎-○ (1点)": ret("馬連", [(m[0], m[1])]),
            "1-c 馬単◎→○ (1点)": ret("馬単", [(m[0], m[1])]),
            "1-d ３連単◎→○→▲ (1点)": ret("３連単", [(m[0], m[1], m[2])]),
            "1-e ３連単◎→○▲→○▲ (2点)": ret("３連単", [(m[0], m[1], m[2]), (m[0], m[2], m[1])]),
            "1-f ３連単◎○▲BOX (6点)": ret("３連単", [(a, b, c) for a in top3 for b in top3 for c in top3 if len({a, b, c}) == 3]),
            "1-g ３連複◎○▲ (1点)": ret("３連複", [tuple(top3)]),
            "参考 ３連単◎○→◎○▲△→◎○▲△△ (18点)": ret("３連単", [(a, b, c) for a in m[:2] for b in m[:4] for c in m[:5] if len({a, b, c}) == 3]),
            "2-a 馬連 人気薄の印-◎○▲ (2点)": ret("馬連", [(axis, x) for x in others3]),
            "2-b 馬連 人気薄の印-印5頭 (5点)": ret("馬連", [(axis, x) for x in others6]),
            "2-c 複勝 人気薄の印 (1点)": ret("複勝", [(axis,)]),
            "2-d 単勝 人気薄の印 (1点)": ret("単勝", [(axis,)]),
            "3-a 馬単 人気薄の印→印5頭 (5点)": ret("馬単", [(axis, x) for x in others6]),
            "3-b 馬単 印5頭→人気薄の印 (5点, 参考)": ret("馬単", [(x, axis) for x in others6]),
        }
        for name, (cost, back, hit) in bets.items():
            rows.append((k[0], int(k[1]), k[2], int(str(k[1])[:4]), name, pop[axis], cost, back, hit))

    r = pd.DataFrame(rows, columns=["競馬場", "日付", "R", "年", "買い方", "軸の人気", "購入", "払戻", "的中"])
    r = r.sort_values(["日付", "競馬場", "R"])
    r.to_csv(T.OUT / "bet_rows.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    def summ(x):
        n = len(x)
        back = x["払戻"]
        # 回収率の標準誤差 (レース単位)
        se = (back / x["購入"]).std() / np.sqrt(n) * 100 if n > 1 else np.nan
        return pd.Series({"レース数": n, "的中率%": x["的中"].mean() * 100, "回収率%": back.sum() / x["購入"].sum() * 100,
                          "誤差±%": se, "最大払戻": back.max(),
                          "最長連敗": longest_losing(x["的中"].values)})

    def longest_losing(v):
        best = cur = 0
        for hit in v:
            cur = 0 if hit else cur + 1
            best = max(best, cur)
        return best

    out = []
    for cond, mask in [("全レース", r["軸の人気"] > 0), ("軸が4番人気以下", r["軸の人気"] >= 4), ("軸が6番人気以下", r["軸の人気"] >= 6)]:
        x = r[mask]
        for y in ["2025", "2026", "計"]:
            xy = x if y == "計" else x[x["年"] == int(y)]
            s = xy.groupby("買い方", sort=False).apply(summ)
            s.insert(0, "期間", y)
            s.insert(0, "条件", cond)
            out.append(s)
    res = pd.concat(out)
    res.to_csv(T.OUT / "bet_strategy.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res[res["期間"] == "計"].round(1).to_string())
    print(res[res["期間"] != "計"][["条件", "期間", "レース数", "的中率%", "回収率%"]].round(1).to_string())


if __name__ == "__main__":
    main()
