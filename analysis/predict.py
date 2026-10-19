"""３連単の予想支援: 的中要因と競馬場別の万馬券特徴を合わせて, レースごとの買い目を出す.

入力 (どちらも UTF-8 CSV, 列名は analysis/predict_sample/ を参照):
  races.csv   : 競馬場,R,距離,馬場,頭数,クラス,種別,1人気オッズ
  runners.csv : 競馬場,R,馬番,人気,単勝オッズ,脚質,騎手,移籍   (任意. 無ければ人気順だけで出す)

使い方:
  python3 analysis/predict.py analysis/predict_sample/races.csv analysis/predict_sample/runners.csv

仕組み (詳細は analysis/REPORT_trifecta_strategy.md):
  1. 条件 (頭数・R番号・距離・馬場・種別・クラス・1番人気オッズ・年次傾向) から,
     その競馬場の集計表 (analysis/out/*.csv) を使って「万馬券率」と「上位4人気決着率」を推定する.
  2. 単勝オッズ (無ければ人気順) から各馬の勝率を作り, Harville 型モデルで全着順の確率を計算する.
     2・3着は人気薄が来やすい補正を入れ, 「上位4人気決着率」が 1. の推定値に合うよう荒れ具合を合わせる.
  3. 穴馬 (6番人気以下) は騎手の穴期待比・その場の穴馬の脚質・移籍種別で, 危ない1番人気は脚質・騎手で補正する.
  4. 確率の高い順に買い目を並べ, 判定 (堅/中/荒) ごとの点数で切る.
"""
import itertools
import math
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
ALIAS = {"帯広": "帯広ば", "ばんえい": "帯広ば", "帯広(ば)": "帯広ば"}
STYLES = ["逃げ", "先行", "差し", "追込"]
POINTS = {"堅": 12, "中": 24, "荒": 36}          # 判定ごとの基本点数
TOP4_MEDIAN_PAY = {}                              # 上位4人気決着時の３連単中央値 (トリガミ判定用)


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rd(name):
    return pd.read_csv(OUT / name, encoding="utf-8-sig")


# ---------------------------------------------------------------- 集計表の読み込み
def load():
    t = {}
    s = rd("summary.csv").set_index("競馬場")
    t["summary"] = s
    t["fs"] = rd("by_fieldsize.csv").set_index("競馬場")
    t["rno"] = rd("by_raceno.csv").set_index("競馬場")
    t["dist"] = rd("by_distance.csv").set_index("競馬場")
    t["baba"] = rd("baba_rate.csv").set_index("競馬場")
    t["baba_n"] = rd("baba_n.csv").set_index("競馬場")
    cc = rd("class_compare.csv")
    t["type"] = cc.set_index(["競馬場", "クラス"])
    t["cls"] = rd("top4_by_class.csv").set_index(["競馬場", "クラス"])
    t["top4"] = rd("top4_summary.csv").set_index("競馬場")
    t["top4_fs"] = rd("top4_by_fieldsize.csv").set_index("競馬場")
    t["top4_odds"] = rd("top4_rate_by_fav_odds.csv").set_index("競馬場")
    t["top4_odds_n"] = rd("top4_n_by_fav_odds.csv").set_index("競馬場")
    t["style_ana"] = rd("style_ana.csv").set_index("競馬場")
    t["fav_flop"] = rd("style_fav_flop.csv").set_index("競馬場")
    t["win_pop"] = rd("winner_pop_pattern.csv").set_index("競馬場")
    jk = pd.concat([rd(f) for f in ["jockey_ana_ratio.csv", "jockey_ana_roi.csv",
                                    "jockey_ana_count.csv", "jockey_fav_flop.csv"]])
    t["jockey"] = jk.drop_duplicates(["競馬場", "騎手名"]).set_index(["競馬場", "騎手名"])
    t["transfer"] = rd("transfer_first_by_track.csv").set_index(["競馬場", "種別"])
    for k, v in t["top4"]["4人気決着時_3連単中央値"].items():
        TOP4_MEDIAN_PAY[k] = v
    return t


def fs_bucket(n):
    return "~8頭" if n <= 8 else "9-10頭" if n <= 10 else "11-12頭" if n <= 12 else "13頭~"


def rno_bucket(r):
    return "1-4R" if r <= 4 else "5-8R" if r <= 8 else "9R~"


def dist_bucket(d):
    return "~1200" if d <= 1200 else "1201-1500" if d <= 1500 else "1501-1800" if d <= 1800 else "1801~"


def odds_bucket(o):
    return "1.0-1.4倍" if o < 1.5 else "1.5-1.9倍" if o < 2.0 else "2.0-2.9倍" if o < 3.0 \
        else "3.0-3.9倍" if o < 4.0 else "4.0倍~"


def get(df, row, col):
    try:
        v = df.loc[row, col]
    except KeyError:
        return None
    return None if pd.isna(v) else float(v)


def baba_key(track, baba):
    """帯広は水分量 (数値%) で受ける. それ以外は 良/稍重/重/不良."""
    if track != "帯広ば":
        return baba
    try:
        w = float(str(baba).rstrip("%"))
    except ValueError:
        return None
    return "水分~1.4%(重い)" if w < 1.5 else "1.5-2.4%" if w < 2.5 else "2.5%~(軽い)"


def race_class(name):
    """クラス欄の表記ゆれを top4_by_class のクラス名に寄せる."""
    s = str(name or "")
    for k in ["2歳", "3歳", "重賞"]:
        if k in s:
            return k
    for k in ["A", "B", "C"]:
        if s.upper().startswith(k):
            return k
    return None


# ---------------------------------------------------------------- 1. 条件からの推定
def estimate(t, r):
    """戻り値: (推定万馬券率, 推定上位4人気決着率, 根拠のリスト)."""
    tr = r["競馬場"]
    s = t["summary"].loc[tr]
    base = s["万馬券率"]
    notes = []

    # 頭数: 最大の要因なので頭数別の値を出発点にする
    fb = fs_bucket(r["頭数"])
    p_fs = get(t["fs"], tr, fb) or get(t["fs"], tr, "11-12頭") or base
    x = logit(p_fs)
    notes.append(f"頭数{r['頭数']}({fb}) 万馬券率{p_fs:.0%}")

    # 年次傾向: 2026年の値と全期間の差を半分だけ反映 (佐賀の低下・姫路の上昇など)
    y26 = s["万馬券率2026"]
    d = 0.5 * (logit(y26) - logit(base))
    x += d
    if abs(d) > 0.05:
        notes.append(f"2026年傾向 {y26:.0%}({'↑' if d > 0 else '↓'})")

    # R番号・距離・馬場・クラスは頭数と重なる部分があるので半分の重みで足す
    for label, df, key in [("R", t["rno"], rno_bucket(r["R"])),
                           ("距離", t["dist"], dist_bucket(r["距離"]) if r["距離"] else None),
                           ("馬場", t["baba"], baba_key(tr, r["馬場"]))]:
        v = get(df, tr, key) if key else None
        if v is None:
            continue
        w = 0.5
        if label == "馬場":
            n = get(t["baba_n"], tr, key) or 0
            w *= n / (n + 300)            # 件数の少ない馬場 (姫路の不良など) は弱める
        d = w * (logit(v) - logit(base))
        x += d
        if abs(d) > 0.08:
            notes.append(f"{label}{key} {v:.0%}")

    # 種別 (一般/特別/重賞): class_compare の「頭数補正後の差」をそのまま使う
    kind = r["種別"] or "一般"
    cls = race_class(r["クラス"])
    v = get(t["type"], (tr, kind), "頭数補正後の差")
    if v:
        x += logit(min(0.95, max(0.05, sigmoid(x) + v / 100))) - x
        if abs(v) >= 3:
            notes.append(f"{kind}戦 頭数補正後{v:+.1f}pt")
    # クラス (A/B/C/2歳/3歳): 重賞は種別で反映済み
    if cls and cls != "重賞":
        v = get(t["cls"], (tr, cls), "万馬券率")
        if v is not None:
            d = 0.3 * (logit(v / 100) - logit(base))     # クラスは頭数と重なりが大きいので弱め
            x += d
            if abs(d) > 0.05:
                notes.append(f"{cls}級 万馬券率{v:.0f}%")

    # ---- 上位4人気決着率
    t4 = get(t["top4_fs"], tr, fb) or get(t["top4_fs"], tr, "11-12頭") or t["top4"].loc[tr, "1-3着が上位4人気"]
    y = logit(t4 / 100)
    fav = r["1人気オッズ"]
    if fav:
        ob = odds_bucket(fav)
        all_rate = t["top4_odds"][ob].mean()
        v = get(t["top4_odds"], tr, ob)
        n = get(t["top4_odds_n"], tr, ob) or 0
        v = all_rate if v is None else (v * n + all_rate * 100) / (n + 100)   # 件数で全場平均に寄せる
        ref = t["top4_odds"].loc[tr].mean()
        d = logit(v / 100) - logit(ref / 100)
        y += d
        y -= 0.4 * d                     # 頭数が少ないほど1番人気は抜けるので, 重なり分を戻す
        x -= 0.3 * 0.6 * d               # 1番人気が抜けているほど万馬券も減る
        notes.append(f"1人気{fav}倍 上位4頭決着{v:.0f}%")
    if cls:
        v = get(t["cls"], (tr, cls), "上位4人気決着率")
        if v is not None:
            y += 0.3 * (logit(v / 100) - logit(t["top4"].loc[tr, "1-3着が上位4人気"] / 100))
    # 要因は互いに重なるので, 頭数別の値からの動きは logit で ±1 までに抑える
    x = min(max(x, logit(p_fs) - 1), logit(p_fs) + 1)
    y = min(max(y, logit(t4 / 100) - 1), logit(t4 / 100) + 1, logit(0.6))   # 6割超は過信になるので頭打ち
    return sigmoid(x), sigmoid(y), notes


# ---------------------------------------------------------------- 2. 着順確率モデル
def harville(p2, p3, p1):
    """p1: 1着の強さ, p2: 2着, p3: 3着. 全3頭順列の確率を返す."""
    n = len(p1)
    res = {}
    for i, j, k in itertools.permutations(range(n), 3):
        a = p1[i]
        b = p2[j] / (1 - p2[i])
        c = p3[k] / (1 - p3[i] - p3[j])
        res[(i, j, k)] = a * b * c
    tot = sum(res.values())
    return {k: v / tot for k, v in res.items()}


def norm(v):
    s = sum(v)
    return [x / s for x in v]


def strengths(base, a, mult2, mult3):
    """1着の強さは単勝オッズの比のまま. 2・3着は指数 a, 0.8a で平らにする (Lo-Bacon-Shone 型).
    単純な Harville は人気馬の2・3着を過大評価するため. 堅いレースで a>1 になるときだけ1着も尖らせる."""
    p1 = norm([b ** max(1.0, a) for b in base])
    p2 = norm([(b ** a) * m for b, m in zip(base, mult2)])
    p3 = norm([(b ** (a * 0.8)) * m for b, m in zip(base, mult3)])
    return p1, p2, p3


def top4_prob(probs, pops):
    return sum(v for (i, j, k), v in probs.items() if max(pops[i], pops[j], pops[k]) <= 4)


def calibrate(base, pops, target, mult1, mult2, mult3):
    """上位4人気決着率が target になる 2・3着の指数を二分法で探す."""
    lo, hi = 0.2, 2.0
    for _ in range(25):
        mid = (lo + hi) / 2
        p1, p2, p3 = strengths(base, mid, mult2, mult3)
        p1 = norm([a * m for a, m in zip(p1, mult1)])
        if top4_prob(harville(p2, p3, p1), pops) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def default_odds(n):
    """オッズが無いときの人気別の目安 (地方競馬の平均的な単勝オッズ)."""
    curve = [2.0, 4.2, 6.5, 9.5, 14, 20, 28, 40, 55, 75, 100, 130, 160, 200, 250, 300]
    return curve[:n] + [300] * max(0, n - len(curve))


# ---------------------------------------------------------------- 3. 馬ごとの補正
def runner_factors(t, tr, h):
    """戻り値: (1着倍率, 2着倍率, 3着倍率, メモ)."""
    pop = h["人気"]
    m1 = m2 = m3 = 1.0
    memo = []
    jk = h.get("騎手") or ""
    st = h.get("脚質") or ""
    tf = h.get("移籍") or ""
    jrow = t["jockey"].loc[(tr, jk)] if (tr, jk) in t["jockey"].index else None

    if pop >= 6:
        f = 1.0
        if jrow is not None and jrow["穴期待比"] >= 1.2:
            f *= min(1.6, jrow["穴期待比"]) ** 0.7
            memo.append(f"穴騎手{jk}({jrow['穴期待比']:.2f}倍)")
        if st in STYLES:
            share = t["style_ana"].loc[tr, st]
            ref = t["style_ana"][st].mean()
            g = (share / ref) ** 0.5
            f *= g
            if g >= 1.1:
                memo.append(f"{st}の穴が多い場")
        key = {"中央": "中央から転入(推定)", "地方間": "地方間移籍", "転厩": "同地区内の転厩"}.get(tf)
        if key and (tr, key) in t["transfer"].index:
            g = t["transfer"].loc[(tr, key), "人気比_複勝率"]
            f *= g
            memo.append(f"{tf}移籍(複勝人気比{g:.2f}{'' if g >= 1 else ' 割引'})")
        # 穴は3着 > 2着 > 1着 の順に来やすい (ana_finish_pos: 約44% / 35% / 21%)
        m1, m2, m3 = f ** 0.5, f ** 0.8, f
    if pop == 1:
        risk = []
        if st in ("差し", "追込"):
            risk.append(f"{st}の1番人気")
        avg_out = 1 - t["summary"].loc[tr, "1人気複勝率_全体"]      # その場の1番人気の平均着外率
        if jrow is not None and jrow["1人気騎乗"] >= 30 and jrow["1人気着外率"] >= max(0.30, avg_out + 0.05):
            risk.append(f"{jk}の1人気着外率{jrow['1人気着外率']:.0%}")
        if risk:
            # 1番人気凡走の55〜70%は差し・追込. 1・2着の評価だけ下げる (3着は残す)
            m1, m2 = 0.8 ** len(risk), 0.9 ** len(risk)
            memo.append("危険: " + "・".join(risk))
    return m1, m2, m3, memo


# ---------------------------------------------------------------- 4. 買い目
def verdict(pm, t4):
    if pm < 0.38 and t4 >= 0.38:
        return "堅"
    if pm >= 0.58 or t4 < 0.27:
        return "荒"
    return "中"


def risky_fav(fac, hs):
    return any(h["人気"] == 1 and any(m.startswith("危険") for m in f[3]) for h, f in zip(hs, fac))


def formation(combos):
    pos = [sorted({c[i] for c in combos}) for i in range(3)]
    return " → ".join(",".join(str(x) for x in p) for p in pos)


def predict_race(t, r, hs):
    tr = r["競馬場"]
    pm, t4, notes = estimate(t, r)
    hs = sorted(hs, key=lambda h: h["人気"])
    n = len(hs)
    odds = [h.get("単勝オッズ") for h in hs]
    if any(o is None or (isinstance(o, float) and math.isnan(o)) for o in odds):
        odds = default_odds(n)
        if r["1人気オッズ"]:
            odds[0] = r["1人気オッズ"]
    base = norm([1 / o for o in odds])
    pops = [h["人気"] for h in hs]
    fac = [runner_factors(t, tr, h) for h in hs]
    m1, m2, m3 = ([f[i] for f in fac] for i in range(3))
    a = calibrate(base, pops, t4, m1, m2, m3)
    p1, p2, p3 = strengths(base, a, m2, m3)
    p1 = norm([a * m for a, m in zip(p1, m1)])
    probs = harville(p2, p3, p1)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    v = verdict(pm, t4)
    if v == "荒":
        # 荒判定では上位3人気だけの組み合わせ (配当が安い) を外し, その分を穴絡みに回す
        ranked = [kv for kv in ranked if max(pops[i] for i in kv[0]) > 3]
    k = min(POINTS[v], len(ranked))
    pick = ranked[:k]
    hit = sum(p for _, p in pick)
    num = [h.get("馬番") or h["人気"] for h in hs]
    tickets = [(num[i], num[j], num[kk]) for (i, j, kk), _ in pick]
    fav_in3 = sum(p for (i, j, kk), p in probs.items() if 0 in (i, j, kk))
    warn = []
    pay = TOP4_MEDIAN_PAY.get(tr)
    if v == "堅" and pay and pay < k * 100 * 1.5:
        warn.append(f"上位4頭決着の中央値{pay:,.0f}円に対し{k}点はトリガミ注意 → 1着固定で6〜9点に絞る")
    if v == "荒":
        warn.append("上位3人気だけの組み合わせは除外済み. 穴は3着付けを厚く")
    if risky_fav(fac, hs):
        warn.append("1番人気は1着固定にせず, 2・3着付けも押さえる")
    ana = sorted([(num[i], hs[i]["人気"], fac[i][3], fac[i][2]) for i in range(n)
                  if hs[i]["人気"] >= 6 and fac[i][2] >= 1.1], key=lambda a: -a[3])[:4]
    risky = [fac[i][3] for i in range(n) if hs[i]["人気"] == 1 and fac[i][3]]
    return dict(track=tr, R=r["R"], pm=pm, t4=t4, verdict=v, notes=notes, k=k, hit=hit,
                tickets=tickets, fav_in3=fav_in3, warn=warn, ana=ana, risky=risky,
                form=formation(tickets), has_num=bool(hs and hs[0].get("馬番")))


# ---------------------------------------------------------------- 入出力
def read_inputs(races_path, runners_path=None):
    r = pd.read_csv(races_path, encoding="utf-8-sig")
    r["競馬場"] = r["競馬場"].replace(ALIAS)
    for c in ["距離", "1人気オッズ"]:
        r[c] = pd.to_numeric(r.get(c), errors="coerce")
    r = r.astype(object).where(r.notna(), None)
    h = None
    if runners_path:
        h = pd.read_csv(runners_path, encoding="utf-8-sig")
        h["競馬場"] = h["競馬場"].replace(ALIAS)
        h["単勝オッズ"] = pd.to_numeric(h.get("単勝オッズ"), errors="coerce")
        h = h.astype(object).where(h.notna(), None)
    return r, h


def render(res):
    lines = [f"## {res['track'].replace('帯広ば', '帯広')} {res['R']}R ― 判定【{res['verdict']}】",
             f"- 推定万馬券率 **{res['pm']:.0%}** ／ 上位4人気決着 **{res['t4']:.0%}** ／ 1番人気の3着内 {res['fav_in3']:.0%}",
             f"- 根拠: {' / '.join(res['notes'])}"]
    if res["risky"]:
        lines.append(f"- 1番人気: {res['risky'][0][0]}")
    if res["ana"]:
        lines.append("- 穴候補: " + "、".join(f"{a}番({p}人気: {'・'.join(m)})" if res["has_num"]
                                            else f"{p}人気({'・'.join(m)})" for a, p, m, _ in res["ana"]))
    lab = "馬番" if res["has_num"] else "人気"
    lines.append(f"- 買い目（{lab}, 確率上位{res['k']}点, モデル上の的中率 {res['hit']:.0%}）: {res['form']}")
    lines.append("  - " + " / ".join("-".join(map(str, c)) for c in res["tickets"]))
    for w in res["warn"]:
        lines.append(f"- ⚠ {w}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    t = load()
    races, runners = read_inputs(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    out = []
    for _, r in races.iterrows():
        r = r.to_dict()
        r["R"], r["頭数"] = int(r["R"]), int(r["頭数"])
        if runners is not None:
            hs = runners[(runners["競馬場"] == r["競馬場"]) & (runners["R"].astype(int) == r["R"])]
            hs = [dict(h, 人気=int(h["人気"]), 馬番=int(h["馬番"]) if h.get("馬番") else None)
                  for h in hs.to_dict("records")]
        else:
            hs = []
        if len(hs) < 3:
            hs = [{"人気": i + 1} for i in range(r["頭数"])]
        out.append(render(predict_race(t, r, hs)))
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
