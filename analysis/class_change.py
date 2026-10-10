"""クラス (級・段・組) の変化と着順が勝率にどれだけ寄与するか (2024〜2026年10月, ばんえい含む全場).

レース名から クラス = (級 A/B/C, 段 1,2,.., 組 1,2,..) を読み取る (数字が小さいほど上).
  例: 佐賀「Ｃ２－２５組」= C・2段・25組, 大井「Ｃ１二 三 四」= C・1段・2組, 盛岡「Ｃ２十二組」= C・2段・12組,
      名古屋・笠松「Ｃ１６組」= C・(段なし)・16組, 門別「Ｂ４－２～Ｃ２－１」= 先頭の B・4段・2組
比較は同じ地区 (園田・姫路=兵庫, 盛岡・水沢=岩手 など) の中だけ.

 1) 前走→今回 のクラス変化 (昇級 / 組上がり / 同じ組 / 組下がり / 降級) × 前走着順
 2) 4走前→今回 の変化 (段・級の上下と組の上下) × 近4走の平均着順
 3) 人気を使わない順位ロジットに クラス変化 を加えたときの予測の改善 (直近3か月で検証)
指標: 勝率, 3着内率, A/E (同じ年・場・人気・頭数帯の平均3着内率に対する比), 勝ち A/E (同, 勝率)
出力: analysis/out/cc_*.csv
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

import predict_race as P
import trifecta_factors as T

KEY = T.KEY
KANJI = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
NO_DAN = {"名古屋", "笠松"}  # 「Ｃ１６組」= 組のみ
CLS = re.compile(r"(?<![A-Za-z])([ABC])(?![A-Za-z])(\d{0,2})\s*(?:[-－‐]\s*)?(\d+|[一二三四五六七八九十]+)?\s*(組)?")


def kanji_num(s):
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + KANJI[s[1]]
    if "十" in s:
        a, b = s.split("十")
        return KANJI[a] * 10 + (KANJI[b] if b else 0)
    return KANJI.get(s)


def parse_class(name, course):
    """レース名 → (級 0/1/2, 段, 組). 読めなければ None"""
    t = unicodedata.normalize("NFKC", str(name))
    if re.search(r"[23]歳", t) and not re.search(r"[23]歳以上", t):
        return None  # 2歳・3歳限定戦は対象外
    for m in CLS.finditer(t):
        letter, d1, d2, kumi = m.groups()
        if d2 and d2.isdigit() and t[m.end(3):m.end(3) + 1] == "歳":
            d2 = None  # 「Ｃ３ ４歳以上」の 4 は年齢
        lv = "ABC".index(letter)
        if course in NO_DAN:
            num = d1 or d2
            if not num:
                return (lv, 0, None)
            return (lv, 0, kanji_num(num))
        if not d1:
            # 「Ｃ－１」「ＡＢ」などは段なし
            return (lv, 0, kanji_num(d2) if d2 else None)
        if len(d1) == 2 and not d2 and not kumi:
            return (lv, int(d1[0]), int(d1[1]))  # 「Ｃ１１」= C1 の1組 とみなす
        return (lv, int(d1), kanji_num(d2) if d2 else None)
    return None


def change_label(prev, cur):
    if prev is None or cur is None:
        return "比較できない"
    pt, ct = prev[0] * 10 + prev[1], cur[0] * 10 + cur[1]
    if ct < pt:
        return "昇級(段・級が上がる)"
    if ct > pt:
        return "降級(段・級が下がる)"
    if prev[2] is None or cur[2] is None:
        return "同じ段(組不明)"
    if cur[2] < prev[2]:
        return "組上がり"
    if cur[2] > prev[2]:
        return "組下がり"
    return "同じ組"


def rate_table(x, a, b=None):
    keys = [a] if b is None else [a, b]
    g = x.groupby(keys, observed=True)
    t = pd.DataFrame({"頭数": g.size(), "勝率%": g["勝"].mean() * 100, "3着内率%": g["top3"].mean() * 100,
                      "勝ちA/E": g["勝"].sum() / g["期待勝"].sum(), "3着内A/E": g["top3"].sum() / g["期待"].sum(),
                      "平均人気": g["人気"].mean()})
    return t


def main():
    h, r, _ = T.load()
    h = T.features(h)
    r = r.drop_duplicates(KEY)
    h = h.merge(r[KEY + ["レース名"]], on=KEY, how="left")
    h["所属地"] = h["競馬場"].map(T.HOME).fillna(h["競馬場"])
    h["クラス"] = [parse_class(n, c) for n, c in zip(h["レース名"], h["競馬場"])]
    h["勝"] = (h["着順"] == 1).astype(int)
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    grp = h.groupby(["年", "競馬場", "人気c", "頭数帯"], observed=True)
    h["期待"] = grp["top3"].transform("mean")
    h["期待勝"] = grp["勝"].transform("mean")

    # 読み取り結果の確認用
    ex = h.drop_duplicates(["競馬場", "レース名"]).groupby("競馬場").head(6)[["競馬場", "レース名", "クラス"]]
    ex.to_csv(T.OUT / "cc_parse_examples.csv", index=False, encoding="utf-8-sig")
    print(ex.to_string(index=False))
    rate = h.groupby("競馬場")["クラス"].apply(lambda s: s.notna().mean() * 100)
    print("クラスを読めたレースの割合%\n", rate.round(1).to_string())

    # ---- 前走〜4走前 ----
    h = h.sort_values(["馬ID", "日付", "レース番号"])
    g = h.groupby("馬ID")
    for i in range(1, 5):
        h[f"{i}走前クラス"] = g["クラス"].shift(i)
        h[f"{i}走前着順"] = g["着順"].shift(i)
        h[f"{i}走前所属地"] = g["所属地"].shift(i)
    same1 = h["1走前所属地"] == h["所属地"]
    h["前走からの変化"] = [change_label(p, c) if s else "比較できない"
                       for p, c, s in zip(h["1走前クラス"], h["クラス"], same1)]
    h["前走着順"] = pd.cut(h["1走前着順"], [0, 1, 3, 5, 99], labels=["1着", "2-3着", "4-5着", "6着~"])

    # 4走前 (なければ一番古い近走) → 今回: 段・級の差 と 組の差
    def net(row):
        cur = row["クラス"]
        for i in (4, 3, 2):
            p = row[f"{i}走前クラス"]
            if isinstance(p, tuple) and row[f"{i}走前所属地"] == row["所属地"]:
                break
        else:
            return np.nan, np.nan
        if not isinstance(cur, tuple):
            return np.nan, np.nan
        tier = (p[0] * 10 + p[1]) - (cur[0] * 10 + cur[1])  # + なら上がった
        kumi = (p[2] - cur[2]) if tier == 0 and p[2] is not None and cur[2] is not None else 0
        return tier, kumi

    d = h[h["年"] >= 2024].copy()
    nets = [net(rw) for rw in d[["クラス"] + [f"{i}走前クラス" for i in (2, 3, 4)] + [f"{i}走前所属地" for i in (2, 3, 4)] + ["所属地"]].to_dict("records")]
    d["段の変化"], d["組の変化"] = zip(*nets)
    d["4走前からの変化"] = np.select(
        [d["段の変化"] >= 2, d["段の変化"] == 1, (d["段の変化"] == 0) & (d["組の変化"] >= 3),
         (d["段の変化"] == 0) & (d["組の変化"].between(1, 2)), (d["段の変化"] == 0) & (d["組の変化"] == 0),
         (d["段の変化"] == 0) & (d["組の変化"] < 0), d["段の変化"] < 0],
        ["2段以上上がった", "1段上がった", "同じ段で組が3以上上がった", "同じ段で組が1-2上がった", "同じ組",
         "同じ段で組が下がった", "段・級が下がった"], "比較できない")
    d["近4走平均着順"] = d[[f"{i}走前着順" for i in range(1, 5)]].mean(axis=1)
    d["近4走の成績"] = pd.cut(d["近4走平均着順"], [0, 2.5, 4, 6, 99], labels=["平均2.5着以内", "2.5-4着", "4-6着", "6着より下"])
    d = d[d["人気"].notna()]

    out = {}
    out["cc_prev_change"] = rate_table(d, "前走からの変化")
    out["cc_prev_change_x_finish"] = rate_table(d, "前走からの変化", "前走着順")
    out["cc_4run_change"] = rate_table(d, "4走前からの変化")
    out["cc_4run_change_x_form"] = rate_table(d, "4走前からの変化", "近4走の成績")
    # 年別の再現性 (前走からの変化 × 前走着順 の 勝ちA/E)
    yy = d.groupby(["前走からの変化", "前走着順", "年"], observed=True).apply(
        lambda x: x["勝"].sum() / x["期待勝"].sum()).unstack()
    out["cc_prev_change_x_finish_by_year"] = yy
    for name, t in out.items():
        t.to_csv(T.OUT / f"{name}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for name in ["cc_prev_change", "cc_prev_change_x_finish", "cc_4run_change", "cc_4run_change_x_form"]:
        print(f"\n== {name}\n", out[name].round(2).to_string())
    print("\n== 勝ちA/E 年別\n", yy.round(2).to_string())

    # ---- 3) 予測への寄与: 人気なしの順位ロジットに クラス変化 を加える ----
    d = d.reset_index(drop=True)
    base = pd.DataFrame(index=d.index)
    base["前走着順(対数)"] = np.log(d["1走前着順"].fillna(6))
    base["近4走平均着順"] = d["近4走平均着順"].fillna(6)
    base["近3走ベスト指数"] = d["近3走ベスト指数"].clip(-3, 3).fillna(d["近3走ベスト指数"].mean())
    n, t3 = T.rec_n(d["全成績"])
    base["通算3着内率"] = ((t3.fillna(0) + 1) / (n.fillna(0) + 4)).values
    cls = pd.get_dummies(d["前走からの変化"]).astype(float).drop(columns=["比較できない"], errors="ignore")
    cls4 = pd.get_dummies(d["4走前からの変化"]).astype(float).drop(columns=["比較できない"], errors="ignore")
    cls4.columns = ["4走前比:" + c for c in cls4.columns]
    full = pd.concat([base, cls, cls4], axis=1)
    cut = d["日付"].max() - pd.DateOffset(months=P.VALID_MONTHS)
    tr, va = d[d["日付"] <= cut], d[d["日付"] > cut]
    rows = []
    for label, f in [("着順・時計・通算成績のみ", base), ("＋クラス変化", full)]:
        X, M, O, _ = P.pad(tr, f.loc[tr.index])
        b, mu, sd = P.fit(X, M, O)
        Xv, Mv, Ov, _ = P.pad(va, f.loc[va.index])
        Pw = P.win_probs(Xv, Mv, b, mu, sd)
        ok = Ov[:, 0] >= 0
        top = Pw.argmax(1)
        rows.append({"モデル": label, "検証レース数": int(ok.sum()),
                     "本命の勝率%": (top == Ov[:, 0])[ok].mean() * 100,
                     "勝ち馬の対数尤度": np.log(Pw[np.arange(len(Pw)), Ov[:, 0]][ok]).mean()})
        if label == "＋クラス変化":
            coef = pd.Series(b, index=f.columns)
    t = pd.DataFrame(rows).set_index("モデル")
    t.to_csv(T.OUT / "cc_model_gain.csv", encoding="utf-8-sig")
    coef.to_csv(T.OUT / "cc_model_coef.csv", encoding="utf-8-sig")
    print("\n== 予測への寄与\n", t.round(3).to_string())
    print("\n係数 (標準化後)\n", coef.round(3).sort_values().to_string())


if __name__ == "__main__":
    sys.exit(main())
