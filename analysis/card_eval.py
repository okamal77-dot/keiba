"""出馬表のオッズを使った 1レース分のまとめ (貼り付けられた出馬表への回答用).

使い方: python3 analysis/card_eval.py <日付YYYYMMDD> <競馬場> <R> "1:44.8,2:82.2,..."
  analysis/out/pred2_<日付>_<競馬場>.csv の印・勝率・3着内率に, 実際の単勝オッズで計算した期待値を付ける.
  期待値 = 補正勝率 (期待値の補正 第22節) × 単勝オッズ.
出力: ◎○▲☆(△) の 勝率/3着内率/期待値, 買い目, 印以外で消しでない期待値上位 (1.2〜2.0は太字), 消しの馬.
"""
import sys

import pandas as pd

import trifecta_factors as T


def main():
    date, course, rno, odds = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    od = {int(k): float(v) for k, v in (x.split(":") for x in odds.replace(" ", "").split(",") if x)}
    d = pd.read_csv(T.OUT / f"pred2_{date}_{course}.csv")
    g = d[d["レース"] == rno].copy()
    g["オッズ"] = g["馬番"].map(od)
    g["人気"] = g["オッズ"].rank(method="min")
    g["期待値"] = g["補正勝率%"] / 100 * g["オッズ"]
    g["消し"] = g["注記"].fillna("").str.contains("消し条件")
    fav = g.sort_values("オッズ").iloc[0]
    top = g[g["印"] == "◎"].iloc[0]
    sym = {"◎": "◎", "○": "○", "▲": "▲", "△": "☆"}
    print(f"## {course} {rno}R {g['レース名'].iloc[0]} {g['距離'].iloc[0]}m")
    print("| 印 | 馬番 馬名 | 人気(オッズ) | 勝率 | 3着内率 | 期待値 | 注記 |\n|---|---|---|---|---|---|---|")
    for k in ["◎", "○", "▲", "△"]:
        for _, x in g[g["印"] == k].iterrows():
            b = "**" if 1.2 <= x.期待値 <= 2.0 else ""  # 期待値1.2〜2.0の帯に入る馬は太字
            print(f"| {sym[k]} | {b}{x.馬番} {x.馬名}{b} | {x.人気:.0f}人気({x.オッズ}) | {x['勝率%']:.0f}% | {x['3着内率%']:.0f}% | "
                  f"{b}{x.期待値:.2f}{b} | {'' if pd.isna(x.注記) else x.注記} |")
    print("\n買い目(ルール):", g["買い目"].dropna().iloc[0])
    print(f"◎{top.馬番}の期待値が1.2〜2.0に入る単勝オッズ: {1.2 / (top['補正勝率%'] / 100):.1f}〜{2.0 / (top['補正勝率%'] / 100):.1f}倍")
    w0 = top["勝率%"] / 100
    hi = 0.50 if "補正前勝率%" in g else 0.40
    if w0 >= hi and top["人気"] != 1 and w0 < hi + 0.10:
        print(f"→ ◎{top.馬番}は1番人気でないので 単勝はやめて 馬連 {top.馬番}-{g[g['印'] == '○']['馬番'].iloc[0]}")
    if w0 >= hi and top["オッズ"] <= 1.5:
        print(f"→ ◎{top.馬番}が{top.オッズ}倍: 単勝の期待値 {top.期待値:.2f}. 保険の馬単 ○▲→◎ は足してもよい (回収率は同じ程度)")
    if fav["印"] != "◎" and fav["オッズ"] <= 1.5:
        print(f"→ 1番人気 {fav.馬番} ({fav.オッズ}倍) は◎ではない: 1着に固定しない")
    # 印なしでも消しでなければ 期待値の上位を出す (期待値1.0以上を高い順に3頭まで. 1.2〜2.0の帯は太字, 2.0超は注意書き)
    ev = g[(g["期待値"] >= 1.0) & ~g["消し"] & g["印"].isna()].sort_values("期待値", ascending=False).head(3)

    def fmt(x):
        b = "**" if 1.2 <= x.期待値 <= 2.0 else ""
        note = " (2.0超: 勝率を高く見すぎの可能性)" if x.期待値 > 2.0 else ""
        return (f"{b}{x.馬番} {x.馬名}{b} {x.人気:.0f}人気({x.オッズ}) 勝率{x['勝率%']:.0f}%/3着内{x['3着内率%']:.0f}%/"
                f"期待値{b}{x.期待値:.2f}{b}{note}")
    print("\n印以外の期待値上位 (消しを除く):", " / ".join(fmt(x) for _, x in ev.iterrows()) or "なし (期待値1.0以上の馬がいない)")
    k = g[g["消し"]]
    print("消し:", ", ".join(f"{x.馬番} {x.馬名} ({x.人気:.0f}人気)" for _, x in k.iterrows()) or "なし")


if __name__ == "__main__":
    main()
