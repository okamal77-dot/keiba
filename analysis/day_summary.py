"""1日分の予想 (predict_v2.py の出力) を, レースごとに ◎○▲ と ★, 期待値 (暫定) の表にまとめる.

使い方: python3 analysis/day_summary.py <日付YYYYMMDD>
  入力: analysis/out/pred2_<日付>_<競馬場>.csv, data/entries/<日付>_racelist.csv (発走時刻)
  出力: 標準出力と analysis/out/summary_<日付>.md
期待値 (暫定) = 補正勝率 × 補正オッズ. 補正オッズは想定人気 (オッズ前にモデルで予測した人気順) から見込んだ単勝オッズで,
  当日の実際のオッズではない (ev_calibrate.py で人気薄ほど出すぎる分を補正済み). 1.0 以上が割安の目安.
  交差検証では 1.2〜2.0 の帯が最も良く (単勝回収率 92〜104%), 2.0 超は補正後も出すぎ (71%).
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
DATA = Path(__file__).parent.parent / "data"
ORDER = ["門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀"]


def ev(x):
    v = x["期待値(補正)"]
    return "-" if pd.isna(v) else f"{v:.2f}"


def main():
    date = sys.argv[1]
    rl = DATA / "entries" / f"{date}_racelist.csv"
    times = {}
    if rl.exists():
        r = pd.read_csv(rl, encoding="utf-8-sig")
        times = {(c, n): f"{int(t) // 100}:{int(t) % 100:02d}" for c, n, t in zip(r["競馬場"], r["レース番号"], r["発走時刻"])}
    files = [f for f in OUT.glob(f"pred2_{date}_*.csv") if not f.name.endswith("_valid.csv")]
    files.sort(key=lambda f: ORDER.index(f.stem.split("_")[2]) if f.stem.split("_")[2] in ORDER else 99)
    lines = [f"# {date[:4]}年{int(date[4:6])}月{int(date[6:])}日 予想 (◎○▲・★・期待値)", "",
             "期待値(暫定) = 補正勝率 × 想定オッズ (当日のオッズではない). 1.0以上が割安の目安, 1.2〜2.0 が検証で最も良い帯.", ""]
    for f in files:
        course = f.stem.split("_")[2]
        d = pd.read_csv(f, encoding="utf-8-sig")
        lines += [f"## {course}", "",
                  "| R | 発走 | レース | ◎ (勝率・期待値) | ○ (勝率・期待値) | ▲ (勝率・期待値) | ★ (期待値) | 期待値上位 | 荒れ |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k, g in d.groupby("レース"):
            cell = {}
            for m in ["◎", "○", "▲"]:
                x = g[g["印"] == m]
                cell[m] = " ".join(f"{int(r['馬番'])} {r['馬名']} ({r['勝率%']:.0f}%・{ev(r)})" for _, r in x.iterrows()) or "-"
            star = " / ".join(f"{int(r['馬番'])} {r['馬名']} ({ev(r)})" for _, r in g[g["過小評価候補"] == "★"].iterrows()) or "-"
            top = g[g["期待値順"].notna()].sort_values("期待値順")
            top = " / ".join(f"{int(r['馬番'])} ({ev(r)})" for _, r in top.iterrows()) or "-"
            rough = g["荒れ判定"].dropna()
            rough = rough.iloc[0][1] if len(rough) else ""
            lines.append(f"| {k} | {times.get((course, k), '')} | {g['レース名'].iloc[0]} {g['距離'].iloc[0]}m | {cell['◎']} | "
                         f"{cell['○']} | {cell['▲']} | {star} | {top} | {rough} |")
        lines.append("")
    path = OUT / f"summary_{date}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
