"""前日の暫定予想をまとめる: 能力モデルの印 (predict_v2.py の出力) + 荒れ判定 + 想定人気で組んだ暫定の ３連単・馬単 + 騎手メモ.

使い方: python3 analysis/provisional_card.py <日付YYYYMMDD>
  入力: analysis/out/pred2_<日付>_<競馬場>.csv (先に predict_v2.py を実行), analysis/out/jk_table.csv (jockey_adjust.py)
  出力: analysis/out/provisional_<日付>.md
暫定の ３連単・馬単 は「想定人気」(能力モデルと同じ項目で人気の順番を予測したもの) の順で組む.
オッズの形 (第8章) は当日のオッズでしか判定できないので, 3強・4強 の形と 1強 の形の両方を書いておき, 当日 odds_bets.py で確定する.
"""
import sys
from pathlib import Path

import pandas as pd

import odds_bets as OB

OUT = Path(__file__).parent / "out"


def main():
    date = sys.argv[1]
    jk = OB.load_jockey()
    files = sorted(OUT.glob(f"pred2_{date}_*.csv"))
    files = [f for f in files if not f.name.endswith("_valid.csv")]
    lines = [f"# {date[:4]}年{int(date[4:6])}月{int(date[6:])}日 暫定予想", "",
             "- 印・買い目（ルール）: 能力モデル `predict_v2.py`（人気・オッズを使わない事前情報のモデル）。",
             "- 暫定 ３連単・馬単: 想定人気の順で組んだもの。**当日のオッズで人気順とオッズの形が変わるので、締切前に `odds_bets.py` で確定する**。",
             "  - 2〜4番人気が接近（3強・4強）なら ３連単 1→2,3,4→2,3,4（6点）、1番人気が抜けている（1強）なら 1→2,3→2,3（2点。A級は6点）、2強か1番人気3倍以上なら ３連単は見送り。馬単はどの形でも 1→2〜5（4点）。",
             "  - 想定1番人気の騎手が、その場の1番人気で勝ち切れない（縮小A/E 0.95未満）ときは、馬単を見送り候補にする。",
             "- どの買い方も過去の検証で回収率100%を安定して超えてはいない。", ""]
    for f in files:
        course = f.stem.split("_")[2]
        d = pd.read_csv(f, encoding="utf-8-sig")
        lines += [f"## {course}", ""]
        for rno, g in d.groupby("レース"):
            name, dist = g["レース名"].iloc[0], g["距離"].iloc[0]
            buy = g["買い目"].dropna()
            rough = g["荒れ判定"].dropna()
            marks = g[g["印"].notna()].copy()
            order = {"◎": 0, "○": 1, "▲": 2, "△": 3}
            marks = marks.assign(o=marks["印"].map(order)).sort_values(["o", "勝率%"], ascending=[True, False])
            lines.append(f"### {rno}R {name} {dist}m")
            if len(rough):
                lines.append(f"- 荒れ判定: {rough.iloc[0]}")
            lines.append("- 印: " + " / ".join(f"{r['印']}{int(r['馬番'])} {r['馬名']}（{r['騎手']}, 勝率{r['勝率%']}%）"
                                                for _, r in marks.iterrows()))
            fav = g[g["印"] == "◎"]
            if len(fav) and isinstance(fav["注記"].iloc[0], str):
                lines.append(f"- ◎の注記: {fav['注記'].iloc[0]}")
            if len(buy):
                lines.append(f"- ルールの買い目: {buy.iloc[0]}")
            p = g.sort_values("想定人気")
            nos = p["馬番"].astype(int).tolist()
            if len(nos) >= 5:
                n1, n2, n3, n4, n5 = nos[:5]
                fav_jk = p["騎手"].iloc[0]
                ae = OB.jk_ae(jk, fav_jk, course, "1番人気")
                ex = f"{n1} → {n2},{n3},{n4},{n5}（4点）"
                if ae is not None and ae < OB.FAV_WEAK:
                    ex += f" ※想定1番人気の{fav_jk}は1番人気で勝ち切れない（A/E {ae:.2f}）→ 1番人気のままなら見送り候補"
                lines.append(f"- 想定人気順: {' '.join(map(str, nos[:6]))}")
                if OB.race_class(name) == "A":   # 第8-6節: A級は1強でも6点
                    lines.append(f"- 暫定 ３連単: 3強・4強・1強とも {n1} → {n2},{n3},{n4} → {n2},{n3},{n4}（6点）※A級は1強でも6点")
                else:
                    lines.append(f"- 暫定 ３連単: 3強・4強なら {n1} → {n2},{n3},{n4} → {n2},{n3},{n4}（6点）／1強なら {n1} → {n2},{n3} → {n2},{n3}（2点）")
                lines.append(f"- 暫定 馬単: {ex}")
            ev = g[g["期待値順"].notna()].sort_values("期待値順")
            if len(ev):
                lines.append("- 期待値上位: " + " / ".join(f"{int(r['馬番'])} {r['馬名']}（期待値{r['期待値(補正)']}）" for _, r in ev.iterrows()))
            star = g[g["過小評価候補"] == "★"]
            if len(star):
                lines.append("- ★過小評価候補: " + " / ".join(f"{int(r.馬番)} {r.馬名}" for r in star.itertuples()))
            lines.append("")
    path = OUT / f"provisional_{date}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
