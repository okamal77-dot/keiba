"""馬ごとの「今の脚質」(近5走で最も多い脚質) の表を作る. odds_bets.py が ②-1 の条件判定に使う.

使い方: python3 analysis/horse_style.py   (data/ の成績を取り込んだ後に実行し直す)
脚質の決め方は ashi_1kyo.py と同じ (trifecta_factors.styles: 最初のコーナー1番手=逃げ, 最終コーナー35%以内=先行, 70%以内=差し, それより後ろ=追込).
「今の脚質」は最後のレースまでを含む近5走で数える (次のレースの事前脚質になる). 同数なら前の脚質.
出力: analysis/out/horse_style.csv (馬名, 生年月日, 脚質, 数えた走数, 最終出走日, 内訳)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], str(Path(__file__).parent.parent / "data")]
import trifecta_factors as T  # noqa: E402

ORDER = ["逃げ", "先行", "差し", "追込"]
OUT = Path(__file__).parent / "out" / "horse_style.csv"


def main():
    h, _, _ = T.load()
    h = h[(h["競馬場"] != "帯広ば") & h["脚質"].notna()].sort_values(["馬ID", "日付", "レース番号"])
    last5 = h.groupby("馬ID").tail(5)
    rows = []
    for mid, g in last5.groupby("馬ID"):
        cnt = [int((g["脚質"] == s).sum()) for s in ORDER]
        rows.append({"馬名": g["馬名"].iloc[-1], "生年月日": g["生年月日"].iloc[-1], "脚質": ORDER[int(np.argmax(cnt))],
                     "数えた走数": len(g), "最終出走日": int(g["競走年月日"].iloc[-1]),
                     "内訳": " ".join(f"{s}{c}" for s, c in zip(ORDER, cnt) if c)})
    t = pd.DataFrame(rows).sort_values(["馬名", "最終出走日"])
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(len(t), "頭 →", OUT)


if __name__ == "__main__":
    main()
