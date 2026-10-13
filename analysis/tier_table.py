"""地区の格付け(1部〜4部)から, 移籍元→移籍先の 格上/同格/格下 表を作る.
transfer_flow.csv (class_transfer.py の出力) があれば, 実績の人気比も併記する."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "out"
TIER = {"大井": 1, "船橋": 1, "浦和": 1, "川崎": 1, "兵庫": 2, "高知": 2,
        "愛知": 3, "岩手": 3, "北海道": 3, "金沢": 4, "佐賀": 4, "笠松": 4}
NAME = {"兵庫": "兵庫(園田・姫路)", "愛知": "愛知(名古屋)", "岩手": "岩手(盛岡・水沢)", "北海道": "北海道(門別)"}
REG = list(TIER)


def rel(src, dst):
    d = TIER[src] - TIER[dst]  # 正 = 移籍先が格上
    if d == 0:
        return "＝"
    return ("↑" if d > 0 else "↓") * min(abs(d), 2)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    flow = pd.read_csv(OUT / "transfer_flow.csv", encoding="utf-8-sig")
    flow = flow.set_index(["元", "調教師所属"])["人気比_複勝率"]
    rows, mat = [], []
    for r in REG:
        up = [x for x in REG if TIER[x] < TIER[r]]
        same = [x for x in REG if TIER[x] == TIER[r] and x != r]
        down = [x for x in REG if TIER[x] > TIER[r]]
        rows.append({"地区": NAME.get(r, r), "格": f"{TIER[r]}部", "格上": "・".join(up) or "―",
                     "ほぼ同格": "・".join(same) or "―", "格下": "・".join(down) or "―"})
        mat.append({"移籍元＼移籍先": r, **{d: ("" if d == r else rel(r, d) + (
            f" {flow[(r, d)]:.2f}" if (r, d) in flow.index else "")) for d in REG}})
    t = pd.DataFrame(rows)
    m = pd.DataFrame(mat)
    t.to_csv(OUT / "tier_list.csv", index=False, encoding="utf-8-sig")
    m.to_csv(OUT / "tier_matrix.csv", index=False, encoding="utf-8-sig")
    print(md(t))
    print()
    print(md(m))


if __name__ == "__main__":
    main()
