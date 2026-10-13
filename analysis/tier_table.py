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

    # 回収率: 移籍後(1-3戦) と 移籍前(元地区の最後3戦)からの変化
    roi = pd.read_csv(OUT / "transfer_flow_roi.csv", encoding="utf-8-sig").set_index(["元", "先"])
    for kind in ("単勝", "複勝"):
        rows = []
        for r in REG:
            row = {"移籍元＼移籍先": r}
            for d in REG:
                if d == r or (r, d) not in roi.index:
                    row[d] = ""
                else:
                    x = roi.loc[(r, d)]
                    row[d] = f"{x[f'移籍後_{kind}回収率']:.0f}({x[f'{kind}回収率_変化']:+.0f})"
            rows.append(row)
        mm = pd.DataFrame(rows)
        mm.to_csv(OUT / f"tier_matrix_{kind}roi.csv", index=False, encoding="utf-8-sig")
        print(f"\n#{kind}")
        print(md(mm))

    # 格の差(段数)ごとの集計
    rr = roi.reset_index()
    rr = rr[rr["元"].isin(TIER) & rr["先"].isin(TIER)]
    rr["差"] = rr["元"].map(TIER) - rr["先"].map(TIER)
    lab = {2: "2段以上格上へ", 1: "1段格上へ", 0: "同格", -1: "1段格下へ", -2: "2段以上格下へ"}
    rr["方向"] = rr["差"].clip(-2, 2).map(lab)
    out = []
    for k in [2, 1, 0, -1, -2]:
        x = rr[rr["方向"] == lab[k]]
        w0, w1 = x["移籍前_出走数"], x["移籍後_出走数"]
        def wm(c, w):
            return (x[c] * w).sum() / w.sum()
        out.append({"方向": lab[k], "組数": len(x), "移籍後出走数": int(w1.sum()),
                    "移籍前_単勝": round(wm("移籍前_単勝回収率", w0)), "移籍後_単勝": round(wm("移籍後_単勝回収率", w1)),
                    "単勝変化": f"{wm('移籍後_単勝回収率', w1) - wm('移籍前_単勝回収率', w0):+.0f}",
                    "移籍前_複勝": round(wm("移籍前_複勝回収率", w0)), "移籍後_複勝": round(wm("移籍後_複勝回収率", w1)),
                    "複勝変化": f"{wm('移籍後_複勝回収率', w1) - wm('移籍前_複勝回収率', w0):+.0f}",
                    "移籍後_人気比": round(wm("移籍後_人気比_複勝率", w1), 2)})
    o = pd.DataFrame(out)
    o.to_csv(OUT / "tier_direction_roi.csv", index=False, encoding="utf-8-sig")
    print("\n#方向")
    print(md(o))


if __name__ == "__main__":
    main()
