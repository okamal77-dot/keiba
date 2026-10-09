"""出馬表・競走成績の貼り付けテキストを読み取り, オッズの動きを記録・集計する.

地方競馬の公式サイトの「出馬表」「競走成績」ページをコピーした文章を想定している.

使い方:
  python3 analysis/odds_log.py card   <出馬表.txt>  [--at "2026-10-09 19:45"]  出馬表を読み取り, その時点のオッズを記録
  python3 analysis/odds_log.py result <競走成績.txt>                          確定オッズ・着順・払戻を記録
  python3 analysis/odds_log.py report                                         オッズの動き別の成績を集計
  python3 analysis/odds_log.py import-odds <odds.csv> [--at ...]              既存のオッズCSV (predict_with_odds 用) を出馬表時点として取り込む

記録先 (リポジトリに残す):
  analysis/odds_log/snapshots.csv  出馬表時点のオッズ (同じレースを何度記録してもよい. 記録時刻つき)
  analysis/odds_log/results.csv    確定オッズ・着順
  analysis/odds_log/payouts.csv    払戻
  analysis/odds_log/raw/           貼り付けた元の文章 (読み取り直し用)
出馬表を読み取ると, predict_with_odds.py 用の入力 data/entries/<日付>_<場>_<R>_odds.csv も作る.
"""
import argparse
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
LOG = ROOT / "odds_log"
ENTRIES = ROOT.parent / "data" / "entries"
KEY = ["競馬場", "競走年月日", "レース番号"]
SNAP_COLS = KEY + ["記録時刻", "発走時刻", "レース名", "距離", "天候", "馬場", "枠番", "馬番", "馬名", "騎手",
                   "単勝オッズ", "人気", "馬体重", "馬体重増減"]
RES_COLS = KEY + ["着順", "枠番", "馬番", "馬名", "騎手", "馬体重", "馬体重増減", "人気", "確定単勝オッズ"]
PAY_COLS = KEY + ["券種", "組番", "払戻", "人気"]


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def to_num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def parse_header(lines):
    """1行目 '2026年10月9日（金）　大　井　第9競走　20:10発走' などからレース情報"""
    info = {}
    for ln in lines[:15]:
        t = nfkc(ln)
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?\)\s*(.+?)\s*第(\d+)競走", t)
        if m and "競走年月日" not in info:
            info["競走年月日"] = int(f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}")
            info["競馬場"] = re.sub(r"\s", "", m.group(4))
            info["レース番号"] = int(m.group(5))
            s = re.search(r"(\d{1,2}:\d{2})発走", t)
            info["発走時刻"] = s.group(1) if s else ""
        m = re.search(r"(ダート|芝)\s*(\d+)m", t)
        if m and "距離" not in info:
            info["距離"] = int(m.group(2))
            w = re.search(r"天候:(\S+)", t)
            b = re.search(r"馬場:(\S+)", t)
            info["天候"] = w.group(1) if w else ""
            info["馬場"] = b.group(1) if b else ""
    if "競走年月日" not in info:
        raise ValueError("1行目の『YYYY年M月D日（曜）　場名　第N競走』が見つかりません")
    # レース名: 日付行より後で, 距離行より前の最後の空でない行
    names = []
    for ln in lines[1:15]:
        t = ln.strip()
        if not t:
            continue
        if re.search(r"(ダート|芝)", nfkc(t)):
            break
        names.append(t)
    info["レース名"] = names[-1] if names else ""
    return info


def parse_card(text, at=None):
    """出馬表の貼り付けテキスト → 馬ごとの行"""
    lines = text.splitlines()
    info = parse_header(lines)
    rows, cur, waku = [], None, None
    start2 = re.compile(r"^(\d+)\t(\d+)\t([^\t\d][^\t]*)\t([^\t]+?)（[^）]*）\t?([\d.]*|取消|除外|-+)?\s*$")
    start1 = re.compile(r"^(\d+)\t([^\t\d][^\t]*)\t([^\t]+?)（[^）]*）\t?([\d.]*|取消|除外|-+)?\s*$")
    pop_re = re.compile(r"^\((\d+)人気\)")
    wt_re = re.compile(r"^[^\t]+\t[^\t]+（[^）]*）\t(\d{3}|計不)\s*$")
    wd_re = re.compile(r"^\(([+\-]?\d+)\)")
    for ln in lines:
        ln = ln.rstrip("\n")
        m = start2.match(ln)
        if m:
            waku = int(m.group(1))
            cur = {"枠番": waku, "馬番": int(m.group(2)), "馬名": m.group(3).strip(), "騎手": m.group(4).strip(),
                   "単勝オッズ": to_num(m.group(5) or "")}
            rows.append(cur)
            continue
        m = start1.match(ln)
        if m and cur is not None:
            cur = {"枠番": waku, "馬番": int(m.group(1)), "馬名": m.group(2).strip(), "騎手": m.group(3).strip(),
                   "単勝オッズ": to_num(m.group(4) or "")}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = pop_re.match(ln)
        if m and "人気" not in cur:
            cur["人気"] = int(m.group(1))
            continue
        m = wt_re.match(ln)
        if m and "馬体重" not in cur:
            cur["馬体重"] = to_num(m.group(1))
            cur["_待ち"] = True
            continue
        m = wd_re.match(ln)
        if m and cur.get("_待ち"):
            cur["馬体重増減"] = int(m.group(1))
            cur.pop("_待ち")
    if not rows:
        raise ValueError("馬の行が見つかりません (出馬表の貼り付けを確認してください)")
    df = pd.DataFrame(rows).drop(columns=["_待ち"], errors="ignore")
    for k, v in info.items():
        df[k] = v
    df["記録時刻"] = at or datetime.now().strftime("%Y-%m-%d %H:%M")
    return df.reindex(columns=SNAP_COLS)


def parse_result(text):
    """競走成績の貼り付けテキスト → (着順の行, 払戻の行)"""
    lines = text.splitlines()
    info = parse_header(lines)
    rows, pays = [], []
    wt = re.compile(r"(\d{3})\(([+\-]?\d+)\)")
    label = None
    pay_kinds = {"単勝", "複勝", "枠連複", "枠連単", "馬連複", "馬連単", "ワイド", "三連複", "三連単"}
    for ln in lines:
        f = ln.split("\t")
        if len(f) >= 10 and re.fullmatch(r"\d+|取消|除外|中止", f[0].strip()) and f[1].strip().isdigit() and f[2].strip().isdigit():
            last = [x.strip() for x in f if x.strip() != ""]
            w = wt.search(ln)
            rows.append({"着順": to_num(f[0]), "枠番": int(f[1]), "馬番": int(f[2]), "馬名": f[3].strip(),
                         "騎手": re.sub(r"\s*（.*", "", f[7]).strip() if len(f) > 7 else "",
                         "馬体重": int(w.group(1)) if w else None, "馬体重増減": int(w.group(2)) if w else None,
                         "人気": to_num(last[-2]), "確定単勝オッズ": to_num(last[-1])})
            continue
        t = [x.strip() for x in ln.split("\t") if x.strip()]
        if t and t[0] in pay_kinds:
            label, t = t[0], t[1:]
        elif t and label and re.fullmatch(r"[\d\-]+", t[0]) and len(t) >= 2 and t[1].endswith("円"):
            pass
        else:
            if t and t[0] not in pay_kinds and not re.fullmatch(r"[\d\-]+", t[0] if t else ""):
                label = None
            continue
        if len(t) >= 2 and t[1].endswith("円"):
            pays.append({"券種": label, "組番": t[0], "払戻": to_num(t[1].rstrip("円")),
                         "人気": to_num(t[2].rstrip("人気")) if len(t) > 2 else None})
    if not rows:
        raise ValueError("着順の行が見つかりません (競走成績の貼り付けを確認してください)")
    r = pd.DataFrame(rows)
    p = pd.DataFrame(pays, columns=["券種", "組番", "払戻", "人気"])
    for k in KEY:
        r[k] = info[k]
        p[k] = info[k]
    return r.reindex(columns=RES_COLS), p.reindex(columns=PAY_COLS)


def append(path, df, cols, dedup):
    LOG.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(dedup, keep="last").reindex(columns=cols)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def save_raw(src, kind, info):
    raw = LOG / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / f"{info['競走年月日']}_{info['競馬場']}_{info['レース番号']:02d}R_{kind}_{datetime.now():%H%M%S}.txt"
    shutil.copy(src, dst)


def write_entry_odds(df):
    ENTRIES.mkdir(parents=True, exist_ok=True)
    k = df.iloc[0]
    out = ENTRIES / f"{k['競走年月日']}_{k['競馬場']}_{int(k['レース番号'])}R_odds.csv"
    df[KEY + ["馬番", "単勝オッズ", "人気", "馬体重", "馬体重増減"]].to_csv(out, index=False, encoding="utf-8-sig")
    return out


def cmd_card(a):
    text = Path(a.file).read_text(encoding="utf-8")
    df = parse_card(text, a.at)
    append(LOG / "snapshots.csv", df, SNAP_COLS, KEY + ["記録時刻", "馬番"])
    save_raw(a.file, "card", df.iloc[0])
    out = write_entry_odds(df)
    k = df.iloc[0]
    print(f"{k['競馬場']} {k['レース番号']}R {k['レース名']} {k['距離']}m  記録時刻 {k['記録時刻']}  {len(df)}頭")
    print(df[["枠番", "馬番", "馬名", "騎手", "単勝オッズ", "人気", "馬体重", "馬体重増減"]].to_string(index=False))
    miss = df[df[["単勝オッズ", "人気"]].isna().any(axis=1)]
    if len(miss):
        print("注意: オッズ・人気が読み取れなかった馬:", miss["馬番"].tolist())
    print(f"予想用の入力: {out}")


def cmd_result(a):
    text = Path(a.file).read_text(encoding="utf-8")
    r, p = parse_result(text)
    append(LOG / "results.csv", r, RES_COLS, KEY + ["馬番"])
    append(LOG / "payouts.csv", p, PAY_COLS, KEY + ["券種", "組番"])
    save_raw(a.file, "result", r.iloc[0])
    k = r.iloc[0]
    print(f"{k['競馬場']} {k['レース番号']}R  {len(r)}頭, 払戻 {len(p)}件")
    print(r[["着順", "馬番", "馬名", "人気", "確定単勝オッズ", "馬体重", "馬体重増減"]].to_string(index=False))
    s = LOG / "snapshots.csv"
    if s.exists():
        sn = pd.read_csv(s, encoding="utf-8-sig")
        sn = sn.merge(r[KEY].drop_duplicates(), on=KEY)
        if len(sn):
            first = sn.sort_values("記録時刻").groupby("馬番").first()
            m = r.set_index("馬番").join(first[["単勝オッズ", "記録時刻"]])
            m["変化率"] = m["確定単勝オッズ"] / m["単勝オッズ"]
            print("\nオッズの動き (最初の記録 → 確定):")
            print(m.reset_index()[["着順", "馬番", "馬名", "単勝オッズ", "確定単勝オッズ", "変化率"]].round(2).to_string(index=False))


def cmd_import(a):
    o = pd.read_csv(a.file, encoding="utf-8-sig")
    o["記録時刻"] = a.at or "不明"
    append(LOG / "snapshots.csv", o.reindex(columns=SNAP_COLS), SNAP_COLS, KEY + ["記録時刻", "馬番"])
    print(f"{len(o)}頭を取り込みました ({a.file})")


BANDS = [0, 0.7, 0.9, 1.1, 1.5, 999]
BLAB = ["大きく買われた(~0.7倍)", "買われた(0.7-0.9倍)", "ほぼ変化なし(0.9-1.1倍)", "売られた(1.1-1.5倍)", "大きく売られた(1.5倍~)"]


def cmd_report(a):
    s = pd.read_csv(LOG / "snapshots.csv", encoding="utf-8-sig")
    r = pd.read_csv(LOG / "results.csv", encoding="utf-8-sig")
    first = s.sort_values("記録時刻").groupby(KEY + ["馬番"], as_index=False).first()
    m = r.merge(first[KEY + ["馬番", "単勝オッズ", "記録時刻"]], on=KEY + ["馬番"], how="inner")
    m = m[m["単勝オッズ"].notna() & m["確定単勝オッズ"].notna() & m["着順"].notna()]
    if m.empty:
        print("出馬表時点と結果の両方がそろったレースがまだありません")
        return
    m["変化率"] = m["確定単勝オッズ"] / m["単勝オッズ"]
    q = 1 / m["確定単勝オッズ"]
    m["確定オッズの勝率"] = q / q.groupby([m[k] for k in KEY]).transform("sum")
    m["動き"] = pd.cut(m["変化率"], BANDS, labels=BLAB)
    m["人気帯"] = pd.cut(m["人気"], [0, 3, 6, 99], labels=["1-3番人気", "4-6番人気", "7番人気~"])
    g = m.groupby("動き", observed=True)
    t = pd.DataFrame({"頭数": g.size(), "勝率%": g["着順"].apply(lambda x: (x == 1).mean() * 100),
                      "確定オッズから見た勝率%": g["確定オッズの勝率"].mean() * 100,
                      "3着内率%": g["着順"].apply(lambda x: (x <= 3).mean() * 100),
                      "単勝回収率%": g.apply(lambda x: (x["確定単勝オッズ"] * (x["着順"] == 1)).sum() / len(x) * 100)})
    n_r = m.groupby(KEY).ngroups
    print(f"== オッズの動き別の成績 ({n_r}レース, {len(m)}頭)")
    print(t.round(1).to_string())
    t2 = m.pivot_table(index="人気帯", columns="動き", values="着順", aggfunc=lambda x: (x <= 3).mean() * 100, observed=False)
    print("\n3着内率% (人気帯 × 動き)\n", t2.round(1).to_string())
    t.to_csv(LOG / "report_movement.csv", encoding="utf-8-sig")
    m.to_csv(LOG / "report_rows.csv", index=False, encoding="utf-8-sig")
    if n_r < 100:
        print(f"\n※ まだ {n_r} レースです. 100レース程度たまるまでは偶然の差が大きいので参考値です")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("card", cmd_card), ("result", cmd_result), ("import-odds", cmd_import)]:
        p = sub.add_parser(name)
        p.add_argument("file")
        p.add_argument("--at", help="記録時刻 (例: '2026-10-09 19:45'). 省略時は今の時刻")
        p.set_defaults(fn=fn)
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
