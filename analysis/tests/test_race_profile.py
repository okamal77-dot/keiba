"""race_profile.py のテスト (python3 analysis/tests/test_race_profile.py で実行). data/ は不要."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import predict_race as P  # noqa: E402
import race_profile as RP  # noqa: E402


def test_race_class():
    assert RP.race_class("汝、星のごとく賞Ｂ２三選抜特別") == "B"
    assert RP.race_class("Ｃ３　４歳以上") == "C"
    assert RP.race_class("２歳新馬") == "2歳"
    assert RP.race_class("3歳以上C1") == "C"
    assert RP.race_class("黒潮スプリンターズカップ") is None


def test_profile():
    # 多頭数の大井・不良は荒, 少頭数の2歳戦は堅
    assert RP.profile("大井", 11, 1600, "不良", 14, "B2特別", "特別")[2] == "荒"
    assert RP.profile("大井", 3, 1200, "良", 8, "２歳", "普通", fav_odds=1.4)[2] == "堅"
    # 高知は 9R以降のほうが荒れる
    assert RP.profile("高知", 10, 1400, "重", 11, "C1", "普通")[0] > RP.profile("高知", 3, 1400, "重", 11, "C1", "普通")[0]
    assert RP.profile("JRA東京", 1, 1600, "良", 16, "", "普通") is None


def addon_for(course, rno, n, name):
    """predict_v2 と同じ流れで addon_text を呼ぶ (勝率は人気順に並べた架空の値)."""
    w = 0.7 ** np.arange(n)
    w = w / w.sum()                     # ◎の勝率 30〜33% (2頭軸の帯)
    _, tri = P.pl_probs(w)
    order = list((-w).argsort())
    nos = np.arange(1, n + 1)
    kind, _, _, _ = P.suggest(w, tri, nos, order)
    pm, t4, v, _ = RP.profile(course, rno, 1400, "良", n, name, "普通")
    pops = np.arange(1, n + 1)
    kesi = np.zeros(n, dtype=bool)
    kesi[-1] = True
    jockeys = np.array([""] * 5 + ["本田重"] + [""] * (n - 6))
    styles = np.array(["先行"] * n)
    text, combos = RP.addon_text(course, v, kind, order, nos, pops, kesi, w, jockeys, styles, tri)
    return kind, v, text, combos


def test_addon_rough():
    kind, v, text, combos = addon_for("大井", 11, 14, "C2")
    assert len(combos) == 12 and (1, 2, 6) in combos and all(c[2] in (6, 7) for c in combos)
    assert (kind, v) == ("2頭軸", "荒")
    assert text.startswith("荒判定の3着押さえ: ３連単 1,2 → 1,2,3,4 → ")
    assert "6: 穴騎手本田重" in text      # 穴騎手を優先して選ぶ
    assert "14" not in text.split("→")[-1].split("(")[0]   # 消しの馬は入れない


def test_addon_solid_low_payout():
    kind, v, text, combos = addon_for("金沢", 2, 7, "C2")
    assert len(combos) == 9 and all(c[0] == 1 for c in combos)
    assert (kind, v) == ("2頭軸", "堅")
    assert text.startswith("配当の安い場の堅いレース: ３連単は◎1着固定 1 → 1,2,3,4 → 1,2,3,4,5 (9点")


if __name__ == "__main__":
    test_race_class()
    test_profile()
    test_addon_rough()
    test_addon_solid_low_payout()
    print("ok")
