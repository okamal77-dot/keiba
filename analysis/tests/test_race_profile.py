"""race_profile.py のテスト (python3 analysis/tests/test_race_profile.py で実行). data/ は不要."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
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


def test_ana_addon():
    # 検証用の穴選び: 荒判定だけ, 消し・上位5頭・5番人気以内は選ばず, 穴騎手を優先する
    n = 14
    nos = list(range(1, n + 1))
    kesi = [False] * (n - 1) + [True]
    jockeys = [""] * 5 + ["本田重"] + [""] * (n - 6)
    strength = 0.7 ** np.arange(n)
    ana = RP.ana_addon("大井", "荒", list(range(n)), nos, nos, kesi, strength, jockeys, ["先行"] * n)
    assert [i for i, _ in ana] == [5, 6] and "穴騎手本田重" in ana[0][1][0]
    assert RP.ana_addon("大井", "中", list(range(n)), nos, nos, kesi, strength, jockeys, ["先行"] * n) == []


if __name__ == "__main__":
    test_race_class()
    test_profile()
    test_ana_addon()
    print("ok")
