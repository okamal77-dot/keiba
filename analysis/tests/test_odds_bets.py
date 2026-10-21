"""odds_bets.py のテスト (python3 analysis/tests/test_odds_bets.py で実行). data/ は不要."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import odds_bets as OB  # noqa: E402

NOS = [7, 12, 4, 11, 8, 9]


def test_shapes():
    one = OB.bets(NOS, [2.0, 5.0, 6.6, 7.8, 11.1, 18.1])            # 10/19 名古屋10R: 1強
    assert one["形"] == "1強" and one["点数3"] == 2 and one["３連単"].startswith("7 → 12,4 → 12,4")
    assert one["馬単"].startswith("7 → 12,4,11,8")
    four = OB.bets(NOS, [2.1, 3.5, 6.4, 7.5, 16.3, 20.0])           # 4強
    assert four["形"] == "4強" and four["点数3"] == 6
    two = OB.bets(NOS, [1.9, 3.3, 7.9, 12.5, 12.8, 15.8])           # 2強
    assert two["点数3"] == 0
    assert OB.bets(NOS, [3.4, 3.7, 5.3, 5.9, 6.9, 9.0])["点数3"] == 0  # 1番人気3倍以上


def test_class_a():
    a = OB.bets(NOS, [2.0, 5.0, 6.6, 7.8, 11.1, 18.1], cls="A")     # A級の1強は6点
    assert a["点数3"] == 6 and a["３連単"].startswith("7 → 12,4,11 → 12,4,11") and "A級" in a["注記"]
    assert OB.bets(NOS, [2.0, 5.0, 6.6, 7.8, 11.1, 18.1], cls="2歳")["点数3"] == 2
    assert OB.bets(NOS, [1.9, 3.3, 7.9, 12.5, 12.8, 15.8], cls="A")["点数3"] == 0   # 2強は見送りのまま
    assert OB.race_class("楽天ポイントが貯まる！楽天競馬特別Ａ１二") == "A"
    assert OB.race_class("カエデ特別Ａ４～Ｂ２") == "A"


def test_weak_fav_jockey():
    b = OB.bets(NOS, [2.0, 5.0, 6.6, 7.8, 11.1, 18.1], fav_ae=0.80)
    assert b["点数2"] == 0 and b["馬単"].startswith("見送り")


def test_allocate_budget_and_shape():
    nos, odds = [1, 2, 3, 4, 5, 6], [1.9, 2.9, 4.9, 9.8, 10.1, 20.0]   # 3強
    b = OB.bets(nos, odds)
    al = OB.allocate(nos, odds, b)
    assert sum(y for _, _, y in al) == 1000
    assert sum(y for k, _, y in al if k == "３連単") == 500
    assert all(y % 100 == 0 for _, _, y in al)
    ex = {t: y for k, t, y in al if k == "馬単"}
    assert ex[(1, 5)] >= ex[(1, 2)]          # 当たりにくい組ほど厚い


def test_allocate_one_strong_goes_to_exacta():
    nos, odds = [1, 2, 3, 4, 5], [1.5, 6.0, 7.0, 9.0, 12.0]          # 1強
    al = OB.allocate(nos, odds, OB.bets(nos, odds))
    assert {k for k, _, _ in al} == {"馬単"} and sum(y for _, _, y in al) == 1000


if __name__ == "__main__":
    test_shapes()
    test_class_a()
    test_weak_fav_jockey()
    test_allocate_budget_and_shape()
    test_allocate_one_strong_goes_to_exacta()
    print("ok")
