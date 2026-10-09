"""odds_log.py の読み取りテスト (python3 analysis/tests/test_odds_log.py で実行)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import odds_log as L  # noqa: E402


def test_card():
    c = L.parse_card((HERE / "fixtures/card_oi9_partial.txt").read_text(encoding="utf-8"), "2026-10-09 19:30")
    assert list(c["馬番"]) == [1, 6, 7, 8]
    assert list(c["枠番"]) == [1, 5, 6, 6]  # 枠番のない行は直前の枠を引き継ぐ
    assert list(c["単勝オッズ"]) == [6.5, 29.2, 4.6, 4.0]
    assert list(c["人気"]) == [5, 9, 3, 1]
    assert list(c["馬体重"]) == [513, 449, 490, 510]
    assert list(c["馬体重増減"]) == [-2, -2, 4, 1]
    r = c.iloc[0]
    assert (r["競馬場"], r["競走年月日"], r["レース番号"], r["距離"], r["発走時刻"]) == ("大井", 20261009, 9, 1400, "20:10")
    assert r["レース名"] == "汝、星のごとく賞Ｂ２三選抜特別"
    assert (r["天候"], r["馬場"]) == ("晴", "良")


def test_result():
    r, p = L.parse_result((HERE / "fixtures/result_oi9.txt").read_text(encoding="utf-8"))
    assert len(r) == 12
    top = r.sort_values("着順").head(3)
    assert list(top["馬番"]) == [11, 8, 9]
    assert list(top["確定単勝オッズ"]) == [22.3, 2.7, 5.7]
    assert list(top["人気"]) == [7, 1, 4]
    assert r.set_index("馬番").loc[8, "馬体重増減"] == 1
    t = p.set_index(["券種", "組番"])["払戻"]
    assert t[("三連単", "11-8-9")] == 42460
    assert t[("複勝", "9")] == 170
    assert len(p[p["券種"] == "ワイド"]) == 3


if __name__ == "__main__":
    test_card()
    test_result()
    print("OK")
