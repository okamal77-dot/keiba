"""能力モデルに足す項目: 血統 (父・母父の条件別 A/E) と 当日の馬場傾向 (同じ日の前のレースの 前有利・内有利).
feature_trial.py (第27節) で, 2025年1月〜2026年10月の時系列検証で勝ち馬の確率の当てはまりが良くなったもの.
どちらも当該レースより前に分かる情報だけで作る.
"""
import numpy as np
import pandas as pd

import trifecta_factors as T

KEY = T.KEY
PERIODS = ["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01", "2027-01-01"]
BLOOD = ["父 距離帯A/E", "父 重不良A/E", "母父 距離帯A/E", "父 距離帯A/E×経験少"]
BIAS = ["当日 前有利×前目", "当日 内有利×内枠"]


def dist_band(x):
    return np.where(x <= 1200, "~1200", np.where(x <= 1600, "1201-1600", "1601~"))


def add_blood(h, last_start=None):
    """父・母父の条件別 3着内 A/E. 半年ごとに, その期間より前のデータで作る (同じ期間の結果は使わない).
    last_start: 予想日. その日以降の行 (予想対象) には, その日より前の全データで作った表を使う"""
    h["人気c"] = h["人気"].clip(upper=10)
    h["頭数帯"] = pd.cut(h["頭数"], [0, 8, 10, 12, 99])
    h["期待3"] = h.groupby(["人気c", "頭数帯"], observed=True)["top3"].transform("mean")
    h["距離帯"] = dist_band(h["距離"])
    h["重不良"] = h["馬場"].isin(["重", "不良"])
    starts = pd.to_datetime(["2024-01-01", "2024-07-01"] + PERIODS[:-1])
    if last_start is not None:
        starts = starts[starts < last_start].append(pd.DatetimeIndex([last_start]))
    ends = list(starts[1:]) + [pd.Timestamp("2099-01-01")]
    for c in ["父 距離帯A/E", "父 重不良A/E", "母父 距離帯A/E"]:
        h[c] = 0.0
    for a, b in zip(starts, ends):
        past = h[h["日付"] < a]
        cur = (h["日付"] >= a) & (h["日付"] < b)
        for col, sire, cond, mask in [("父 距離帯A/E", "父馬名", "距離帯", None), ("母父 距離帯A/E", "母父馬名", "距離帯", None),
                                      ("父 重不良A/E", "父馬名", None, "重不良")]:
            p = past[past[mask]] if mask else past
            keys = [sire] + ([cond] if cond else [])
            t = p.groupby(keys, observed=True).agg(o=("top3", "sum"), e=("期待3", "sum"))
            ae = np.log((t["o"] + 20) / (t["e"] + 20))
            idx = pd.MultiIndex.from_frame(h.loc[cur, keys]) if cond else h.loc[cur, sire]
            v = pd.Series(ae.reindex(idx).values, index=h.index[cur]).fillna(0)
            if mask:
                v = v.where(h.loc[cur, mask], 0)
            h.loc[cur, col] = v
    few = (h["近走数"] <= 2) | ((h["距離"] - h["平均距離"]).abs() >= 200)
    h["父 距離帯A/E×経験少"] = h["父 距離帯A/E"] * few
    return h


def add_bias(h):
    """当日・同じ競馬場の, それより前のレースの 前有利度・内有利度"""
    if "前目度" not in h:
        h["前目度"] = 1 - (h["4角位置"] - 1) / (h["頭数"] - 1).clip(lower=1)
    h["内枠度"] = 1 - (h["馬番"] - 1) / (h["頭数"] - 1).clip(lower=1)
    rk = h[KEY].drop_duplicates()
    t3 = h[h["着順"] <= 3].groupby(KEY)[["前目度", "内枠度"]].mean()
    al = h.groupby(KEY)[["前目度", "内枠度"]].mean()
    diff = (t3 - al).reset_index().rename(columns={"前目度": "前有利", "内枠度": "内有利"})
    rk = rk.merge(diff, on=KEY, how="left").sort_values(KEY)
    g = rk.groupby(["競馬場", "競走年月日"])
    for c in ["前有利", "内有利"]:
        s = g[c].transform(lambda s: s.shift(1).expanding().sum())
        n = g[c].transform(lambda s: s.shift(1).notna().astype(float).expanding().sum())
        rk["当日" + c] = (s / n.clip(lower=1) * n / (n + 3)).fillna(0)
    h = h.merge(rk[KEY + ["当日前有利", "当日内有利"]], on=KEY, how="left")
    my_front = 1 - h["位置取り"].fillna(0.5)
    h["当日 前有利×前目"] = h["当日前有利"] * (my_front - 0.5)
    h["当日 内有利×内枠"] = h["当日内有利"] * (h["内枠度"] - 0.5)
    return h
