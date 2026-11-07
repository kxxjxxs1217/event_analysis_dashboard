"""리텐션 계산 (dashboard.py가 사용). 일 단위 정의는 12_retention_curve.py, 13_retention_by_action.py와 같다.

- 활동일 = UTC 날짜. 2015-09-15는 21:00 UTC 전까지만 있어 하루가 온전하지 않으므로 마지막 관측일은 2015-09-14
- Day 0 = 기준 행동을 처음 한 날짜
- 30일 이내 재방문율 = Day 1~30 중 하루라도 재방문한 사용자 ÷ Day 0 + 30 <= 마지막 관측일인 사용자
- 코호트 리텐션 (단위 = 일 / 주 / 월)
  - 코호트 = Day 0이 속한 기간. 주는 월요일 시작, 월은 달력 월
  - 기간 N 재방문 = 코호트 기간 + N번째 기간에 재방문 행동이 1번이라도 있음. 재방문율 = 재방문자 ÷ 코호트 사용자 (누적 아님)
  - 관측 가능한 칸 = 코호트 기간 + N이 마지막 완전한 기간 이하. 마지막 완전한 기간은
    일 2015-09-14, 주 2015-09-07(월)~09-13(일), 월 2015-08
  - 가장 앞선 4주(2015-05-31 전)를 피하려고 시작일이 2015-05-31 이후인 코호트 기간만 쓴다
    (일 2015-05-31부터, 주 2015-06-01 주부터, 월 2015-06부터)
"""
import numpy as np
import pandas as pd

LAST_FULL_DAY = pd.Timestamp("2015-09-14")
FIRST_4W_END = pd.Timestamp("2015-05-31")
MAX_N = 30
EPOCH = pd.Timestamp("2015-01-05")  # 월요일. 일·주 번호의 기준
UNITS = {"일별": "D", "주별": "W", "월별": "M"}
UNIT_LABEL = {"D": "Day", "W": "Week", "M": "Month"}


def returns(d0, days):
    """d0: visitorid별 Day 0 날짜. days: 재방문 행동이 있는 (visitorid, date), 중복 없음.
    반환: Day 1~MAX_N 재방문 기록 (visitorid, n). 마지막 관측일 이후 활동은 뺀다."""
    a = days[days["date"] <= LAST_FULL_DAY].join(d0.rename("d0"), on="visitorid", how="inner")
    n = (a["date"] - a["d0"]).dt.days
    keep = (n >= 1) & (n <= MAX_N)
    return pd.DataFrame({"visitorid": a.loc[keep, "visitorid"].values, "n": n[keep].astype("int16").values})


def within(d0, r):
    """visitorid별 30일 이내 재방문 여부(1.0/0.0). 30일을 관측할 수 없으면 NaN."""
    hit = d0.index.isin(r["visitorid"].unique()).astype(float)
    observable = (d0 <= LAST_FULL_DAY - pd.Timedelta(days=MAX_N)).values
    return pd.Series(np.where(observable, hit, np.nan), index=d0.index)


def period(dates, unit):
    """날짜 Series → 기간 번호. 두 번호의 차이가 N이다."""
    if unit == "M":
        return dates.dt.year * 12 + dates.dt.month - 1
    days = (dates - EPOCH).dt.days
    return days // 7 if unit == "W" else days


def period_start(p, unit):
    if unit == "M":
        return pd.Timestamp(year=int(p) // 12, month=int(p) % 12 + 1, day=1)
    return EPOCH + pd.Timedelta(days=int(p) * (7 if unit == "W" else 1))


def _period_of(ts, unit):
    return int(period(pd.Series([ts]), unit).iloc[0])


def last_full_period(unit):
    return _period_of(LAST_FULL_DAY + pd.Timedelta(days=1), unit) - 1


def first_cohort_period(unit):
    return _period_of(FIRST_4W_END - pd.Timedelta(days=1), unit) + 1


def cohort_matrix(d0, days, unit):
    """d0: visitorid별 Day 0 날짜, days: 재방문 행동이 있는 (visitorid, date).
    반환 (sizes, counts): sizes = 코호트 기간 번호별 사용자 수,
    counts = 코호트 × N 재방문자 수 (관측할 수 없는 칸은 NaN)."""
    last = last_full_period(unit)
    c = period(d0, unit).rename("c")
    c = c[(c >= first_cohort_period(unit)) & (c <= last)]
    sizes = c.value_counts().sort_index()
    if sizes.empty:
        return sizes, pd.DataFrame()
    a = days[days["date"] <= LAST_FULL_DAY].join(c, on="visitorid", how="inner")
    a["n"] = period(a["date"], unit) - a["c"]
    a = a[a["n"] >= 0].drop_duplicates(["visitorid", "n"])
    counts = (a.groupby(["c", "n"]).size().unstack(fill_value=0)
              .reindex(index=sizes.index, columns=range(last - sizes.index.min() + 1), fill_value=0))
    observable = sizes.index.values[:, None] + counts.columns.values[None, :] <= last
    return sizes, counts.where(observable)


def curve(sizes, counts, n_max=None):
    """코호트 표를 합친 N = 1 ~ n_max 재방문율. 각 N은 그 칸을 관측할 수 있는 코호트만 더한다."""
    if sizes.empty:
        return pd.DataFrame(columns=["N", "대상 사용자 수", "재방문자 수", "재방문율(%)"])
    eligible = counts.notna().mul(sizes, axis=0).sum()
    back = counts.fillna(0).sum()
    t = pd.DataFrame({"N": counts.columns, "대상 사용자 수": eligible.values.astype(int),
                      "재방문자 수": back.values.astype(int)})
    t["재방문율(%)"] = (t["재방문자 수"] / t["대상 사용자 수"] * 100).where(t["대상 사용자 수"] > 0)
    return t[(t["N"] >= 1) & (t["N"] <= (n_max or t["N"].max()))].reset_index(drop=True)
