"""CLAUDE.md의 분석 기준을 적용한 가공 데이터 생성.

- 완전 중복 행 제거
- 기준 시점 2015-09-15 21:00:00 UTC 이후(>=) 이벤트 제외
- 시간대는 UTC 그대로 사용
- 가장 앞선 4주(UTC 2015-05-03 ~ 2015-05-30)는 제외하지 않고 is_first_4w로 표시
- UTC 기준 year, month, day, weekday(요일), day_type(평일/주말) 열 추가
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "events.csv"
OUT = ROOT / "data" / "processed" / "events_clean.csv"
LOG = ROOT / "outputs" / "tables" / "cleaning_log.csv"

CUTOFF = pd.Timestamp("2015-09-15 21:00:00", tz="UTC")
FIRST_4W_END = pd.Timestamp("2015-05-31 00:00:00", tz="UTC")

df = pd.read_csv(RAW, dtype={"transactionid": "Int64"})
log = [("원본 행 수", "", len(df))]

before = len(df)
df = df.drop_duplicates()
log.append(("완전 중복 제거", "5개 열 모두 동일한 행 중 첫 행만 유지", before - len(df)))

df["event_time_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
before = len(df)
df = df[df["event_time_utc"] < CUTOFF]
log.append(("기준 시점 이후 제외", f"event_time_utc >= {CUTOFF}", before - len(df)))

df["is_first_4w"] = df["event_time_utc"] < FIRST_4W_END

# UTC 날짜 파생 열 (timestamp 원값은 유지)
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
dt = df["event_time_utc"].dt
df["year"] = dt.year
df["month"] = dt.month
df["day"] = dt.day
df["weekday"] = dt.day_of_week.map(dict(enumerate(WEEKDAYS)))
df["day_type"] = dt.day_of_week.map(lambda d: "주말" if d >= 5 else "평일")
log.append(("가장 앞선 4주 (포함, 표시만)", f"event_time_utc < {FIRST_4W_END}", int(df["is_first_4w"].sum())))
log.append(("최종 행 수", "", len(df)))

df = df.sort_values(["timestamp", "visitorid", "itemid"], kind="stable")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8-sig")

log = pd.DataFrame(log, columns=["단계", "기준", "건수"])
LOG.parent.mkdir(parents=True, exist_ok=True)
log.to_csv(LOG, index=False, encoding="utf-8-sig")
print(log.to_string(index=False))
print("기간:", df["event_time_utc"].min(), "~", df["event_time_utc"].max())
print("이벤트별 건수:\n", df["event"].value_counts())
print(f"저장: {OUT.relative_to(ROOT)}, {LOG.relative_to(ROOT)}")
