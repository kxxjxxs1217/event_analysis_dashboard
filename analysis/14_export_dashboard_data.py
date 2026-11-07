"""배포용 대시보드 데이터 생성 (Streamlit Community Cloud용).

events_with_category.csv(약 312MB)는 GitHub 파일 한도(100MB)를 넘으므로,
대시보드가 쓰는 7개 열만 zstd 압축 Parquet으로 저장한다. 행과 값은 그대로다.
입력: data/processed/events_with_category.csv (07_join_category.py 결과)
출력: data/processed/events_dashboard.parquet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_with_category.csv"
OUT = ROOT / "data" / "processed" / "events_dashboard.parquet"
COLS = ["timestamp", "visitorid", "event", "itemid", "transactionid", "categoryid", "root_categoryid"]
DTYPES = {"transactionid": "Int64", "categoryid": "Int64", "root_categoryid": "Int64"}

ev = pd.read_csv(SRC, encoding="utf-8-sig", usecols=COLS, dtype=DTYPES)[COLS]
ev.to_parquet(OUT, compression="zstd", index=False)

# 검증: 다시 읽어 원본과 같은지 확인
back = pd.read_parquet(OUT)
assert back.equals(ev), "Parquet 값이 원본과 다름"
print(f"저장: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 2**20:.1f} MB, {len(back):,}행, 원본과 동일)")
