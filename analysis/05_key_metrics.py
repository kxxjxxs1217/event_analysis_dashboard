"""주요 지표: 이벤트 테이블과 방문자 테이블에서 각각 계산하고 서로 대조한다.

입력
- data/processed/events_clean.csv (03_clean_events.py)
- data/processed/visitors.csv (04_visitor_table.py)
기간: 2015-05-03 03:00 ~ 2015-09-15 21:00 미만 (UTC), 완전 중복 제거
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
VISITORS = ROOT / "data" / "processed" / "visitors.csv"
OUT = ROOT / "outputs" / "tables" / "key_metrics.csv"

ev = pd.read_csv(EVENTS, encoding="utf-8-sig", usecols=["visitorid", "event", "transactionid"],
                 dtype={"transactionid": "Int64"})
vi = pd.read_csv(VISITORS, encoding="utf-8-sig")

rows = [
    ("활성 사용자 수", "view·addtocart·transaction 중 어떤 이벤트든 1회 이상 한 고유 방문자",
     ev["visitorid"].nunique(),
     int(vi[["view_count", "addtocart_count", "purchased_item_count"]].sum(axis=1).gt(0).sum())),
    ("구매자 수", "transaction을 1회 이상 한 고유 방문자",
     ev.loc[ev["event"] == "transaction", "visitorid"].nunique(), int(vi["purchase_count"].gt(0).sum())),
    ("구매 횟수", "고유 transactionid 수 (주문 수)",
     ev["transactionid"].nunique(), int(vi["purchase_count"].sum())),
    ("장바구니에 담은 횟수", "addtocart 이벤트 수",
     int(ev["event"].eq("addtocart").sum()), int(vi["addtocart_count"].sum())),
    ("이벤트 횟수", "전체 이벤트 수 (view + addtocart + transaction)",
     len(ev), int(vi[["view_count", "addtocart_count", "purchased_item_count"]].sum().sum())),
]
metrics = pd.DataFrame(rows, columns=["지표", "정의", "값(이벤트 테이블)", "값(방문자 테이블)"])
metrics["일치"] = metrics["값(이벤트 테이블)"] == metrics["값(방문자 테이블)"]

OUT.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(OUT, index=False, encoding="utf-8-sig")
print(metrics.to_string(index=False))

# 참고 수치
print("\n참고")
print("구매 이벤트 행 수(주문에 담긴 상품 건수):", int(ev["event"].eq("transaction").sum()))
print("view 이벤트 수:", int(ev["event"].eq("view").sum()))
print("이전 정의(view 1회 이상) 활성 사용자:", ev.loc[ev["event"] == "view", "visitorid"].nunique())
print(f"저장: {OUT.relative_to(ROOT)}")
