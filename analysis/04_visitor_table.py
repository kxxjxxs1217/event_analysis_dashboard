"""방문자 단위 파생 테이블 (1행 = 1 visitorid).

입력: data/processed/events_clean.csv (03_clean_events.py 결과, CLAUDE.md 분석 기준 적용)
- 날짜는 UTC 날짜(YYYY-MM-DD)
- purchase_count: 고유 transactionid 수 (주문 수)
- purchased_item_count: 구매 행 수 (주문에 담긴 상품 건수)
- purchase_amount: 가격 데이터가 없어 전부 결측(NA)
- first_visit_in_first_4w: 첫 방문일이 가장 앞선 4주에 속함 → 실제 첫 방문은 관측 이전일 수 있음(좌측 절단)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_clean.csv"
OUT = ROOT / "data" / "processed" / "visitors.csv"

df = pd.read_csv(SRC, encoding="utf-8-sig", dtype={"transactionid": "Int64"},
                 usecols=["visitorid", "event", "transactionid", "event_time_utc", "is_first_4w"])
df["date"] = pd.to_datetime(df["event_time_utc"]).dt.strftime("%Y-%m-%d")

g = df.groupby("visitorid")
first_idx = df.sort_values("event_time_utc", kind="stable").groupby("visitorid").head(1).set_index("visitorid")
tx = df[df["event"] == "transaction"].groupby("visitorid")
counts = pd.crosstab(df["visitorid"], df["event"])

visitors = pd.DataFrame({
    "purchase_count": tx["transactionid"].nunique(),
    "purchased_item_count": counts.get("transaction"),
    "purchase_amount": pd.NA,
    "first_visit_date": g["date"].min(),
    "last_purchase_date": tx["date"].max(),
    "view_count": counts.get("view"),
    "addtocart_count": counts.get("addtocart"),
    "first_visit_in_first_4w": first_idx["is_first_4w"],
}, index=g.size().index)
for col in ["purchase_count", "purchased_item_count", "view_count", "addtocart_count"]:
    visitors[col] = visitors[col].fillna(0).astype(int)
visitors["purchase_amount"] = visitors["purchase_amount"].astype("Float64")
visitors = visitors.reset_index().sort_values("visitorid")

OUT.parent.mkdir(parents=True, exist_ok=True)
visitors.to_csv(OUT, index=False, encoding="utf-8-sig")

# 검증: 이벤트 원천과 합계 대조
ev = df["event"].value_counts()
checks = {
    "방문자 수 = 고유 visitorid": len(visitors) == df["visitorid"].nunique(),
    "조회 수 합계 = view 건수": visitors["view_count"].sum() == ev["view"],
    "장바구니 수 합계 = addtocart 건수": visitors["addtocart_count"].sum() == ev["addtocart"],
    "구매 상품 수 합계 = transaction 건수": visitors["purchased_item_count"].sum() == ev["transaction"],
    "구매 횟수 합계 = 고유 transactionid": visitors["purchase_count"].sum() == df["transactionid"].nunique(),
    "구매 0회인데 마지막 구매일 있음 없음": not (visitors["purchase_count"].eq(0) & visitors["last_purchase_date"].notna()).any(),
    "마지막 구매일 >= 첫 방문일": (visitors["last_purchase_date"].dropna() >= visitors.loc[visitors["last_purchase_date"].notna(), "first_visit_date"]).all(),
}
for k, v in checks.items():
    print(f"[{'OK' if v else 'FAIL'}] {k}")

print("\n행 수:", len(visitors))
print("구매 방문자:", int(visitors["purchase_count"].gt(0).sum()))
print("첫 방문이 가장 앞선 4주인 방문자:", int(visitors["first_visit_in_first_4w"].sum()))
print(visitors.describe().round(2).to_string())
print(visitors.sort_values("purchase_count", ascending=False).head(3).to_string(index=False))
print(f"저장: {OUT.relative_to(ROOT)}")
