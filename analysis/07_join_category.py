"""events_clean에 상품 카테고리를 left join.

item_category_history는 상품별 주간 스냅샷(effective_timestamp)이라 itemid만으로 조인하면 이벤트 행이 복제된다.
그래서 이벤트 행 수를 그대로 유지하는 시점 기준(as-of) left join을 한다.
- 시점일치: 이벤트 시각 이전(<=) 가장 최근 스냅샷의 카테고리
- 첫스냅샷대체: 상품의 첫 스냅샷보다 이른 이벤트 → 그 상품의 가장 이른 스냅샷 카테고리로 채움
- 이력없음: history에 없는 상품 → 카테고리 결측
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
HISTORY = ROOT / "data" / "raw" / "item_category_history.csv"
TREE = ROOT / "data" / "raw" / "category_tree.csv"
OUT = ROOT / "data" / "processed" / "events_with_category.csv"
LOG = ROOT / "outputs" / "tables" / "category_join_log.csv"

ev = pd.read_csv(EVENTS, encoding="utf-8-sig", dtype={"transactionid": "Int64"})
hist = pd.read_csv(HISTORY, dtype={"root_categoryid": "Int64"})
hist = hist.rename(columns={"effective_timestamp": "timestamp"}).sort_values("timestamp")
n_events = len(ev)

ev["_row"] = range(n_events)
left = ev[["_row", "timestamp", "itemid"]].sort_values("timestamp")
cols = ["timestamp", "itemid", "categoryid", "root_categoryid"]
back = pd.merge_asof(left, hist[cols], on="timestamp", by="itemid", direction="backward")
fwd = pd.merge_asof(left, hist[cols], on="timestamp", by="itemid", direction="forward")
back, fwd = back.set_index("_row").sort_index(), fwd.set_index("_row").sort_index()

use_back = back["categoryid"].notna()
use_fwd = ~use_back & fwd["categoryid"].notna()
ev["categoryid"] = back["categoryid"].where(use_back, fwd["categoryid"]).astype("Int64").values
ev["root_categoryid"] = back["root_categoryid"].where(use_back, fwd["root_categoryid"]).astype("Int64").values
ev["category_match"] = "이력없음"
ev.loc[use_back.values, "category_match"] = "시점일치"
ev.loc[use_fwd.values, "category_match"] = "첫스냅샷대체"
ev = ev.drop(columns="_row")

# 검증
assert len(ev) == n_events, "이벤트 행 수가 바뀜"
assert ev["categoryid"].isna().eq(ev["category_match"].eq("이력없음")).all()
tree_ids = set(pd.read_csv(TREE)["categoryid"])
not_in_tree = ev["categoryid"].notna() & ~ev["categoryid"].isin(tree_ids)

OUT.parent.mkdir(parents=True, exist_ok=True)
ev.to_csv(OUT, index=False, encoding="utf-8-sig")

match = ev["category_match"].value_counts().reindex(["시점일치", "첫스냅샷대체", "이력없음"])
log = pd.DataFrame({
    "구분": ["이벤트 행 수(조인 전=후)"] + list(match.index) + ["category_tree에 없는 categoryid (root 결측)"],
    "건수": [n_events] + list(match.values) + [int(not_in_tree.sum())],
})
log["비율(%)"] = (log["건수"] / n_events * 100).round(2)
LOG.parent.mkdir(parents=True, exist_ok=True)
log.to_csv(LOG, index=False, encoding="utf-8-sig")
print(log.to_string(index=False))

print("\n이벤트 종류별 매칭 비율(%):")
print((pd.crosstab(ev["event"], ev["category_match"], normalize="index") * 100).round(2).to_string())
print("\n카테고리 결측 열:", ev[["categoryid", "root_categoryid"]].isna().sum().to_dict())
print("고유 categoryid:", ev["categoryid"].nunique(), "/ 고유 root_categoryid:", ev["root_categoryid"].nunique())
print(f"저장: {OUT.relative_to(ROOT)}, {LOG.relative_to(ROOT)}")
