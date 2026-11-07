"""주요 지표 월별 추이 (UTC 월 기준).

지표 정의는 05_key_metrics.py와 같고 월 단위로 나눠 계산한다.
- 활성 사용자 수 / 구매자 수: 해당 월에 해당 이벤트가 있는 고유 방문자 (월별 값의 합은 전체 기간 값보다 클 수 있음)
- 구매 횟수: 고유 transactionid 수. 주문 월은 그 주문의 가장 이른 시각 기준
- 장바구니에 담은 횟수 / 이벤트 횟수: 이벤트 수
부분 월: 5월(2015-05-03 03:00부터)과 9월(2015-09-15 21:00 전까지)은 관측 일수가 짧아 따로 표시한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "monthly_key_metrics.png"
TABLE = ROOT / "outputs" / "tables" / "monthly_key_metrics.csv"
CUTOFF = pd.Timestamp("2015-09-15 21:00:00", tz="UTC")

ev = pd.read_csv(EVENTS, encoding="utf-8-sig", dtype={"transactionid": "Int64"},
                 usecols=["visitorid", "event", "transactionid", "event_time_utc"])
ev["event_time_utc"] = pd.to_datetime(ev["event_time_utc"])
ev["month"] = ev["event_time_utc"].dt.strftime("%Y-%m")

tx = ev[ev["event"] == "transaction"]
order_month = tx.groupby("transactionid")["event_time_utc"].min().dt.strftime("%Y-%m")

monthly = pd.DataFrame({
    "활성 사용자 수": ev.groupby("month")["visitorid"].nunique(),
    "구매자 수": tx.groupby("month")["visitorid"].nunique(),
    "구매 횟수": order_month.value_counts(),
    "장바구니에 담은 횟수": ev[ev["event"] == "addtocart"].groupby("month").size(),
    "이벤트 횟수": ev.groupby("month").size(),
}).sort_index().fillna(0).astype(int)

# 월별 관측 일수 (첫 이벤트 시각 ~ 기준 시점)
start = ev["event_time_utc"].min()
days = {}
for m in monthly.index:
    m0 = pd.Timestamp(m + "-01", tz="UTC")
    m1 = m0 + pd.offsets.MonthBegin(1)
    days[m] = round((min(m1, CUTOFF) - max(m0, start)).total_seconds() / 86400, 2)
monthly.insert(0, "관측 일수", pd.Series(days))
monthly["부분 월"] = monthly["관측 일수"] < monthly.index.map(lambda m: pd.Period(m).days_in_month)

TABLE.parent.mkdir(parents=True, exist_ok=True)
monthly.to_csv(TABLE, encoding="utf-8-sig", index_label="월")
print(monthly.to_string())

# 검증: 더할 수 있는 지표는 월별 합 = 전체 기간 값
assert monthly["구매 횟수"].sum() == tx["transactionid"].nunique()
assert monthly["장바구니에 담은 횟수"].sum() == int(ev["event"].eq("addtocart").sum())
assert monthly["이벤트 횟수"].sum() == len(ev)
print("검증 통과: 구매 횟수·장바구니·이벤트 월별 합 = 전체 기간 값")

# 그래프: 지표마다 별도 패널 (축 하나씩)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FULL, PARTIAL = "#2a78d6", "#86b6ef"

metrics = ["활성 사용자 수", "구매자 수", "구매 횟수", "장바구니에 담은 횟수", "이벤트 횟수"]
units = {"활성 사용자 수": "명", "구매자 수": "명", "구매 횟수": "회", "장바구니에 담은 횟수": "회", "이벤트 횟수": "회"}
labels = [f"{int(m[5:])}월" + (f"\n(부분 {d:g}일)" if p else "")
          for m, d, p in zip(monthly.index, monthly["관측 일수"], monthly["부분 월"])]
colors = [PARTIAL if p else FULL for p in monthly["부분 월"]]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor=SURFACE)
for ax, name in zip(axes.flat, metrics):
    vals = monthly[name]
    ax.set_facecolor(SURFACE)
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title(f"{name} ({units[name]})", loc="left", fontsize=12, color=INK)
    ax.set_xticks(range(len(vals)), labels, fontsize=9, color=INK2)
    ax.set_ylim(0, vals.max() * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

note = axes.flat[-1]
note.axis("off")
note.legend(handles=[Patch(color=FULL, label="전체 월"), Patch(color=PARTIAL, label="부분 월 (관측 일수가 짧음)")],
            loc="upper left", frameon=False, fontsize=10)
note.text(0.0, 0.62,
          "· 월은 UTC 기준\n"
          "· 활성 사용자: 어떤 이벤트든 1회 이상 한 고유 방문자\n"
          "· 구매자: transaction을 1회 이상 한 고유 방문자\n"
          "· 구매 횟수: 고유 주문(transactionid) 수\n"
          "· 5월은 첫 4주에 해당해 이전 행동이 관측되지 않음\n"
          "· 9월은 2015-09-15 21:00 UTC 전까지",
          va="top", fontsize=9.5, color=INK2, linespacing=1.7, transform=note.transAxes)

fig.suptitle("주요 지표 월별 추이 (2015-05 ~ 2015-09)", x=0.01, ha="left", fontsize=15, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
