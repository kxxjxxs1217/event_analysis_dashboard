"""행동별 N-Day 리텐션 비교: 장바구니 사용자 / 구매 사용자 / 장바구니와 구매를 모두 한 사용자.

입력: data/processed/events_clean.csv
- 재방문 = 어떤 이벤트(view/addtocart/transaction)든 1건 이상 있는 UTC 날짜 (12_retention_curve.py와 같음)
- Day 0 = 그 행동을 처음 한 UTC 날짜
  - 장바구니: 첫 addtocart 날짜
  - 구매: 첫 transaction 날짜
  - 장바구니+구매: 둘 다 마친 날짜 = max(첫 addtocart 날짜, 첫 transaction 날짜). 둘 중 하나라도 없으면 제외
  - 전체(기준선): 첫 활동 날짜. 12_retention_curve.py의 기본 곡선과 같다
  행동 이후의 재방문만 보므로, "나중에 구매하려면 다시 와야 한다"는 미래 정보가 집단 구분에 섞이지 않는다.
- 세 집단은 서로 겹친다 (구매자 대부분이 장바구니 사용자이기도 하다)
- N-Day 재방문율 = 정확히 Day N 당일 활동한 사용자 ÷ Day N을 관측할 수 있는 사용자 (누적 아님)
- 관측 가능 조건: Day 0 + N <= 2015-09-14 (09-15는 21:00 UTC 전까지만 있어 하루가 온전하지 않음)
- 좌측 절단: Day 0이 첫 4주(2015-05-31 전)인 사용자는 그 행동이 실제 첫 행동이 아닐 수 있어 제외한다
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
BASELINE = ROOT / "outputs" / "tables" / "retention_nday.csv"
FIG = ROOT / "outputs" / "figures" / "retention_by_action.png"
TABLE = ROOT / "outputs" / "tables" / "retention_by_action.csv"
FIRST_4W_END = pd.Timestamp("2015-05-31")
LAST_FULL_DAY = pd.Timestamp("2015-09-14")
PLOT_MAX_N = 30

ev = pd.read_csv(EVENTS, encoding="utf-8-sig", usecols=["visitorid", "event", "event_time_utc"])
ev["date"] = pd.to_datetime(ev["event_time_utc"].str[:10])  # UTC 날짜
act = ev[["visitorid", "date"]].drop_duplicates()

first_cart = ev[ev["event"] == "addtocart"].groupby("visitorid")["date"].min()
first_tx = ev[ev["event"] == "transaction"].groupby("visitorid")["date"].min()
both = pd.concat([first_cart, first_tx], axis=1, join="inner").max(axis=1)
GROUPS = {
    "전체 사용자 (기준선, Day 0 = 첫 방문)": act.groupby("visitorid")["date"].min(),
    "장바구니 사용자 (Day 0 = 첫 장바구니)": first_cart,
    "구매 사용자 (Day 0 = 첫 구매)": first_tx,
    "장바구니+구매 사용자 (Day 0 = 둘 다 마친 날)": both,
}


def retention(d0):
    """d0: visitorid별 Day 0 날짜. N별 (관측 가능 사용자, Day N 재방문자, 재방문율)."""
    d0 = d0[d0 >= FIRST_4W_END].rename("d0")
    sizes = d0.value_counts().sort_index()
    a = act.join(d0, on="visitorid", how="inner")
    a["n"] = (a["date"] - a["d0"]).dt.days
    returned = a[a["n"] >= 0].groupby(["d0", "n"])["visitorid"].nunique()
    rows = []
    for n in range(0, (LAST_FULL_DAY - sizes.index.min()).days + 1):
        eligible = sizes[sizes.index <= LAST_FULL_DAY - pd.Timedelta(days=n)]
        idx = pd.MultiIndex.from_arrays([eligible.index, [n] * len(eligible)])
        back = int(returned.reindex(idx).fillna(0).sum())
        rows.append((n, int(eligible.sum()), back, back / eligible.sum() * 100))
    return pd.DataFrame(rows, columns=["N", "대상 사용자 수", "Day N 재방문자 수", "재방문율(%)"])


res = {name: retention(d0) for name, d0 in GROUPS.items()}
table = pd.concat([r.assign(집단=name) for name, r in res.items()])
table = table[["집단", "N", "대상 사용자 수", "Day N 재방문자 수", "재방문율(%)"]]
TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")

# 검증: 기준선은 12_retention_curve.py 결과와 같아야 한다
base = pd.read_csv(BASELINE, encoding="utf-8-sig")
base = base[base["코호트"].str.startswith("첫 4주 제외")].reset_index(drop=True)
mine = res["전체 사용자 (기준선, Day 0 = 첫 방문)"]
assert mine["Day N 재방문자 수"].equals(base["Day N 재방문자 수"]) and mine["대상 사용자 수"].equals(base["대상 사용자 수"])
assert all((r.loc[r["N"] == 0, "재방문율(%)"] == 100).all() for r in res.values())
print("검증 통과: 기준선 = 12_retention_curve.py 결과, 모든 집단 Day 0 = 100%")

print("\n집단별 사용자 수 (전체 / Day 0 >= 2015-05-31):")
for name, d0 in GROUPS.items():
    print(f"  {name}: {len(d0):,} / {int((d0 >= FIRST_4W_END).sum()):,}")
print(f"  구매했지만 장바구니 기록 없음: {len(first_tx.index.difference(first_cart.index)):,}명")

key = table[table["N"].isin([1, 3, 7, 14, 21, 28, 30])]
print("\n" + key.pivot(index="N", columns="집단", values="재방문율(%)").round(2).to_string())

# 그래프: Day 1 ~ 30
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "전체 사용자 (기준선, Day 0 = 첫 방문)": dict(color="#9a9893", linestyle="--", linewidth=1.6),
    "장바구니 사용자 (Day 0 = 첫 장바구니)": dict(color="#2a78d6", linewidth=2.2, marker="o", markersize=3.5),
    "구매 사용자 (Day 0 = 첫 구매)": dict(color="#d95f02", linewidth=2.2, marker="o", markersize=3.5),
    "장바구니+구매 사용자 (Day 0 = 둘 다 마친 날)": dict(color="#1b9e77", linewidth=1.6, marker="s", markersize=3, linestyle=":"),
}

fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for name, r in res.items():
    r = r[(r["N"] >= 1) & (r["N"] <= PLOT_MAX_N)]
    users = int(res[name].loc[0, "대상 사용자 수"])
    ax.plot(r["N"], r["재방문율(%)"], label=f"{name}  n={users:,}", **STYLE[name])

ax.set_xlim(0.5, PLOT_MAX_N + 0.5)
ax.set_xticks([1] + list(range(7, PLOT_MAX_N + 1, 7)) + [PLOT_MAX_N])
ax.set_ylim(0, None)
ax.set_xlabel("N-Day (Day 0 행동일로부터 지난 일수)", fontsize=10, color=INK2)
ax.set_ylabel("재방문율 (%)", fontsize=10, color=INK2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
ax.tick_params(colors=INK2, length=0, labelsize=9)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)

fig.suptitle("행동별 N-Day 재방문율 (Day 1 ~ 30, Day 0 = 100%)", x=0.01, ha="left", fontsize=15, color=INK)
fig.text(0.01, 0.01,
         "· 재방문 = 어떤 이벤트든 활동한 UTC 날짜 · 재방문율 = Day N 당일 활동 사용자 ÷ Day N을 관측할 수 있는 사용자 (누적 아님)   "
         "· Day 0이 2015-05-31 이후인 사용자만 · 관측 마지막 날 2015-09-14 · 세 집단은 서로 겹침",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
