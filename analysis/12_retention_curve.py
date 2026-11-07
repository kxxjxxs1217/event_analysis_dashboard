"""N-Day 리텐션 곡선 (UTC 날짜 기준).

입력: data/processed/events_clean.csv (03_clean_events.py 결과)
- 사용자 = visitorid. 활동일 = 어떤 이벤트(view/addtocart/transaction)든 1건 이상 있는 UTC 날짜
- Day 0 = 사용자의 첫 활동일(관측 기준). 코호트 = Day 0 날짜
- N-Day 리텐션 = 정확히 Day N 날짜에 활동한 사용자 수 / Day N을 관측할 수 있는 코호트의 사용자 수
  (그 사이 날짜에 왔는지는 따지지 않는다. Day N 이후 재방문을 누적하지 않는다)
- 관측 가능 조건: Day 0 + N <= 2015-09-14.
  2015-09-15는 21:00 UTC 기준 시점 전까지만 있어 하루가 온전하지 않으므로 마지막 관측일에서 뺀다.
- 좌측 절단: 첫 4주(2015-05-31 이전)에 처음 보인 사용자는 실제로는 그 전부터 온 기존 사용자일 수 있다.
  기본 곡선은 Day 0이 2015-05-31 이후인 코호트로 계산하고, 첫 4주를 포함한 전체 코호트 곡선은 비교용으로 함께 그린다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "retention_curve_nday.png"
TABLE = ROOT / "outputs" / "tables" / "retention_nday.csv"
FIRST_4W_END = pd.Timestamp("2015-05-31")
LAST_FULL_DAY = pd.Timestamp("2015-09-14")
PLOT_MAX_N = 30

ev = pd.read_csv(EVENTS, encoding="utf-8-sig", usecols=["visitorid", "event_time_utc"])
ev["date"] = pd.to_datetime(ev["event_time_utc"].str[:10])  # UTC 날짜
act = ev[["visitorid", "date"]].drop_duplicates()
first = act.groupby("visitorid")["date"].min().rename("d0")
act = act.join(first, on="visitorid")
act["n"] = (act["date"] - act["d0"]).dt.days


def retention(cohort_users):
    """코호트(Day 0 날짜)별 크기와 Day N 재방문자 수를 모아 N별 가중 리텐션 계산."""
    sizes = first[cohort_users].value_counts().sort_index()
    returned = act[act["visitorid"].isin(first.index[cohort_users])].groupby(["d0", "n"])["visitorid"].nunique()
    rows = []
    for n in range(0, (LAST_FULL_DAY - sizes.index.min()).days + 1):
        eligible = sizes[sizes.index <= LAST_FULL_DAY - pd.Timedelta(days=n)]
        idx = pd.MultiIndex.from_arrays([eligible.index, [n] * len(eligible)])
        back = int(returned.reindex(idx).fillna(0).sum())
        rows.append((n, len(eligible), int(eligible.sum()), back, back / eligible.sum() * 100))
    return pd.DataFrame(rows, columns=["N", "관측 가능 코호트 수", "대상 사용자 수", "Day N 재방문자 수", "재방문율(%)"])


main = retention(first >= FIRST_4W_END)
everyone = retention(first == first)

table = pd.concat([main.assign(코호트="첫 4주 제외 (Day 0 ≥ 2015-05-31)"),
                   everyone.assign(코호트="전체 (첫 4주 포함)")])
table = table[["코호트"] + list(main.columns)]
TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")

# 검증
assert (main.loc[main["N"] == 0, "재방문율(%)"] == 100).all()
assert act.loc[act["n"] == 0, "visitorid"].nunique() == len(first)
assert (main["Day N 재방문자 수"] <= main["대상 사용자 수"]).all()
print("검증 통과: Day 0 = 100%, 재방문자 수 <= 대상 사용자 수")
print("사용자 수: 전체", f"{len(first):,}", "/ 첫 4주 제외", f"{int((first >= FIRST_4W_END).sum()):,}")
print(main.head(PLOT_MAX_N + 1).round(3).to_string(index=False))

# 그래프: 왼쪽 Day 0~30 전체, 오른쪽 Day 1~30 확대
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAIN, COMP = "#2a78d6", "#9a9893"

m = main[main["N"] <= PLOT_MAX_N]
a = everyone[everyone["N"] <= PLOT_MAX_N]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.4]})
for ax, start, title in [(axes[0], 0, "Day 0 ~ 30"), (axes[1], 1, "Day 1 ~ 30 확대 (Day 0 = 100% 제외)")]:
    mm, aa = m[m["N"] >= start], a[a["N"] >= start]
    ax.set_facecolor(SURFACE)
    ax.plot(aa["N"], aa["재방문율(%)"], color=COMP, linewidth=1.4, linestyle="--", label="전체 코호트 (첫 4주 포함, 비교용)")
    ax.plot(mm["N"], mm["재방문율(%)"], color=MAIN, linewidth=2.2, marker="o", markersize=3.5,
            label="첫 4주 제외 코호트 (Day 0 ≥ 2015-05-31)")
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("N-Day (첫 방문일로부터 지난 일수)", fontsize=10, color=INK2)
    ax.set_ylabel("재방문율 (%)", fontsize=10, color=INK2)
    ax.set_xlim(start - 0.5, PLOT_MAX_N + 0.5)
    ax.set_xticks([start] + list(range(7, PLOT_MAX_N + 1, 7)) + [PLOT_MAX_N])
    ax.set_ylim(0, mm["재방문율(%)"].max() * 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

zoom = axes[1]
for n in [1, 7, 14, 28]:
    v = m.loc[m["N"] == n, "재방문율(%)"].item()
    zoom.annotate(f"Day {n}\n{v:.2f}%", (n, v), xytext=(6, 10), textcoords="offset points",
                  fontsize=9, color=INK, va="bottom")
zoom.legend(loc="upper right", frameon=False, fontsize=9.5)

fig.suptitle("N-Day 리텐션 곡선 (UTC 날짜 기준, 어떤 이벤트든 활동하면 재방문)", x=0.01, ha="left", fontsize=15, color=INK)
fig.text(0.01, 0.01,
         "· 재방문율 = Day N 당일에 활동한 사용자 수 ÷ Day N을 관측할 수 있는 코호트 사용자 수 (누적 아님)   "
         "· 관측 마지막 날 2015-09-14 (09-15는 21:00 UTC 전까지만 있어 제외)   "
         "· N이 클수록 일찍 들어온 코호트만 남음",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
