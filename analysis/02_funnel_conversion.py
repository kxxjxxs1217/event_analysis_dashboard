"""조회 -> 장바구니 -> 구매 방문자 단위 퍼널과 단계별 전환율.

기준
- 집계 단위: 고유 방문자(visitorid)
- 정제: 완전 중복 행 제거 (01_data_quality_check.py에서 460건 확인)
- 순서 강제: 장바구니는 그 방문자의 첫 조회 이후, 구매는 조회 이후 첫 장바구니 이후 시각만 인정
- 헤비 유저와 불완전한 날은 제외하지 않음 (전체 기간, 전체 방문자)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "events.csv"
FIG = ROOT / "outputs" / "figures" / "funnel_view_cart_purchase.png"
TABLE = ROOT / "outputs" / "tables" / "funnel_conversion.csv"

df = pd.read_csv(RAW)
n_raw = len(df)
df = df.drop_duplicates()
print(f"완전 중복 제거: {n_raw - len(df)}건 (남은 행 {len(df)})")

# 단계별 첫 도달 시각 (순서 강제)
first_view = df[df["event"] == "view"].groupby("visitorid")["timestamp"].min().rename("view_ts")

cart = df[df["event"] == "addtocart"].join(first_view, on="visitorid")
cart = cart[cart["timestamp"] >= cart["view_ts"]]
first_cart = cart.groupby("visitorid")["timestamp"].min().rename("cart_ts")

tx = df[df["event"] == "transaction"].join(first_cart, on="visitorid")
tx = tx[tx["timestamp"] >= tx["cart_ts"]]
buyers = tx["visitorid"].unique()

stages = ["조회 (view)", "장바구니 (addtocart)", "구매 (transaction)"]
counts = [first_view.size, first_cart.size, len(buyers)]

# 참고: 순서/이전 단계 조건 없이 이벤트별 고유 방문자 수
simple = df.groupby("event")["visitorid"].nunique()
simple_counts = [simple["view"], simple["addtocart"], simple["transaction"]]

table = pd.DataFrame({"단계": stages, "방문자 수": counts})
table["이전 단계 대비 전환율(%)"] = (table["방문자 수"] / table["방문자 수"].shift(1) * 100).round(2)
table["첫 단계 대비 전환율(%)"] = (table["방문자 수"] / counts[0] * 100).round(2)
table["참고: 단순 고유 방문자 수"] = simple_counts
TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")
print(table.to_string(index=False))

# 퍼널 그래프 (가운데 정렬 가로 막대, 단일 색)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, BLUE, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"

fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y = range(len(stages))[::-1]
top = counts[0]
for yi, c in zip(y, counts):
    ax.barh(yi, c, left=(top - c) / 2, height=0.62, color=BLUE)
    ax.text(top / 2 + max(c, top * 0.02) / 2 + top * 0.015, yi, f"{c:,}명",
            va="center", ha="left", color=INK, fontsize=11)

# 단계 사이 전환율
for i in range(1, len(counts)):
    rate = counts[i] / counts[i - 1] * 100
    ax.text(top * 0.02, (y[i - 1] + y[i]) / 2, f"↓ {rate:.2f}%",
            va="center", ha="left", color=INK2, fontsize=10)

ax.set_yticks(list(y), stages, color=INK, fontsize=11)
ax.set_xticks([])
ax.set_xlim(0, top * 1.25)
for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
overall = counts[-1] / counts[0] * 100
ax.set_title(f"방문자 단위 구매 퍼널 (조회 → 구매 전체 전환율 {overall:.2f}%)",
             loc="left", color=INK, fontsize=13, pad=12)
fig.text(0.01, 0.01, "기준: 고유 방문자, 완전 중복 제거, 단계 순서 강제 · 2015-05-03 ~ 2015-09-18 · Retailrocket (CC BY-NC-SA 4.0)",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
