"""View -> AddToCart -> Transaction 방문자 단위 퍼널 (CLAUDE.md 분석 기준 적용).

입력: data/processed/events_clean.csv (UTC, 2015-09-15 21:00 전, 완전 중복 제거)
단계 정의 (순서 강제, 전 단계를 수행한 방문자 중 다음 단계 수행)
- View: view가 1회 이상 있는 방문자
- AddToCart: 그 방문자의 첫 view 이후(같은 시각 포함) addtocart가 있는 방문자
- Transaction: 그 방문자의 (첫 view 이후) 첫 addtocart 이후(같은 시각 포함) transaction이 있는 방문자
전환율 = 해당 단계 방문자 ÷ 전 단계 방문자
가장 앞선 4주 고려: 첫 view가 첫 4주(2015-05-31 00:00 UTC 전)인 방문자와 그 이후인 방문자를 나눠 함께 집계
(02_funnel_conversion.py는 이 기준을 정하기 전의 원본 전체 기간 결과)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "funnel_view_cart_transaction.png"
TABLE = ROOT / "outputs" / "tables" / "funnel_view_cart_transaction.csv"
FIRST_4W_END = pd.Timestamp("2015-05-31 00:00:00").value // 10**6  # ms, UTC
STAGES = ["View", "AddToCart", "Transaction"]

ev = pd.read_csv(SRC, encoding="utf-8-sig", usecols=["timestamp", "visitorid", "event"])

first_view = ev[ev["event"] == "view"].groupby("visitorid")["timestamp"].min().rename("view_ts")
cart = ev[ev["event"] == "addtocart"].join(first_view, on="visitorid", how="inner")
first_cart = cart[cart["timestamp"] >= cart["view_ts"]].groupby("visitorid")["timestamp"].min().rename("cart_ts")
tx = ev[ev["event"] == "transaction"].join(first_cart, on="visitorid", how="inner")
buyers = tx.loc[tx["timestamp"] >= tx["cart_ts"], "visitorid"].unique()

visitors = first_view.to_frame()
visitors["cart"] = visitors.index.isin(first_cart.index)
visitors["buy"] = visitors.index.isin(buyers)
visitors["segment"] = (visitors["view_ts"] < FIRST_4W_END).map({True: "첫 view가 첫 4주", False: "첫 view가 첫 4주 이후"})


def funnel(df, label):
    n = [len(df), int(df["cart"].sum()), int(df["buy"].sum())]
    rows = []
    for i, s in enumerate(STAGES):
        rows.append({
            "구분": label, "단계": s, "방문자 수": n[i],
            "전 단계 대비 전환율(%)": round(n[i] / n[i - 1] * 100, 2) if i else None,
            "View 대비 전환율(%)": round(n[i] / n[0] * 100, 2),
        })
    return rows


table = pd.DataFrame(funnel(visitors, "전체")
                     + funnel(visitors[visitors["segment"] == "첫 view가 첫 4주"], "첫 view가 첫 4주")
                     + funnel(visitors[visitors["segment"] == "첫 view가 첫 4주 이후"], "첫 view가 첫 4주 이후"))

# 검증: 각 단계는 전 단계의 부분집합, 구분 합 = 전체
assert (visitors["buy"] <= visitors["cart"]).all()
pivot = table.pivot(index="단계", columns="구분", values="방문자 수")
assert (pivot["첫 view가 첫 4주"] + pivot["첫 view가 첫 4주 이후"]).equals(pivot["전체"])

# 참고: 순서·전 단계 조건 없이 이벤트별 고유 방문자
open_n = ev.groupby("event")["visitorid"].nunique()
table.loc[table["구분"] == "전체", "참고: 조건 없는 고유 방문자 수"] = [
    open_n["view"], open_n["addtocart"], open_n["transaction"]]
table["참고: 조건 없는 고유 방문자 수"] = table["참고: 조건 없는 고유 방문자 수"].astype("Int64")

TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")
print(table.to_string(index=False))

# ---------- 그래프 ----------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, SURFACE, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"

t = table[table["구분"] == "전체"].reset_index(drop=True)
counts = t["방문자 수"].tolist()
top = counts[0]
fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ys = [2, 1, 0]
for y, c in zip(ys, counts):
    ax.barh(y, c, left=(top - c) / 2, height=0.55, color=BLUE)
    ax.text(top / 2 + max(c, top * 0.02) / 2 + top * 0.015, y, f"{c:,}명", va="center", fontsize=12, color=INK)
for i in (1, 2):
    y_mid = (ys[i - 1] + ys[i]) / 2
    ax.text(top * 0.5, y_mid, f"↓ {t.loc[i, '전 단계 대비 전환율(%)']:.2f}%  (전 단계 대비)",
            va="center", ha="center", fontsize=11.5, color=INK, fontweight="bold")
ax.set_yticks(ys, [f"{s}\n(View 대비 {r:.2f}%)" if i else s
                   for i, (s, r) in enumerate(zip(STAGES, t["View 대비 전환율(%)"]))],
              fontsize=11, color=INK)
ax.set_xticks([])
ax.set_xlim(0, top * 1.22)
ax.set_ylim(-0.5, 2.5)
for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title(f"View → AddToCart → Transaction 퍼널 (방문자 단위, View → Transaction {t.loc[2, 'View 대비 전환율(%)']:.2f}%)",
             loc="left", fontsize=13.5, color=INK, pad=12)
fig.text(0.01, 0.015,
         "기준: 고유 방문자, 단계 순서 강제(첫 view 이후 장바구니, 그 첫 장바구니 이후 구매) · UTC · "
         "2015-05-03 ~ 2015-09-15 21:00 · 완전 중복 제거",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
