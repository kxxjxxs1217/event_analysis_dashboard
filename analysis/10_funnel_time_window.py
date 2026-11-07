"""시간 제한 퍼널: 첫 조회부터 24시간 / 7일 안에 장바구니·구매까지 간 방문자 비교.

단계 정의는 09_funnel.py와 같다 (순서 강제: 첫 view → 그 이후 첫 addtocart → 그 이후 transaction).
시간 제한: 장바구니와 구매 모두 첫 view 시각으로부터 W 이내(<=)여야 인정.
비교 코호트: 세 조건(24시간, 7일, 제한 없음)이 같은 방문자 집단을 보도록,
  첫 view가 기준 시점(2015-09-15 21:00 UTC) 7일 전보다 이른 방문자만 사용한다.
  → 모든 방문자에게 7일 관측 창이 보장된다. '제한 없음'은 관측 종료까지.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_clean.csv"
FIG_FUNNEL = ROOT / "outputs" / "figures" / "funnel_time_window.png"
FIG_CDF = ROOT / "outputs" / "figures" / "funnel_time_to_convert.png"
TABLE = ROOT / "outputs" / "tables" / "funnel_time_window.csv"
MS_H = 3600 * 1000
CUTOFF_MS = pd.Timestamp("2015-09-15 21:00:00").value // 10**6
WINDOWS = {"24시간 이내": 24 * MS_H, "7일 이내": 7 * 24 * MS_H, "제한 없음": None}
STAGES = ["View", "AddToCart", "Transaction"]

ev = pd.read_csv(SRC, encoding="utf-8-sig", usecols=["timestamp", "visitorid", "event"])

v = ev[ev["event"] == "view"].groupby("visitorid")["timestamp"].min().rename("view_ts").to_frame()
cart = ev[ev["event"] == "addtocart"].join(v, on="visitorid", how="inner")
v["cart_ts"] = cart[cart["timestamp"] >= cart["view_ts"]].groupby("visitorid")["timestamp"].min()
tx = ev[ev["event"] == "transaction"].join(v["cart_ts"], on="visitorid", how="inner")
v["tx_ts"] = tx[tx["timestamp"] >= tx["cart_ts"]].groupby("visitorid")["timestamp"].min()

n_all = len(v)
cohort = v[v["view_ts"] < CUTOFF_MS - WINDOWS["7일 이내"]].copy()
print(f"첫 view 방문자 {n_all:,}명 중 7일 관측 창이 보장된 코호트 {len(cohort):,}명 "
      f"(마지막 7일에 첫 view인 {n_all - len(cohort):,}명 제외)")
cohort["to_cart_h"] = (cohort["cart_ts"] - cohort["view_ts"]) / MS_H
cohort["to_tx_h"] = (cohort["tx_ts"] - cohort["view_ts"]) / MS_H

rows = []
for name, w in WINDOWS.items():
    lim = np.inf if w is None else w / MS_H
    n = [len(cohort), int((cohort["to_cart_h"] <= lim).sum()), int((cohort["to_tx_h"] <= lim).sum())]
    for i, s in enumerate(STAGES):
        rows.append({"시간 제한": name, "단계": s, "방문자 수": n[i],
                     "전 단계 대비 전환율(%)": round(n[i] / n[i - 1] * 100, 2) if i else None,
                     "View 대비 전환율(%)": round(n[i] / n[0] * 100, 2)})
table = pd.DataFrame(rows)

# 검증: 제한이 길수록 단계별 방문자 수가 줄지 않음, 구매 ⊂ 장바구니
p = table.pivot(index="단계", columns="시간 제한", values="방문자 수")
assert ((p["24시간 이내"] <= p["7일 이내"]) & (p["7일 이내"] <= p["제한 없음"])).all()
assert (cohort["to_tx_h"].dropna() >= cohort["to_cart_h"][cohort["to_tx_h"].notna()]).all()

TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")
print(table.to_string(index=False))

buy_h = cohort["to_tx_h"].dropna()
cart_h = cohort["to_cart_h"].dropna()
print("\n구매 전환자(제한 없음) 첫 view→구매 소요시간(시간) 분위:\n", buy_h.quantile([.25, .5, .75, .9]).round(2).to_string())
print("구매 전환자 중 24시간 이내 비율(%):", round((buy_h <= 24).mean() * 100, 1),
      "/ 7일 이내 비율(%):", round((buy_h <= 168).mean() * 100, 1))
print("장바구니 전환자 첫 view→장바구니 소요시간 중앙값(시간):", round(cart_h.median(), 2))

# ---------- 그래프 1: 시간 제한별 퍼널 (작은 다중 그래프, 같은 축) ----------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"

fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE, sharey=True)
top = len(cohort)
ys = [2, 1, 0]
for ax, name in zip(axes, WINDOWS):
    t = table[table["시간 제한"] == name].reset_index(drop=True)
    ax.set_facecolor(SURFACE)
    for y, c in zip(ys, t["방문자 수"]):
        ax.barh(y, c, left=(top - c) / 2, height=0.5, color=BLUE)
        if c > top * 0.5:  # 넓은 막대는 안쪽에
            ax.text(top / 2, y, f"{c:,}명", ha="center", va="center", fontsize=11, color="white")
        else:
            ax.text(top / 2 + c / 2 + top * 0.03, y, f"{c:,}명", ha="left", va="center", fontsize=11, color=INK)
    for i in (1, 2):
        ax.text(top / 2, (ys[i - 1] + ys[i]) / 2, f"↓ {t.loc[i, '전 단계 대비 전환율(%)']:.2f}%",
                ha="center", va="center", fontsize=12, color=INK, fontweight="bold")
    ax.set_title(f"{name}\nView → Transaction {t.loc[2, 'View 대비 전환율(%)']:.2f}%",
                 fontsize=12.5, color=INK, pad=8)
    ax.set_xlim(0, top)
    ax.set_ylim(-0.6, 2.45)
    ax.set_xticks([])
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ys, STAGES, fontsize=11.5, color=INK)
fig.suptitle(f"첫 조회 이후 시간 제한별 퍼널 비교 (같은 코호트 {len(cohort):,}명)", x=0.01, ha="left",
             fontsize=15, color=INK)
fig.text(0.01, 0.015,
         "↓ 전 단계 대비 전환율 · 시간 제한: 장바구니·구매 모두 첫 view로부터 해당 시간 이내 · 단계 순서 강제 · "
         "코호트: 첫 view가 2015-09-08 21:00 UTC 전인 방문자(7일 관측 창 보장) · UTC · 완전 중복 제거",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
FIG_FUNNEL.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_FUNNEL, dpi=150, facecolor=SURFACE)

# ---------- 그래프 2: 첫 조회 이후 경과 시간별 누적 전환율 ----------
grid_h = np.logspace(np.log10(1 / 60), np.log10(24 * 135), 400)
cum_cart = np.searchsorted(np.sort(cart_h.values), grid_h, side="right") / len(cohort) * 100
cum_buy = np.searchsorted(np.sort(buy_h.values), grid_h, side="right") / len(cohort) * 100

fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(grid_h, cum_cart, color=BLUE, linewidth=2)
ax.plot(grid_h, cum_buy, color=ORANGE, linewidth=2)
ax.text(grid_h[-1] * 1.05, cum_cart[-1], f"AddToCart {cum_cart[-1]:.2f}%", va="center", color=INK, fontsize=10)
ax.text(grid_h[-1] * 1.05, cum_buy[-1], f"Transaction {cum_buy[-1]:.2f}%", va="center", color=INK, fontsize=10)
for h, label in [(24, "24시간"), (168, "7일")]:
    ax.axvline(h, color=INK2, linewidth=1, linestyle="--")
    c_cart = (cart_h <= h).sum() / len(cohort) * 100
    c_buy = (buy_h <= h).sum() / len(cohort) * 100
    ax.text(h * 1.06, cum_cart[-1] * 1.08, label, color=INK2, fontsize=10, va="bottom")
    for val, col in [(c_cart, BLUE), (c_buy, ORANGE)]:
        ax.plot(h, val, "o", color=col, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(h / 1.08, val, f"{val:.2f}%", ha="right", va="bottom", fontsize=9.5, color=INK)
ax.set_xscale("log")
ticks = [1 / 60, 10 / 60, 1, 6, 24, 72, 168, 24 * 30, 24 * 90]
ax.set_xticks(ticks, ["1분", "10분", "1시간", "6시간", "1일", "3일", "7일", "30일", "90일"])
ax.set_xlim(grid_h[0], grid_h[-1] * 3.2)
ax.set_ylim(0, cum_cart[-1] * 1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
ax.set_xlabel("첫 조회 이후 경과 시간 (로그 축)", color=INK2, fontsize=10)
ax.set_ylabel("코호트 대비 누적 도달 비율", color=INK2, fontsize=10)
ax.legend(["AddToCart 도달", "Transaction 도달"], loc="upper left", frameon=False, fontsize=10)
ax.set_title(f"첫 조회 이후 경과 시간별 누적 전환율 (코호트 {len(cohort):,}명)", loc="left", fontsize=14,
             color=INK, pad=10)
fig.text(0.01, 0.015, "단계 순서 강제(첫 view → 첫 장바구니 → 구매) · 점선: 24시간, 7일 · 코호트는 7일 관측 창이 보장된 방문자",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG_CDF, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG_FUNNEL.relative_to(ROOT)}, {FIG_CDF.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
