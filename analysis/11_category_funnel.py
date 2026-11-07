"""대분류·카테고리별 24시간 퍼널 성과 분석 (정의는 funnel_lib.py).

입력: data/processed/events_with_category.csv
출력: outputs/tables/funnel_24h_overall.csv, funnel_24h_by_root.csv, funnel_24h_by_category.csv,
      funnel_24h_distributions.csv, outputs/figures/funnel_24h_by_root.png
비교 시 소표본 왜곡을 줄이려고 View 방문자 1,000명 이상인 대분류·카테고리만 순위 비교에 쓴다(표에는 전부 저장).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import funnel_lib as fl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_with_category.csv"
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" / "funnel_24h_by_root.png"
MIN_VIEW = 1000

ev = fl.load_events(SRC)
overall = fl.build(ev, ["전체"])
by_root = fl.build(ev, ["대분류"])
by_cat = fl.build(ev, ["대분류", "카테고리"])

# 검증: 10_funnel_time_window.py의 24시간 결과(7일 창 코호트)와 같은 정의인지 확인
check = fl.build(ev, ["전체"], cutoff_ms=fl.CUTOFF_MS - 6 * fl.WINDOW_MS)
print("10번 스크립트 24시간 결과와 일치(26,727 / 7,519):", fl.stage_counts(check)[1:] == [26727, 7519])

t_all = fl.summary_table(overall, "전체")
t_root = fl.summary_table(by_root, "대분류")
t_cat = fl.summary_table(by_cat, "카테고리").merge(
    by_cat[["카테고리", "대분류"]].drop_duplicates().groupby("카테고리", observed=True)["대분류"]
    .agg(lambda s: ", ".join(sorted(map(str, s)))).reset_index(), on="카테고리")

dists = []
for name, f in [("전체", overall)] + [(r, by_root[by_root["대분류"] == r]) for r in t_root["대분류"]]:
    for kind, col, func in [("View→AddToCart 소요 시간", "to_cart_min", fl.time_dist),
                            ("AddToCart→Transaction 소요 시간", "to_tx_min", fl.time_dist),
                            ("AddToCart 전 View 횟수", "n_view_before_cart", fl.count_dist),
                            ("Transaction 전 AddToCart 횟수", "n_cart_before_tx", fl.count_dist)]:
        d = func(f, col)
        d.insert(0, "분포", kind)
        d.insert(0, "범위", name)
        dists.append(d)
dists = pd.concat(dists)

TAB.mkdir(parents=True, exist_ok=True)
t_all.to_csv(TAB / "funnel_24h_overall.csv", index=False, encoding="utf-8-sig")
t_root.to_csv(TAB / "funnel_24h_by_root.csv", index=False, encoding="utf-8-sig")
t_cat.to_csv(TAB / "funnel_24h_by_category.csv", index=False, encoding="utf-8-sig")
dists.to_csv(TAB / "funnel_24h_distributions.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\n[전체]\n", t_all.round(2).to_string(index=False))
print("\n[대분류]\n", t_root.round(2).to_string(index=False))
big_cat = t_cat[t_cat["View"] >= MIN_VIEW]
print(f"\n[카테고리] 전체 {len(t_cat)}개 중 View {MIN_VIEW:,}명 이상 {len(big_cat)}개")
cols = ["카테고리", "대분류", "View", "AddToCart", "Transaction", "View→AddToCart(%)", "AddToCart→Transaction(%)",
        "View→Transaction(%)"]
print("View→Transaction 상위 10:\n", big_cat.sort_values("View→Transaction(%)", ascending=False)[cols].head(10).round(2).to_string(index=False))
print("View→Transaction 하위 10:\n", big_cat.sort_values("View→Transaction(%)")[cols].head(10).round(2).to_string(index=False))
print("View 상위 10:\n", big_cat[cols].head(10).round(2).to_string(index=False))
print("AddToCart→Transaction 상위 10 (AddToCart 100명 이상):\n",
      t_cat[t_cat["AddToCart"] >= 100].sort_values("AddToCart→Transaction(%)", ascending=False)[cols].head(10).round(2).to_string(index=False))
print("AddToCart→Transaction 하위 10 (AddToCart 100명 이상):\n",
      t_cat[t_cat["AddToCart"] >= 100].sort_values("AddToCart→Transaction(%)")[cols].head(10).round(2).to_string(index=False))
q = big_cat["View→Transaction(%)"]
print("큰 카테고리 View→Transaction 분포:", q.describe().round(3).to_dict())
print("구매 0인 큰 카테고리:", int((big_cat["Transaction"] == 0).sum()))
print("\n[전체 분포]\n", dists[dists["범위"] == "전체"].round(1).to_string(index=False))
print("\n전체 중앙값(분): View→Cart", round(float(overall["to_cart_min"].median()), 2),
      "/ Cart→Tx", round(float(overall["to_tx_min"].median()), 2))
print("전체 평균 이전 단계 횟수: view before cart", round(float(overall["n_view_before_cart"].mean()), 2),
      "/ cart before tx", round(float(overall["n_cart_before_tx"].mean()), 2))

# ---------- 대분류 비교 그래프 ----------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURFACE, BLUE, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#a8a7a2"
r = t_root[t_root["View"] >= MIN_VIEW].iloc[::-1]
avg = {"View→AddToCart(%)": t_all.loc[0, "View→AddToCart(%)"],
       "AddToCart→Transaction(%)": t_all.loc[0, "AddToCart→Transaction(%)"],
       "View→Transaction(%)": t_all.loc[0, "View→Transaction(%)"]}
labels = [f"{k}  (View {v:,})" for k, v in zip(r["대분류"], r["View"])]
fig, axes = plt.subplots(1, 3, figsize=(16, 0.42 * len(r) + 2.2), facecolor=SURFACE, sharey=True)
for ax, col in zip(axes, avg):
    vals = r[col]
    ax.set_facecolor(SURFACE)
    ax.barh(labels, vals, color=[GRAY if k == fl.NO_CAT else BLUE for k in r["대분류"]], height=0.65)
    base = r["AddToCart"] if col.startswith("AddToCart") else r["View"]
    for y, (v, n) in enumerate(zip(vals, base)):
        label = "–" if pd.isna(v) else f"{v:.2f}%"
        if col.startswith("AddToCart"):  # 분모(장바구니 방문자)가 작으면 함께 표시
            label += f"  (n={n:,})"
        ax.text((0 if pd.isna(v) else v) + vals.max() * 0.015, y, label, va="center", fontsize=8.5, color=INK)
    ax.axvline(avg[col], color=INK2, linestyle="--", linewidth=1)
    ax.text(avg[col], len(r) - 0.3, f" 전체 {avg[col]:.2f}%", fontsize=8.5, color=INK2, va="bottom")
    ax.set_title(col.replace("(%)", " 전환율"), loc="left", fontsize=12, color=INK)
    ax.set_xlim(0, vals.max() * 1.35)
    digits = 1 if vals.max() < 5 else 0
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _, d=digits: f"{x:.{d}f}%"))
    ax.tick_params(axis="x", labelsize=8, colors=INK2, length=0)
    ax.tick_params(axis="y", labelsize=9, colors=INK, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
fig.suptitle(f"대분류별 24시간 퍼널 전환율 (View 방문자 {MIN_VIEW:,}명 이상 대분류, View 많은 순)", x=0.01, ha="left",
             fontsize=14, color=INK)
fig.text(0.01, 0.01, "대분류 안 상품만으로 계산한 방문자 단위 퍼널 · 장바구니·구매 모두 그 대분류 첫 view로부터 24시간 이내 · "
         "단계 순서 강제 · 첫 view가 2015-09-14 21:00 UTC 전인 방문자 · 점선: 전체 퍼널", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print("저장:", FIG.relative_to(ROOT))
