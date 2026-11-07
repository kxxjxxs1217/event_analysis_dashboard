"""카테고리별 첫 상품 조회 집계.

첫 상품 조회 = 방문자별 가장 이른 view 이벤트 1건 (같은 시각이면 itemid가 작은 것)
카테고리는 07_join_category.py의 시점 기준 카테고리를 사용. 카테고리가 없는 상품은 '카테고리 없음'.
가장 앞선 4주에 첫 조회가 있는 방문자는 그 이전 조회가 관측되지 않았을 수 있어 비율을 함께 남긴다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_with_category.csv"
FIG = ROOT / "outputs" / "figures" / "first_view_by_category.png"
T_ROOT = ROOT / "outputs" / "tables" / "first_view_by_root_category.csv"
T_LEAF = ROOT / "outputs" / "tables" / "first_view_by_category.csv"
TOP_N = 20

ev = pd.read_csv(SRC, encoding="utf-8-sig", usecols=["timestamp", "visitorid", "event", "itemid", "is_first_4w",
                                                        "categoryid", "root_categoryid", "category_match"],
                 dtype={"categoryid": "Int64", "root_categoryid": "Int64"})
views = ev[ev["event"] == "view"].sort_values(["visitorid", "timestamp", "itemid"])
first = views.drop_duplicates("visitorid", keep="first")
ties = views.duplicated(["visitorid", "timestamp"], keep=False) & views.index.isin(first.index)
print("첫 조회 방문자:", len(first), "/ 같은 시각 동시 조회로 itemid 기준 선택:", int(ties.sum()))


def summarize(col):
    key = first[col].astype("string").fillna("카테고리 없음")
    t = first.groupby(key).agg(첫조회_방문자수=("visitorid", "size"),
                               첫4주_비율=("is_first_4w", "mean"),
                               첫스냅샷대체_비율=("category_match", lambda s: (s == "첫스냅샷대체").mean()))
    t["비율(%)"] = (t["첫조회_방문자수"] / len(first) * 100).round(2)
    t[["첫4주_비율", "첫스냅샷대체_비율"]] = (t[["첫4주_비율", "첫스냅샷대체_비율"]] * 100).round(1)
    t = t.rename(columns={"첫4주_비율": "첫4주 비율(%)", "첫스냅샷대체_비율": "첫스냅샷대체 비율(%)"})
    return t.sort_values("첫조회_방문자수", ascending=False)


by_root = summarize("root_categoryid")
by_leaf = summarize("categoryid")
assert by_root["첫조회_방문자수"].sum() == len(first) == by_leaf["첫조회_방문자수"].sum()
T_ROOT.parent.mkdir(parents=True, exist_ok=True)
by_root.to_csv(T_ROOT, encoding="utf-8-sig", index_label="root_categoryid")
by_leaf.to_csv(T_LEAF, encoding="utf-8-sig", index_label="categoryid")
print(by_root.to_string())
print(by_leaf.head(TOP_N).to_string())
print("카테고리 수(없음 제외):", len(by_leaf) - ("카테고리 없음" in by_leaf.index))

# 그래프
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURFACE, BLUE, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#a8a7a2"


def hbar(ax, t, title, prefix):
    t = t.iloc[::-1]
    labels = [k if k == "카테고리 없음" else f"{prefix} {k}" for k in t.index]
    colors = [GRAY if k == "카테고리 없음" else BLUE for k in t.index]
    ax.barh(labels, t["첫조회_방문자수"], color=colors, height=0.7)
    xmax = t["첫조회_방문자수"].max()
    for y, (n, p) in enumerate(zip(t["첫조회_방문자수"], t["비율(%)"])):
        ax.text(n + xmax * 0.01, y, f"{n:,} ({p:.1f}%)", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, xmax * 1.28)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=10)
    ax.set_facecolor(SURFACE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="x", labelsize=8, colors=INK2, length=0)
    ax.tick_params(axis="y", labelsize=9, colors=INK, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 9), facecolor=SURFACE)
hbar(a1, by_root, f"최상위 카테고리별 (전체 {len(by_root)}개 구분)", "root")
hbar(a2, by_leaf.drop(index="카테고리 없음", errors="ignore").head(TOP_N), f"세부 카테고리 상위 {TOP_N}개", "cat")
fig.suptitle(f"카테고리별 첫 상품 조회 방문자 수 (첫 조회 방문자 {len(first):,}명)", x=0.01, ha="left",
             fontsize=15, color=INK)
fig.text(0.01, 0.01, "첫 상품 조회: 방문자별 가장 이른 view 1건 · 카테고리는 조회 시점 기준(첫 스냅샷 이전은 첫 스냅샷 카테고리) · "
         "괄호는 전체 첫 조회 대비 비율 · 2015-05-03 ~ 2015-09-15 21:00 UTC · 5월 첫 4주 첫 조회는 그 이전 조회가 관측되지 않았을 수 있음",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150, facecolor=SURFACE)
print(f"저장: {FIG.relative_to(ROOT)}, {T_ROOT.relative_to(ROOT)}, {T_LEAF.relative_to(ROOT)}")
