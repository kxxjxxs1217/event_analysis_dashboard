"""구매 전환 대시보드 (Streamlit).

실행: streamlit run analysis/dashboard.py   (프로젝트 루트에서. Streamlit Community Cloud도 같은 경로)
입력: data/processed/events_dashboard.parquet (14_export_dashboard_data.py, 배포용 7개 열)
      없으면 data/processed/events_with_category.csv
      (events_clean.csv와 행·열이 같고 시점 기준 카테고리만 추가된 07_join_category.py 결과)
- 주요 지표 탭: 05_key_metrics.py, 06_monthly_trends.py와 같은 정의. 날짜와 월은 UTC 기준.
  카테고리별 전환율 산포도: 아래 첫 조회 성과 탭과 같은 정의로 카테고리마다 x = 장바구니 전환율, y = 구매 전환율,
  풍선 크기 = 첫 조회 방문자 수, 색 = 대분류.
- 카테고리별 첫 조회 성과 탭: 08_first_view_by_category.py와 같은 정의
  (첫 조회 = 방문자별 가장 이른 view 1건, 같은 시각이면 itemid가 작은 것).
  장바구니/구매 전환율 = 첫 조회 이후(같은 시각 포함) 관측 종료까지 어떤 상품이든 addtocart/transaction이 있는
  방문자 ÷ 첫 조회 방문자.
  30일 재방문율 = 첫 조회일(Day 0) 다음 날부터 30일 안(Day 1~30)에 어떤 이벤트든 다시 한 방문자
  ÷ 30일을 관측할 수 있는(Day 0 + 30 <= 2015-09-14) 첫 조회 방문자.
- 퍼널 탭: funnel_lib.py의 24시간 퍼널 (전체 / 대분류 / 카테고리 범위). 11_category_funnel.py와 같은 정의.
- 리텐션 탭: retention_lib.py (12_retention_curve.py, 13_retention_by_action.py와 같은 정의).
  재방문율(첫 이벤트 → 어떤 이벤트든), 재구매율(첫 구매 → 구매), 구매자 재방문율(첫 구매 → 어떤 이벤트든).
  카테고리 = Day 0 행동(첫 이벤트 / 첫 구매)의 상품 카테고리, 재방문은 카테고리와 무관하게 센다.
  Day 0이 첫 4주(2015-05-31 전)인 사용자는 실제 첫 행동이 아닐 수 있어 뺀다.
  이 탭은 전용 캘린더(Day 0 기간, 2015-05-31 ~ 2015-09-14)를 쓴다. 집계 단위(일·주·월)는 곡선과
  코호트 표(행 = 첫 방문 기간, 열 = N, 칸 = 재방문율과 재방문자 수 히트맵)에 적용된다.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import funnel_lib as fl  # noqa: E402
import retention_lib as rl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_with_category.csv"
EVENTS_PARQUET = ROOT / "data" / "processed" / "events_dashboard.parquet"  # 배포용 (저장소에 포함)
EVENT_COLS = ["timestamp", "visitorid", "event", "itemid", "transactionid", "categoryid", "root_categoryid"]
DATA_NOTICE = ("데이터: Retailrocket recommender system dataset (Kaggle, Retailrocket / Roman Zykov), "
               "CC BY-NC-SA 4.0 · 비상업적 교육 목적 · 원본을 정제(중복 제거, 기준 시점 적용)하고 "
               "상품 카테고리 이력을 결합한 가공 데이터")
NO_CAT = "카테고리 없음"
FV_COUNT = "첫 조회 방문자 수"
CUTOFF = pd.Timestamp("2015-09-15 21:00:00")  # UTC, 이 시각 전까지만 관측
METRICS = {
    "활성 사용자 수": ("명", "어떤 이벤트든 1회 이상 한 고유 방문자"),
    "구매자 수": ("명", "transaction을 1회 이상 한 고유 방문자"),
    "구매 횟수": ("회", "고유 주문(transactionid) 수, 주문일은 주문의 가장 이른 시각"),
    "장바구니에 담은 횟수": ("회", "addtocart 이벤트 수"),
    "이벤트 횟수": ("회", "view + addtocart + transaction 이벤트 수"),
}
BLUE = "#2a78d6"

st.set_page_config(page_title="구매 전환 대시보드", layout="wide")


@st.cache_resource(show_spinner="데이터를 불러오는 중...")
def load():  # 읽기 전용으로만 사용 (복사 없이 공유)
    if EVENTS_PARQUET.exists():
        ev = pd.read_parquet(EVENTS_PARQUET, columns=EVENT_COLS)
    else:
        ev = pd.read_csv(EVENTS, encoding="utf-8-sig", usecols=EVENT_COLS,
                         dtype={"transactionid": "Int64", "categoryid": "Int64", "root_categoryid": "Int64"})
    ts = pd.to_datetime(ev["timestamp"], unit="ms")  # UTC (시간대 변환 없음)
    ev["date"] = ts.dt.normalize()
    order_ts = ev[ev["event"] == "transaction"].groupby("transactionid")["timestamp"].min()
    orders = pd.DataFrame({"date": pd.to_datetime(order_ts, unit="ms").dt.normalize()})

    # 방문자별 첫 상품 조회 1건 + 그 이후(같은 시각 포함) 장바구니·구매 여부 (상품·카테고리 무관, 관측 종료까지)
    views = ev[ev["event"] == "view"].sort_values(["visitorid", "timestamp", "itemid"])
    fv = views.drop_duplicates("visitorid", keep="first").set_index("visitorid")

    def converted(event):
        e = ev.loc[ev["event"] == event, ["visitorid", "timestamp"]]
        e = e[e["visitorid"].isin(fv.index)]
        after = e["timestamp"].values >= fv.loc[e["visitorid"], "timestamp"].values
        return fv.index.isin(e.loc[after, "visitorid"].unique())

    first_views = pd.DataFrame({
        "date": fv["date"].values,
        "대분류": fv["root_categoryid"].astype("string").fillna(NO_CAT).values,
        "카테고리": fv["categoryid"].astype("string").fillna(NO_CAT).values,
        "상품": fv["itemid"].astype(str).values,
        "cart": converted("addtocart"),
        "buy": converted("transaction"),
    })

    # 리텐션: 활동일(어떤 이벤트든) / 구매일
    days = ev[["visitorid", "date"]].drop_duplicates()
    tx_days = ev.loc[ev["event"] == "transaction", ["visitorid", "date"]].drop_duplicates()
    first_views["ret30"] = rl.within(fv["date"], rl.returns(fv["date"], days)).values

    def cohort(rows):  # 방문자별 첫 행동 1건(같은 시각이면 itemid가 작은 것)의 날짜와 카테고리
        f = rows.sort_values(["visitorid", "timestamp", "itemid"]).drop_duplicates("visitorid").set_index("visitorid")
        return pd.DataFrame({"d0": f["date"], "대분류": f["root_categoryid"].astype("string").fillna(NO_CAT),
                             "카테고리": f["categoryid"].astype("string").fillna(NO_CAT)})

    visit, buy = cohort(ev), cohort(ev[ev["event"] == "transaction"])
    returns = {"재방문율": rl.returns(visit["d0"], days), "재구매율": rl.returns(buy["d0"], tx_days),
               "구매자 재방문율": rl.returns(buy["d0"], days)}
    for name, r in returns.items():
        c = visit if name == "재방문율" else buy
        c[name] = rl.within(c["d0"], r)
    retention = {"visit": visit, "buy": buy, "days": days, "tx_days": tx_days}

    funnels = fl.build_all(fl.prepare(ev))  # 24시간 퍼널: 전체 / 대분류 / 카테고리
    return ev[["date", "visitorid", "event"]], orders, ts.min(), first_views, funnels, retention


def summarize(ev, orders, ev_key, order_key):
    """기간 키별 지표. key가 None이면 선택 기간 전체."""
    tx = ev[ev["event"] == "transaction"]
    if ev_key is None:
        return pd.Series({
            "활성 사용자 수": ev["visitorid"].nunique(),
            "구매자 수": tx["visitorid"].nunique(),
            "구매 횟수": len(orders),
            "장바구니에 담은 횟수": int(ev["event"].eq("addtocart").sum()),
            "이벤트 횟수": len(ev),
        })
    return pd.DataFrame({
        "활성 사용자 수": ev.groupby(ev_key)["visitorid"].nunique(),
        "구매자 수": tx.groupby(ev_key[tx.index])["visitorid"].nunique(),
        "구매 횟수": orders.groupby(order_key).size(),
        "장바구니에 담은 횟수": ev[ev["event"] == "addtocart"].groupby(ev_key[ev["event"] == "addtocart"]).size(),
        "이벤트 횟수": ev.groupby(ev_key).size(),
    }).fillna(0).astype(int)


@st.cache_data(show_spinner="지표를 계산하는 중...")
def compute(start, end):
    ev, orders, data_start = load()[:3]
    ev = ev[(ev["date"] >= start) & (ev["date"] <= end)]
    orders = orders[(orders["date"] >= start) & (orders["date"] <= end)]

    total = summarize(ev, orders, None, None)
    daily = summarize(ev, orders, ev["date"], orders["date"])
    to_month = lambda s: s.dt.to_period("M").dt.to_timestamp()
    monthly = summarize(ev, orders, to_month(ev["date"]), to_month(orders["date"]))

    # 관측 시간(일): 선택 기간 ∩ 데이터 관측 구간
    lo, hi = max(start, data_start), min(end + pd.Timedelta(days=1), CUTOFF)

    def covered(p0, p1):
        return max((min(p1, hi) - max(p0, lo)).total_seconds() / 86400, 0)

    daily["관측 일수"] = [covered(d, d + pd.Timedelta(days=1)) for d in daily.index]
    monthly["관측 일수"] = [covered(m, m + pd.offsets.MonthBegin(1)) for m in monthly.index]
    monthly["달 일수"] = monthly.index.days_in_month
    return total, daily, monthly, data_start


def line_chart(df, col, unit, monthly):
    full = df["관측 일수"] >= (df["달 일수"] if monthly else 1) - 1e-9
    fig = go.Figure(go.Scatter(
        x=df.index, y=df[col], mode="lines+markers" if monthly else "lines",
        line=dict(color=BLUE, width=2),
        marker=dict(size=9, color=[BLUE if f else "white" for f in full],
                    line=dict(color=BLUE, width=2)) if monthly else None,
        customdata=df["관측 일수"].round(2),
        hovertemplate=("%{x|%Y-%m}" if monthly else "%{x|%Y-%m-%d (%a)}")
        + f"<br>{col}: %{{y:,}}{unit}<br>관측 일수: %{{customdata}}일<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{col} ({unit})", font=dict(size=15)),
        height=280, margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified" if not monthly else "closest",
        showlegend=False,
    )
    fig.update_yaxes(rangemode="tozero", tickformat=",")
    if monthly:
        fig.update_xaxes(dtick="M1", tickformat="%Y-%m")
    else:
        fig.update_xaxes(tickformat="%m-%d")
    return fig


def chart_grid(df, monthly):
    names = list(METRICS)
    for row in (names[:3], names[3:]):
        cols = st.columns(3)
        for c, name in zip(cols, row):
            with c:
                st.plotly_chart(line_chart(df, name, METRICS[name][0], monthly), width="stretch",
                                key=f"{'monthly' if monthly else 'daily'}-{name}")


# ---------- 공통: 지표 카드 ----------
KPI_CSS = """<style>
.kpi-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin:4px 0 8px;}
.kpi {border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:14px 16px;}
.kpi-label {font-size:.95rem; opacity:.8;}
.kpi-value {font-size:clamp(1.6rem, 2.4vw, 2.2rem); font-weight:700; line-height:1.3; white-space:nowrap;}
.kpi-unit {font-size:1rem; font-weight:500; margin-left:3px; opacity:.8;}
.kpi-desc {font-size:.78rem; opacity:.65; margin-top:4px;}
.kpi-compact {grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); gap:8px;}
.kpi-compact .kpi {padding:10px 12px;}
.kpi-compact .kpi-value {font-size:1.45rem;}
</style>"""


def kpi_cards(cards, compact=False):
    """cards: [(라벨, 값 문자열, 단위, 설명)]. compact이면 설명 없이 좁은 카드."""
    html = "".join(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
        + ("" if compact else f'<div class="kpi-desc">{desc}</div>') + "</div>"
        for label, value, unit, desc in cards
    )
    cls = "kpi-grid kpi-compact" if compact else "kpi-grid"
    st.markdown(KPI_CSS + f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


# ---------- 공통: 히트맵 표 ----------
# 단일 색상(파랑) 순차 척도: 값이 클수록 진하게. 열마다 따로 척도를 잡는다(열 안에서 비교).
HEAT_CMAP = LinearSegmentedColormap.from_list("heat_blue", ["#f2f7fd", "#9ec5f4", "#3987e5", "#104281"])
LABEL_COLS = {"순위": 44, "대분류": 90, "카테고리": 90, "상품": 70}  # 색을 입히지 않는 식별 열과 너비
MAX_STYLED_ROWS = 1000  # 색을 입혀 보여줄 최대 행 수 (그 이상은 상위만 표시)


def number_format(col):
    if col.endswith("(%)"):
        return "{:.2f}%"
    if col.endswith("(분)"):
        return "{:.1f}분"
    return "{:,.0f}"


def heat_table(t, height=None):
    """숫자 열에 열별 배경색 그라데이션(값이 클수록 진함)을 입힌 표. 결측은 '–'."""
    shown = t.head(MAX_STYLED_ROWS)
    num_cols = [c for c in shown.columns if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(shown[c])]
    styler = (shown.style
              .format({c: number_format(c) for c in num_cols}, na_rep="–")
              .background_gradient(cmap=HEAT_CMAP, subset=num_cols, axis=0))
    st.dataframe(
        styler, hide_index=True, width="stretch", height=height or "auto",
        column_config={c: st.column_config.Column(c, width=w) for c, w in LABEL_COLS.items() if c in shown.columns},
    )
    if len(t) > MAX_STYLED_ROWS:
        st.caption(f"전체 {len(t):,}행 중 상위 {MAX_STYLED_ROWS:,}행만 표시합니다. 색은 표시된 행 안에서 비교합니다.")


# ---------- 카테고리별 첫 조회 ----------
CART_RATE, BUY_RATE = "장바구니 전환율(%)", "구매 전환율(%)"
CART_N, BUY_N = "장바구니 전환 방문자", "구매 전환 방문자"
RET_RATE, RET_N = "30일 재방문율(%)", "30일 재방문 방문자"


def pct(s):
    """0/1(결측은 관측 불가) 열의 비율(%) 문자열과 단위. 관측 가능한 사용자가 없으면 '–'."""
    s = s.dropna()
    return (f"{s.mean() * 100:.2f}", "%") if len(s) else ("–", "")


def conversion_cards(df):
    n = len(df)
    r = df["ret30"].dropna()
    return [
        ("첫 조회 방문자", f"{n:,}", "명", "방문자별 첫 상품 조회 1건 기준"),
        ("장바구니 전환율", f"{df['cart'].mean() * 100:.2f}", "%",
         f"첫 조회 이후 장바구니에 담은 방문자 {int(df['cart'].sum()):,}명 ÷ 첫 조회 방문자"),
        ("구매 전환율", f"{df['buy'].mean() * 100:.2f}", "%",
         f"첫 조회 이후 구매한 방문자 {int(df['buy'].sum()):,}명 ÷ 첫 조회 방문자"),
        ("30일 재방문율", *pct(df["ret30"]),
         f"첫 조회 다음 날부터 30일 안에 다시 온 방문자 {int(r.sum()):,}명 ÷ 30일 관측 가능한 첫 조회 방문자 {len(r):,}명"),
    ]


def rank_table(df, col, n=None):
    """col 기준 첫 조회 방문자 수 순위와 전환율. 비중은 df 전체(상위 단계) 대비."""
    g = df.groupby(col)
    t = pd.DataFrame({FV_COUNT: g.size(), CART_N: g["cart"].sum(), BUY_N: g["buy"].sum(),
                      RET_N: g["ret30"].sum(), RET_RATE: g["ret30"].mean() * 100}).reset_index()
    t = t.sort_values([FV_COUNT, col], ascending=[False, True]).head(n or len(t))
    t["비중(%)"] = t[FV_COUNT] / len(df) * 100
    t[CART_RATE] = t[CART_N] / t[FV_COUNT] * 100
    t[BUY_RATE] = t[BUY_N] / t[FV_COUNT] * 100
    t.insert(0, "순위", range(1, len(t) + 1))
    return t[["순위", col, FV_COUNT, "비중(%)", CART_N, CART_RATE, BUY_N, BUY_RATE, RET_N, RET_RATE]
             ].reset_index(drop=True)


def hbar(t, col, title, key, top=20):
    t = t.head(top)
    fig = go.Figure(go.Bar(
        x=t[FV_COUNT], y=t[col], orientation="h", marker_color=BLUE,
        customdata=t[["비중(%)", CART_RATE, BUY_RATE, RET_RATE]].round(2).fillna("–").values,
        text=[f"{n:,}" for n in t[FV_COUNT]], textposition="outside", cliponaxis=False,
        hovertemplate=(f"{col} %{{y}}<br>{FV_COUNT}: %{{x:,}}명<br>비중: %{{customdata[0]}}%"
                       "<br>장바구니 전환율: %{customdata[1]}%<br>구매 전환율: %{customdata[2]}%"
                       "<br>30일 재방문율: %{customdata[3]}%<extra></extra>"),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=60 + 26 * len(t),
                      margin=dict(l=10, r=40, t=45, b=10), showlegend=False)
    fig.update_yaxes(type="category", autorange="reversed", title=None)
    fig.update_xaxes(tickformat=",", rangemode="tozero")
    st.plotly_chart(fig, width="stretch", key=key)


def lab(kind, value):
    return value if value == NO_CAT else f"{kind} {value}"


def category_scatter(start, end):
    """카테고리별 x = 장바구니 전환율, y = 구매 전환율, 풍선 크기 = 첫 조회 방문자 수, 색 = 대분류."""
    fv_all = load()[3]
    fv = fv_all[(fv_all["date"] >= start) & (fv_all["date"] <= end)]
    if fv.empty:
        st.info("선택한 기간에 첫 조회가 없습니다.")
        return
    g = fv.groupby(["대분류", "카테고리"])
    t = pd.DataFrame({FV_COUNT: g.size(), CART_RATE: g["cart"].mean() * 100,
                      BUY_RATE: g["buy"].mean() * 100}).reset_index()
    min_n = st.number_input("최소 첫 조회 방문자 수", min_value=1, value=100, step=50, key="scatter-min",
                            help="첫 조회가 적은 카테고리는 전환율이 0%나 100% 같은 극단값이 되기 쉬워 이 수 미만은 그리지 않습니다.")
    shown = t[t[FV_COUNT] >= min_n]
    st.caption(
        "카테고리별 첫 조회 성과 탭과 같은 정의 · 첫 조회 = 방문자별 가장 이른 view 1건(방문자마다 1건이라 첫 조회 수 = 첫 조회 방문자 수) · "
        "전환율 = 첫 조회 이후 관측 종료까지 어떤 상품이든 장바구니에 담거나 구매한 방문자 ÷ 첫 조회 방문자 · "
        "풍선 크기(면적) = 첫 조회 방문자 수 · 색 = 대분류(범례를 누르면 숨기기, 두 번 누르면 그 대분류만 보기) · "
        f"표시 {len(shown):,}개 카테고리 / 전체 {len(t):,}개 (첫 조회 방문자의 {shown[FV_COUNT].sum() / len(fv) * 100:.1f}%)"
    )
    if shown.empty:
        st.info("조건에 맞는 카테고리가 없습니다. 최소 첫 조회 방문자 수를 낮춰 보세요.")
        return

    roots = shown.groupby("대분류")[FV_COUNT].sum().sort_values(ascending=False).index
    colors = iter(qualitative.Alphabet)  # 26색: 대분류 23개 + 여유
    sizeref = 2 * shown[FV_COUNT].max() / 60 ** 2  # 가장 큰 풍선 지름 약 60px
    fig = go.Figure()
    for root in roots:
        d = shown[shown["대분류"] == root]
        fig.add_trace(go.Scatter(
            x=d[CART_RATE], y=d[BUY_RATE], mode="markers", name=lab("대분류", root),
            marker=dict(size=d[FV_COUNT], sizemode="area", sizeref=sizeref, sizemin=3,
                        color="#9a9893" if root == NO_CAT else next(colors), opacity=0.75,
                        line=dict(width=0.5, color="rgba(255,255,255,0.6)")),
            customdata=d[["카테고리", FV_COUNT]].values,
            hovertemplate=(f"{lab('대분류', root)} · 카테고리 %{{customdata[0]}}<br>첫 조회 방문자: %{{customdata[1]:,}}명"
                           "<br>장바구니 전환율: %{x:.2f}%<br>구매 전환율: %{y:.2f}%<extra></extra>"),
        ))
    fig.update_layout(title=dict(text=f"카테고리별 장바구니 전환율 × 구매 전환율 ({start:%Y-%m-%d} ~ {end:%Y-%m-%d})",
                                 font=dict(size=14)),
                      height=620, margin=dict(l=10, r=10, t=45, b=10), legend=dict(title="대분류"))
    fig.update_xaxes(title="장바구니 전환율 (%)", ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(title="구매 전환율 (%)", ticksuffix="%", rangemode="tozero")
    st.plotly_chart(fig, width="stretch", key="category-scatter")


def first_view_tab(start, end):
    fv_all = load()[3]
    fv = fv_all[(fv_all["date"] >= start) & (fv_all["date"] <= end)]
    if fv.empty:
        st.info("선택한 기간에 첫 조회가 없습니다.")
        return

    st.subheader(f"주요 성과 · {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")
    kpi_cards(conversion_cards(fv))
    st.caption(
        "첫 조회: 방문자별 가장 이른 view 1건(전체 관측 기간 기준)이 선택 기간에 있으면 집계 · "
        "전환율: 첫 조회 이후(같은 시각 포함) 관측 종료(2015-09-15 21:00 UTC)까지 어떤 상품이든 장바구니에 담거나 구매한 "
        "방문자 ÷ 첫 조회 방문자 (첫 조회가 늦을수록 전환할 시간이 짧음) · "
        "카테고리는 조회 시점 기준(첫 스냅샷 이전 조회는 첫 스냅샷 카테고리) · "
        f"'{NO_CAT}'은 카테고리 이력이 없는 상품 · 5월 첫 4주의 첫 조회는 그 이전 조회가 관측되지 않았을 수 있음 · "
        "비중은 선택 기간 전체 첫 조회 대비 · 첫 조회가 적은 항목의 전환율은 변동이 큼 · "
        "30일 재방문율: 첫 조회일(Day 0) 다음 날부터 30일 안(Day 1~30)에 어떤 이벤트든 다시 한 방문자 ÷ "
        "30일을 관측할 수 있는 첫 조회 방문자(Day 0이 2015-08-15 이전, 9/15는 하루가 온전하지 않아 관측 마지막 날은 9/14) · "
        "30일 재방문율의 분모는 첫 조회 방문자 수가 아니라 30일 관측 가능한 방문자"
    )
    items = rank_table(fv, "상품", 10)
    top_items = fv[fv["상품"].isin(items["상품"])]
    item_cat = top_items.groupby("상품")["카테고리"].agg(lambda s: s.mode().iloc[0])
    items = items.join(item_cat, on="상품")
    items = items[["순위", "상품", "카테고리"] + [c for c in items.columns if c not in ("순위", "상품", "카테고리")]]

    for title, t in [
        ("대분류 기준 첫 조회 상위 10", rank_table(fv, "대분류", 10)),
        ("카테고리 기준 첫 조회 상위 10", rank_table(fv, "카테고리", 10)),
        ("상품 기준 첫 조회 상위 10", items),
    ]:
        st.markdown(f"**{title}**")
        heat_table(t)

    st.divider()
    st.subheader("대분류 → 카테고리 → 상품 살펴보기")
    roots = rank_table(fv, "대분류")
    left, right = st.columns(2)
    with left:
        root_n = dict(zip(roots["대분류"], roots[FV_COUNT]))
        sel_root = st.selectbox("대분류 선택", roots["대분류"],
                                format_func=lambda r: f"{r} · {root_n[r]:,}명 ({root_n[r] / len(fv) * 100:.1f}%)")
        sub = fv[fv["대분류"] == sel_root]
        kpi_cards(conversion_cards(sub), compact=True)
        cats = rank_table(sub, "카테고리")
        hbar(cats, "카테고리", f"{lab('대분류', sel_root)}의 카테고리 (상위 {min(20, len(cats))}개 / 전체 {len(cats)}개)",
             key="bar-cat")
        with st.expander(f"카테고리 전체 표 ({len(cats)}개, 비중은 {lab('대분류', sel_root)} 대비)"):
            heat_table(cats, height=min(400, 38 + 35 * len(cats)))
    with right:
        cat_n = dict(zip(cats["카테고리"], cats[FV_COUNT]))
        sel_cat = st.selectbox("카테고리 선택", cats["카테고리"],
                               format_func=lambda c: f"{c} · {cat_n[c]:,}명 ({cat_n[c] / len(sub) * 100:.1f}%)")
        cat_sub = sub[sub["카테고리"] == sel_cat]
        kpi_cards(conversion_cards(cat_sub), compact=True)
        prods = rank_table(cat_sub, "상품")
        hbar(prods, "상품", f"{lab('카테고리', sel_cat)}의 상품 (상위 {min(20, len(prods))}개 / 전체 {len(prods):,}개)",
             key="bar-item")
        with st.expander(f"상품 전체 표 ({len(prods):,}개, 비중은 {lab('카테고리', sel_cat)} 대비)"):
            heat_table(prods, height=min(400, 38 + 35 * min(len(prods), MAX_STYLED_ROWS)))


# ---------- 퍼널 (24시간) ----------
def funnel_fig(counts, title):
    text = [f"{counts[0]:,}명"]
    for i in (1, 2):
        prev = f"{counts[i] / counts[i - 1] * 100:.2f}%" if counts[i - 1] else "–"
        text.append(f"{counts[i]:,}명 · 전 단계 대비 {prev}")
    fig = go.Figure(go.Funnel(
        y=fl.STAGES, x=counts, text=text, textinfo="text",
        textposition=["inside", "outside", "outside"], marker=dict(color=BLUE),
        connector=dict(fillcolor="rgba(42,120,214,0.15)", line=dict(width=0)),
        hovertemplate="%{y}: %{x:,}명<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=330,
                      margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    return fig


def dist_bar(d, title, key):
    fig = go.Figure(go.Bar(
        x=d["구간"], y=d["방문자 수"], marker_color=BLUE,
        text=[f"{p:.1f}%" for p in d["비율(%)"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:,}명<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=13)), height=250,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    fig.update_xaxes(type="category", tickfont=dict(size=10))
    fig.update_yaxes(tickformat=",", rangemode="tozero")
    st.plotly_chart(fig, width="stretch", key=key)


def funnel_panel(f, name, key):
    counts = fl.stage_counts(f)
    if counts[0] == 0:
        st.info("선택한 기간에 해당 범위의 View 방문자가 없습니다.")
        return
    rate = lambda a, b: f"{a / b * 100:.2f}" if b else "–"
    kpi_cards([
        ("View", f"{counts[0]:,}", "명", ""),
        ("View→AddToCart", rate(counts[1], counts[0]), "%", ""),
        ("AddToCart→Transaction", rate(counts[2], counts[1]), "%", ""),
        ("View→Transaction", rate(counts[2], counts[0]), "%", ""),
    ], compact=True)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.plotly_chart(funnel_fig(counts, f"{name} 퍼널 (24시간 이내)"), width="stretch", key=f"{key}-funnel")
    with c2:
        dist_bar(fl.time_dist(f, "to_cart_min"), f"View → AddToCart 소요 시간 (n={counts[1]:,})", f"{key}-t1")
        dist_bar(fl.time_dist(f, "to_tx_min"), f"AddToCart → Transaction 소요 시간 (n={counts[2]:,})", f"{key}-t2")
    with c3:
        dist_bar(fl.count_dist(f, "n_view_before_cart"), f"AddToCart 전 View 횟수 (n={counts[1]:,})", f"{key}-n1")
        dist_bar(fl.count_dist(f, "n_cart_before_tx"), f"Transaction 전 AddToCart 횟수 (n={counts[2]:,})", f"{key}-n2")


def funnel_tab(start, end):
    fz = load()[4]
    flt = lambda f: f[(f["v_date"] >= start) & (f["v_date"] <= end)]
    st.caption(
        "범위(전체·대분류·카테고리) 안 상품 이벤트만으로 만든 방문자 단위 퍼널 · View 기준 시각 = 범위 안 첫 view · "
        "AddToCart: 첫 view 이후 24시간 이내 장바구니 · Transaction: 그 첫 장바구니 이후, 첫 view로부터 24시간 이내 구매 · "
        "첫 view가 선택 기간 안이고 2015-09-14 21:00 UTC 전인 방문자(24시간 관측 창 보장) · "
        "소요 시간: View→AddToCart는 첫 view~첫 장바구니, AddToCart→Transaction은 첫 장바구니~첫 구매 · "
        "이전 단계 횟수: 첫 view~첫 장바구니 사이 view 수, 첫 장바구니~첫 구매 사이 장바구니 수(양 끝 포함) · "
        "View가 적은 범위의 전환율은 변동이 큼"
    )
    st.subheader("전체 퍼널")
    funnel_panel(flt(fz["전체"]), "전체", "fz-all")

    st.divider()
    st.subheader("대분류별 퍼널")
    roots_f = flt(fz["대분류"])
    t_root = fl.summary_table(roots_f, "대분류")
    if t_root.empty:
        st.info("선택한 기간에 대분류 퍼널 데이터가 없습니다.")
        return
    heat_table(t_root, height=min(420, 38 + 35 * len(t_root)))
    root_n = dict(zip(t_root["대분류"], t_root["View"]))
    sel_root = st.selectbox("대분류 선택", t_root["대분류"], key="fz-root",
                            format_func=lambda r: f"{r} · View {root_n[r]:,}명")
    funnel_panel(roots_f[roots_f["대분류"] == sel_root], lab("대분류", sel_root), "fz-root")

    st.divider()
    st.subheader(f"카테고리별 퍼널 · {lab('대분류', sel_root)}")
    cats_f = flt(fz["카테고리"])
    cats_f = cats_f[cats_f["대분류"] == sel_root]
    t_cat = fl.summary_table(cats_f, "카테고리")
    heat_table(t_cat, height=min(420, 38 + 35 * min(len(t_cat), MAX_STYLED_ROWS)))
    cat_n = dict(zip(t_cat["카테고리"], t_cat["View"]))
    sel_cat = st.selectbox("카테고리 선택", t_cat["카테고리"], key=f"fz-cat-{sel_root}",
                           format_func=lambda c: f"{c} · View {cat_n[c]:,}명")
    funnel_panel(cats_f[cats_f["카테고리"] == sel_cat], lab("카테고리", sel_cat), "fz-cat")


# ---------- 리텐션 ----------
# 지표: (Day 0 집단, 설명, 선 색)
RET_METRICS = {
    "재방문율": ("visit", "첫 이벤트 → 어떤 이벤트든 재방문", BLUE),
    "재구매율": ("buy", "첫 구매 → 다시 구매", "#d95f02"),
    "구매자 재방문율": ("buy", "첫 구매 → 어떤 이벤트든 재방문", "#1b9e77"),
}
COHORT_N = {"visit": "첫 방문 사용자 수", "buy": "첫 구매 사용자 수"}


def retention_cards(subs, compact=False):
    cards = []
    for name, (grp, desc, _) in RET_METRICS.items():
        s = subs[grp][name].dropna()
        cards.append((f"30일 {name}", *pct(s),
                      f"{desc} · Day 1~30 안에 {int(s.sum()):,}명 ÷ 30일 관측 가능한 {len(s):,}명"))
    return cards if compact else [(COHORT_N["visit"], f"{len(subs['visit']):,}", "명", "Day 0 = 첫 이벤트일"),
                                  (COHORT_N["buy"], f"{len(subs['buy']):,}", "명", "Day 0 = 첫 구매일")] + cards


RET_MIN, RET_MAX = rl.FIRST_4W_END.date(), rl.LAST_FULL_DAY.date()  # 리텐션 탭에서 고를 수 있는 Day 0 범위
UNIT_KO = {"D": "일별", "W": "주별", "M": "월별"}
X_TITLE = {"D": "N-Day (Day 0로부터 지난 일수)", "W": "N-Week (Day 0가 속한 주로부터 지난 주 수, 월요일 시작)",
           "M": "N-Month (Day 0가 속한 달로부터 지난 개월 수)"}
COHORT_FMT = {"D": "%Y-%m-%d", "W": "%Y-%m-%d 주", "M": "%Y-%m"}


@st.cache_data(show_spinner="리텐션을 계산하는 중...", max_entries=64)
def ret_matrix(metric, unit, start, end, root=None, cat=None):
    """지표·단위·Day 0 기간·범위(대분류/카테고리)별 코호트 표 (retention_lib.cohort_matrix)."""
    data = load()[5]
    c = data[RET_METRICS[metric][0]]
    c = c[(c["d0"] >= start) & (c["d0"] <= end)]
    if root is not None:
        c = c[c["대분류"] == root]
    if cat is not None:
        c = c[c["카테고리"] == cat]
    return rl.cohort_matrix(c["d0"], data["tx_days" if metric == "재구매율" else "days"], unit)


def retention_fig(scope, title):
    """scope = (단위, Day 0 시작, Day 0 끝, 대분류, 카테고리)."""
    unit = scope[0]
    u = rl.UNIT_LABEL[unit]
    fig = go.Figure()
    for name, (grp, desc, color) in RET_METRICS.items():
        sizes, counts = ret_matrix(name, *scope)
        c = rl.curve(sizes, counts, rl.MAX_N if unit == "D" else None)
        fig.add_trace(go.Scatter(
            x=c["N"], y=c["재방문율(%)"], mode="lines+markers", name=f"{name} (n={int(sizes.sum()):,})",
            line=dict(color=color, width=2), marker=dict(size=5),
            customdata=c[["재방문자 수", "대상 사용자 수"]].values,
            hovertemplate=f"{name} {u} %{{x}}: %{{y:.2f}}%<br>%{{customdata[0]:,}}명 ÷ %{{customdata[1]:,}}명<extra></extra>",
        ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=360, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1), hovermode="x unified")
    fig.update_xaxes(title=X_TITLE[unit], dtick=7 if unit == "D" else 1, tick0=0,
                     range=[0.5, rl.MAX_N + 0.5] if unit == "D" else None)
    fig.update_yaxes(title=f"{u} N 재방문율 (%)", rangemode="tozero", ticksuffix="%")
    return fig


def retention_panel(subs, name, key, scope, compact=True):
    if subs["visit"].empty and subs["buy"].empty:
        st.info("선택한 기간에 해당 범위의 사용자가 없습니다.")
        return
    kpi_cards(retention_cards(subs, compact), compact=compact)
    u = rl.UNIT_LABEL[scope[0]]
    st.plotly_chart(retention_fig(scope, f"{name} N-{u} 리텐션 곡선 ({UNIT_KO[scope[0]]}, {u} 0 = 100%)"),
                    width="stretch", key=key)


def heat_css(x):
    """0~1 값을 히트맵 배경색과 글자색 CSS로. 결측은 스타일 없음."""
    if pd.isna(x):
        return ""
    r, g, b, _ = HEAT_CMAP(min(max(x, 0), 1))
    text = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "#0b0b0b"
    return f"background-color: {to_hex((r, g, b))}; color: {text}"


def cohort_table(unit, start, end):
    """재방문율 코호트 히트맵: 행 = 첫 방문 기간, 열 = N, 칸 = 재방문율 (재방문자 수)."""
    sizes, counts = ret_matrix("재방문율", unit, start, end)
    if sizes.empty:
        st.info("선택한 기간에 해당하는 코호트가 없습니다. 주별은 6/1 주, 월별은 6월 코호트부터 있습니다.")
        return
    if unit == "D":
        counts = counts.loc[:, :rl.MAX_N]
    rates = counts.div(sizes, axis=0) * 100
    u = rl.UNIT_LABEL[unit]
    cols = [f"{u} {n}" for n in counts.columns]
    t = pd.DataFrame([["" if pd.isna(c) else f"{r:.2f}% ({c:,.0f}명)" for r, c in zip(rr, cc)]
                      for rr, cc in zip(rates.values, counts.values)], columns=cols)
    t.insert(0, "사용자 수", sizes.values)
    t.insert(0, "코호트", [rl.period_start(p, unit).strftime(COHORT_FMT[unit]) for p in sizes.index])

    later = rates.iloc[:, 1:]  # N = 0은 항상 100%라 색 척도에서 뺀다
    vmax = np.nanmax(later.values) if later.notna().any().any() else 1
    css = pd.DataFrame("", index=t.index, columns=cols)
    for j, col in enumerate(cols[1:], start=1):
        css[col] = [heat_css(v / vmax) for v in rates.iloc[:, j].values]
    styler = t.style.apply(lambda _: css, axis=None, subset=cols).format({"사용자 수": "{:,}"})
    st.dataframe(styler, hide_index=True, width="stretch", height=min(640, 38 + 35 * len(t)),
                 column_config={"코호트": st.column_config.Column(pinned=True, width=110),
                                "사용자 수": st.column_config.Column(pinned=True)})


def retention_table(subs, col):
    """col별 사용자 수와 30일 이내 재방문율·재구매율·구매자 재방문율 (첫 방문 사용자 수 내림차순)."""
    v, b = subs["visit"].groupby(col), subs["buy"].groupby(col)
    t = pd.DataFrame({
        COHORT_N["visit"]: v.size(), "30일 재방문율(%)": v["재방문율"].mean() * 100,
        COHORT_N["buy"]: b.size(), "30일 재구매율(%)": b["재구매율"].mean() * 100,
        "30일 구매자 재방문율(%)": b["구매자 재방문율"].mean() * 100,
    })
    t[list(COHORT_N.values())] = t[list(COHORT_N.values())].fillna(0)
    t = t.rename_axis(col).reset_index().sort_values([COHORT_N["visit"], col], ascending=[False, True])
    t.insert(0, "순위", range(1, len(t) + 1))
    return t.reset_index(drop=True)


def retention_tab():
    data = load()[5]
    c1, c2 = st.columns([1, 2])
    with c1:
        sel = st.date_input("Day 0 기간 선택 (UTC)", value=(RET_MIN, RET_MAX), min_value=RET_MIN, max_value=RET_MAX,
                            format="YYYY-MM-DD", key="ret-dates")
    with c2:
        unit = rl.UNITS[st.radio("집계 단위", list(rl.UNITS), horizontal=True, key="ret-unit")]
    st.caption(
        "이 탭은 맨 위 '기간 선택' 대신 이 캘린더를 씁니다 · 선택한 기간에 Day 0(첫 이벤트일 / 첫 구매일)이 있는 사용자만 · "
        "고를 수 있는 범위는 2015-05-31 \\~ 2015-09-14 (첫 4주는 이전 행동이 관측되지 않아 제외, 9/15는 21시 전까지만 있어 제외) · "
        "재방문율: Day 0 = 첫 이벤트일, 어떤 이벤트든 다시 하면 재방문 · 재구매율: Day 0 = 첫 구매일, 다시 구매하면 재방문 · "
        "구매자 재방문율: Day 0 = 첫 구매일, 어떤 이벤트든 다시 하면 재방문 · 날짜는 UTC · "
        "카테고리 = Day 0 행동(첫 이벤트 / 첫 구매) 상품의 카테고리, 재방문은 카테고리와 무관 · "
        "집계 단위는 곡선과 코호트 표에 적용: 코호트 = Day 0이 속한 일·주(월요일 시작)·월, "
        "N번째 기간에 한 번이라도 재방문한 사용자 ÷ 코호트 사용자(누적 아님), 끝까지 관측되지 않은 기간은 빼거나 비워 둠 "
        "(마지막 완전한 주 9/7\\~9/13, 마지막 완전한 달 8월) · 주별·월별은 시작일이 5/31 이후인 6/1 주, 6월 코호트부터 · "
        "선택 기간이 주·월 중간에서 시작하거나 끝나면 그 주·월 코호트는 선택 기간에 Day 0이 있는 사용자만 · "
        "30일 지표와 대분류·카테고리 표는 단위와 무관하게 Day 1\\~30 중 하루라도 재방문한 사용자 ÷ 30일을 관측할 수 있는 사용자"
        "(Day 0이 2015-08-15 이전) · 사용자가 적은 범위(특히 구매자)는 값의 변동이 큼"
    )
    if len(sel) != 2:
        st.info("캘린더에서 종료일까지 선택해 주세요.")
        return
    start, end = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])
    subs = {g: data[g][(data[g]["d0"] >= start) & (data[g]["d0"] <= end)] for g in ("visit", "buy")}
    st.subheader(f"주요 지표 · Day 0 {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")
    retention_panel(subs, "전체", "ret-all", (unit, start, end, None, None), compact=False)

    st.divider()
    st.subheader(f"리텐션 코호트 · 재방문율 ({UNIT_KO[unit]})")
    u = rl.UNIT_LABEL[unit]
    st.caption(
        f"행 = 첫 방문(첫 이벤트)이 속한 {UNIT_KO[unit][0]}, 열 = {u} N, 칸 = 그 기간에 재방문한 사용자 비율 (재방문자 수) · "
        f"{u} 0은 첫 방문 기간이라 항상 100% · 색이 진할수록 재방문율이 높음 ({u} 0은 색 척도에서 제외, 표 전체가 같은 척도) · "
        "빈칸은 아직 끝까지 관측되지 않은 기간" + (f" · 일별은 Day {rl.MAX_N}까지 표시" if unit == "D" else "")
    )
    cohort_table(unit, start, end)

    st.divider()
    st.subheader("대분류별 리텐션")
    t_root = retention_table(subs, "대분류")
    heat_table(t_root, height=min(420, 38 + 35 * len(t_root)))
    root_n = dict(zip(t_root["대분류"], t_root[COHORT_N["visit"]]))
    sel_root = st.selectbox("대분류 선택", t_root["대분류"], key="ret-root",
                            format_func=lambda r: f"{r} · 첫 방문 {root_n[r]:,.0f}명")
    in_root = {g: s[s["대분류"] == sel_root] for g, s in subs.items()}
    retention_panel(in_root, lab("대분류", sel_root), "ret-root-chart", (unit, start, end, sel_root, None))

    st.divider()
    st.subheader(f"카테고리별 리텐션 · {lab('대분류', sel_root)}")
    t_cat = retention_table(in_root, "카테고리")
    heat_table(t_cat, height=min(420, 38 + 35 * min(len(t_cat), MAX_STYLED_ROWS)))
    cat_n = dict(zip(t_cat["카테고리"], t_cat[COHORT_N["visit"]]))
    sel_cat = st.selectbox("카테고리 선택", t_cat["카테고리"], key=f"ret-cat-{sel_root}",
                           format_func=lambda c: f"{c} · 첫 방문 {cat_n[c]:,.0f}명")
    retention_panel({g: s[s["카테고리"] == sel_cat] for g, s in in_root.items()}, lab("카테고리", sel_cat), "ret-cat-chart",
                    (unit, start, end, sel_root, sel_cat))


# ---------------- 화면 ----------------
data_start = load()[2]
min_d = data_start.date()
max_d = (CUTOFF - pd.Timedelta(milliseconds=1)).date()

st.title("구매 전환 대시보드")
c1, c2 = st.columns([1, 2])
with c1:
    sel = st.date_input("기간 선택 (UTC)", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                        format="YYYY-MM-DD")
with c2:
    st.caption(
        f"관측 기간: {data_start:%Y-%m-%d %H:%M} ~ {CUTOFF:%Y-%m-%d %H:%M} 전 (UTC) · 완전 중복 제거 · "
        "5/3은 03시부터, 9/15는 21시 전까지만 관측 · 5월(첫 4주)은 이전 행동이 관측되지 않음"
    )
    st.caption(DATA_NOTICE)
if len(sel) != 2:
    st.info("캘린더에서 종료일까지 선택해 주세요.")
    st.stop()

start, end = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])
total, daily, monthly, _ = compute(start, end)

tab_main, tab_cat, tab_funnel, tab_ret = st.tabs(["주요 지표", "카테고리별 첫 조회 성과", "퍼널", "리텐션"])
with tab_main:
    st.subheader(f"주요 지표 · {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")
    kpi_cards([(name, f"{int(total[name]):,}", unit, desc) for name, (unit, desc) in METRICS.items()])

    st.divider()
    st.subheader("일자별 추이")
    chart_grid(daily, monthly=False)

    st.subheader("월별 추이")
    st.caption("빈 원은 부분 월(선택 기간이나 관측 구간 때문에 한 달 전체가 들어가지 않은 달)입니다. "
               "활성 사용자·구매자는 기간별 고유 방문자라 일·월 값을 더하면 전체 값보다 큽니다.")
    chart_grid(monthly, monthly=True)

    st.divider()
    st.subheader("카테고리별 전환율 산포도")
    category_scatter(start, end)

with tab_cat:
    first_view_tab(start, end)

with tab_funnel:
    funnel_tab(start, end)

with tab_ret:
    retention_tab()
