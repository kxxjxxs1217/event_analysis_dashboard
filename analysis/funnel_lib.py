"""24시간 퍼널 계산 (11_category_funnel.py와 dashboard.py가 같이 사용).

범위(scope) 안의 이벤트만으로 방문자 단위 퍼널을 만든다. 범위 = 전체 / 대분류 / 카테고리.
- View: 범위 안 상품을 조회한 방문자. 기준 시각 = 범위 안 첫 view
- AddToCart: 첫 view 이후(같은 시각 포함) 24시간 이내 범위 안 상품을 장바구니에 담은 방문자 (첫 장바구니 시각 사용)
- Transaction: 그 첫 장바구니 이후, 첫 view로부터 24시간 이내 범위 안 상품을 구매한 방문자
- 코호트: 첫 view가 기준 시점(2015-09-15 21:00 UTC) 24시간 전보다 이른 방문자만 (24시간 관측 창 보장)
- 이전 단계 수행 횟수: AddToCart 전환자의 첫 view~첫 장바구니 사이 view 수(양 끝 포함),
  Transaction 전환자의 첫 장바구니~첫 구매 사이 장바구니 수(양 끝 포함). 모두 범위 안 이벤트만 센다.
카테고리는 07_join_category.py의 이벤트 시점 기준 카테고리이고, 없으면 '카테고리 없음'.
"""
import pandas as pd

WINDOW_MS = 24 * 3600 * 1000
CUTOFF_MS = pd.Timestamp("2015-09-15 21:00:00").value // 10**6  # UTC
NO_CAT = "카테고리 없음"
STAGES = ["View", "AddToCart", "Transaction"]

TIME_BINS = [0, 1, 5, 15, 30, 60, 180, 360, 720, 1440.0001]  # 분
TIME_LABELS = ["1분 미만", "1~5분", "5~15분", "15~30분", "30분~1시간", "1~3시간", "3~6시간", "6~12시간", "12~24시간"]
COUNT_BINS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5, 20.5, float("inf")]  # 정수 횟수 경계
COUNT_LABELS = ["1회", "2회", "3회", "4회", "5회", "6~10회", "11~20회", "21회 이상"]


def load_events(path):
    ev = pd.read_csv(path, encoding="utf-8-sig",
                     usecols=["timestamp", "visitorid", "event", "categoryid", "root_categoryid"],
                     dtype={"categoryid": "Int64", "root_categoryid": "Int64"})
    return prepare(ev)


def prepare(ev):
    """timestamp, visitorid, event, categoryid, root_categoryid → 퍼널 범위 키(전체/대분류/카테고리) 추가"""
    return pd.DataFrame({
        "timestamp": ev["timestamp"], "visitorid": ev["visitorid"], "event": ev["event"], "전체": "전체",
        "대분류": ev["root_categoryid"].astype("string").fillna(NO_CAT),
        "카테고리": ev["categoryid"].astype("string").fillna(NO_CAT),
    })


def build_all(ev):
    """dashboard용: 전체 / 대분류 / (대분류, 카테고리) 범위 퍼널"""
    return {"전체": build(ev, ["전체"]), "대분류": build(ev, ["대분류"]), "카테고리": build(ev, ["대분류", "카테고리"])}


def build(ev, keys, window=WINDOW_MS, cutoff_ms=CUTOFF_MS):
    """keys별 방문자 단위 퍼널 행. 반환 열: keys, v_date, to_cart_min, to_tx_min, n_view_before_cart, n_cart_before_tx"""
    g = list(keys) + ["visitorid"]
    sel = {e: ev.loc[ev["event"] == e, g + ["timestamp"]] for e in ("view", "addtocart", "transaction")}

    f = sel["view"].groupby(g, sort=False)["timestamp"].min().rename("v_ts").reset_index()
    f = f[f["v_ts"] < cutoff_ms - window]

    c = sel["addtocart"].merge(f, on=g)
    c = c[(c["timestamp"] >= c["v_ts"]) & (c["timestamp"] <= c["v_ts"] + window)]
    f = f.merge(c.groupby(g, sort=False)["timestamp"].min().rename("c_ts").reset_index(), on=g, how="left")

    conv = f.loc[f["c_ts"].notna(), g + ["v_ts", "c_ts"]]
    t = sel["transaction"].merge(conv, on=g)
    t = t[(t["timestamp"] >= t["c_ts"]) & (t["timestamp"] <= t["v_ts"] + window)]
    f = f.merge(t.groupby(g, sort=False)["timestamp"].min().rename("t_ts").reset_index(), on=g, how="left")

    vv = sel["view"].merge(conv, on=g)
    n_v = vv[(vv["timestamp"] >= vv["v_ts"]) & (vv["timestamp"] <= vv["c_ts"])].groupby(g).size()
    buy = f.loc[f["t_ts"].notna(), g + ["c_ts", "t_ts"]]
    cc = sel["addtocart"].merge(buy, on=g)
    n_c = cc[(cc["timestamp"] >= cc["c_ts"]) & (cc["timestamp"] <= cc["t_ts"])].groupby(g).size()
    f = f.merge(n_v.rename("n_view_before_cart").reset_index(), on=g, how="left")
    f = f.merge(n_c.rename("n_cart_before_tx").reset_index(), on=g, how="left")

    out = pd.DataFrame({k: f[k].astype("category") for k in keys})
    out["v_date"] = pd.to_datetime(f["v_ts"], unit="ms").dt.normalize()
    out["to_cart_min"] = ((f["c_ts"] - f["v_ts"]) / 60000).astype("float32")
    out["to_tx_min"] = ((f["t_ts"] - f["c_ts"]) / 60000).astype("float32")
    out["n_view_before_cart"] = f["n_view_before_cart"].astype("float32")
    out["n_cart_before_tx"] = f["n_cart_before_tx"].astype("float32")
    return out


def stage_counts(f):
    return [len(f), int(f["to_cart_min"].notna().sum()), int(f["to_tx_min"].notna().sum())]


def summary_table(f, key):
    """key별 퍼널 표 (View 방문자 수 내림차순)."""
    g = f.groupby(key, observed=True)
    t = pd.DataFrame({
        "View": g.size(),
        "AddToCart": g["to_cart_min"].count(),
        "Transaction": g["to_tx_min"].count(),
    })
    t["View→AddToCart(%)"] = t["AddToCart"] / t["View"] * 100
    t["AddToCart→Transaction(%)"] = (t["Transaction"] / t["AddToCart"] * 100).where(t["AddToCart"] > 0)
    t["View→Transaction(%)"] = t["Transaction"] / t["View"] * 100
    t["View→AddToCart 중앙값(분)"] = g["to_cart_min"].median()
    t["AddToCart→Transaction 중앙값(분)"] = g["to_tx_min"].median()
    t = t.reset_index().sort_values(["View", key], ascending=[False, True]).reset_index(drop=True)
    return t


def binned(series, bins, labels):
    s = series.dropna()
    counts = pd.cut(s, bins=bins, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({"구간": labels, "방문자 수": counts.values,
                         "비율(%)": (counts.values / max(len(s), 1) * 100)})


def time_dist(f, col):
    return binned(f[col], TIME_BINS, TIME_LABELS)


def count_dist(f, col):
    return binned(f[col], COUNT_BINS, COUNT_LABELS)
