"""events.csv 데이터 품질 점검 (원본은 읽기만 한다)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "events.csv"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# 1) 원형 확인을 위해 모든 열을 문자열로 읽는다
raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
section("기본 정보")
print("행, 열:", raw.shape)
print("열 이름:", list(raw.columns))

# 2) 결측: 빈 문자열/공백만 있는 값
section("열별 결측 (빈 값)")
blank = raw.apply(lambda s: s.str.strip().eq(""))
print(pd.DataFrame({"결측 건수": blank.sum(), "결측 비율(%)": (blank.mean() * 100).round(2)}))

# 3) 이벤트 종류 (공백/대소문자 변형 포함 원값 그대로)
section("이벤트 종류별 건수 (원값)")
ev = raw["event"].value_counts(dropna=False)
print(pd.DataFrame({"건수": ev, "비율(%)": (ev / len(raw) * 100).round(3)}))
print("공백/대소문자 정규화 후 고유값:", sorted(raw["event"].str.strip().str.lower().unique()))

# 4) 숫자 형식 위반
section("숫자 형식 점검")
num = {}
for col in ["timestamp", "visitorid", "itemid", "transactionid"]:
    s = raw[col].str.strip()
    nonblank = s.ne("")
    bad = nonblank & ~s.str.fullmatch(r"\d+")
    print(f"{col}: 비정상 형식 {bad.sum()}건 (비어있지 않은 값 {nonblank.sum()}건 중)")
    num[col] = pd.to_numeric(s.where(nonblank), errors="coerce").astype("Int64")

df = pd.DataFrame(num)
df["event"] = raw["event"].str.strip().str.lower()
df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

# 5) 이벤트 vs transactionid 일관성
section("event x transactionid 유무")
df["has_tid"] = df["transactionid"].notna()
print(pd.crosstab(df["event"], df["has_tid"].map({True: "tid 있음", False: "tid 없음"}), margins=True))
print("구매가 아닌데 tid 있음:", int((df["event"].ne("transaction") & df["has_tid"]).sum()))
print("구매인데 tid 없음:", int((df["event"].eq("transaction") & ~df["has_tid"]).sum()))

# 6) 시간 범위/정렬/일별 추이
section("timestamp")
print("최소:", df["ts"].min(), "/ 최대:", df["ts"].max())
print("파일이 시간순 정렬되어 있는가:", bool(df["timestamp"].is_monotonic_increasing))
daily = df.set_index("ts").resample("D")["event"].count()
print(f"일수: {len(daily)}, 이벤트 0인 날: {(daily == 0).sum()}")
print("일별 이벤트 수 요약:\n", daily.describe().round(0))
print("이벤트 적은 날 하위 5:\n", daily.nsmallest(5))
print("이벤트 많은 날 상위 5:\n", daily.nlargest(5))

# 7) 중복
section("중복")
key = ["timestamp", "visitorid", "event", "itemid"]
print("완전 중복 행:", int(raw.duplicated().sum()))
print("(timestamp, visitorid, event, itemid) 중복 행:", int(df.duplicated(key).sum()))
dup_ts = df.duplicated(["timestamp", "visitorid", "itemid"], keep=False)
same_time_diff_event = df[dup_ts].groupby(["timestamp", "visitorid", "itemid"])["event"].nunique()
print("같은 시각·방문자·상품인데 이벤트가 다른 그룹:", int((same_time_diff_event > 1).sum()))
# 짧은 간격 연속 중복 (같은 방문자·상품·이벤트, 1초 이내)
s = df.sort_values(key)
gap = s.groupby(["visitorid", "event", "itemid"])["timestamp"].diff()
print("같은 방문자·상품·이벤트가 1초 이내 반복된 행:", int((gap <= 1000).sum()))

# 8) transactionid 구조
section("transactionid 구조")
tx = df[df["event"].eq("transaction")]
print("구매 행:", len(tx), "/ 고유 transactionid:", tx["transactionid"].nunique())
g = tx.groupby("transactionid")
per = pd.DataFrame({
    "rows": g.size(),
    "items": g["itemid"].nunique(),
    "visitors": g["visitorid"].nunique(),
    "timestamps": g["timestamp"].nunique(),
})
print("transactionid당 상품 수 분포:\n", per["items"].describe().round(2))
print("방문자가 2명 이상인 transactionid:", int((per["visitors"] > 1).sum()))
print("시각이 2개 이상인 transactionid:", int((per["timestamps"] > 1).sum()))
print("같은 transactionid 안에 같은 상품 중복:", int((per["rows"] > per["items"]).sum()))

# 9) 퍼널 순서 (방문자-상품 단위)
section("퍼널 순서 점검 (방문자-상품 단위)")
first = df.groupby(["visitorid", "itemid", "event"])["timestamp"].min().unstack()
t = first[first["transaction"].notna()]
print("구매된 방문자-상품 쌍:", len(t))
print("  view 기록 없음:", int(t["view"].isna().sum()),
      "/ 첫 view가 첫 구매보다 늦음:", int((t["view"] > t["transaction"]).sum()))
print("  addtocart 기록 없음:", int(t["addtocart"].isna().sum()),
      "/ 첫 장바구니가 첫 구매보다 늦음:", int((t["addtocart"] > t["transaction"]).sum()))
a = first[first["addtocart"].notna()]
print("장바구니 방문자-상품 쌍:", len(a), "/ view 기록 없음:", int(a["view"].isna().sum()))

# 10) 방문자 활동 분포 (봇/이상 사용자)
section("방문자 활동 분포")
vc = df["visitorid"].value_counts()
print("고유 방문자:", len(vc), "/ 고유 상품:", df["itemid"].nunique())
print("방문자당 이벤트 수 분위:\n", vc.quantile([.5, .9, .99, .999, 1]).round(0))
print("이벤트 1건뿐인 방문자 비율(%):", round((vc == 1).mean() * 100, 2))
top = vc.head(10)
print("상위 10 방문자 이벤트 비중(%):", round(top.sum() / len(df) * 100, 2))
prof = df[df["visitorid"].isin(top.index)].groupby(["visitorid", "event"]).size().unstack(fill_value=0)
print(prof.loc[top.index])
txv = tx.groupby("visitorid")["transactionid"].nunique().sort_values(ascending=False)
print("구매 방문자:", len(txv), "/ 구매 건수 상위 5:\n", txv.head())
