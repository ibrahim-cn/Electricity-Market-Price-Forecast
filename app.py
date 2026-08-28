"""Kilitli gün-öncesi fiyat pipeline'ı için salt-okunur sunum panosu.

Eğitmez, tune etmez, model seçmez, forecast üretmez. Eksik dosyada uyarı gösterir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

LOCKED_MAE = 3.990091
LOCKED_RMSE = 5.878929
LOCKED_R2 = 0.668961
LOCKED_SMAPE = 7.314412
LOCKED_BIAS = 1.419419
NAIVE_MAE = 6.045924
NAIVE_RMSE = 9.030375
NAIVE_R2 = 0.218923
NAIVE_SMAPE = 11.675683
P75_BIAS = 0.840776
P90_BIAS = 0.920483
P95_BIAS = 0.556602

# Walk-forward (TRAIN+VAL). Test yalnızca kilitli finalde.
WF_BARS = pd.DataFrame(
    {
        "Model": ["Naive Lag-24", "Ridge", "LightGBM", "XGBoost", "Ridge+B+AR(1)"],
        "MAE": [7.351456, 5.796612, 5.916945, 5.945953, 4.529617],
    }
)

PRED_CANDIDATES = (
    ROOT / "data" / "processed" / "final_model" / "final_model_test_predictions.parquet",
    ROOT / "data" / "processed" / "predictions" / "final_test_predictions.parquet",
)
HOUR_CSV = ROOT / "reports" / "final" / "final_test_error_by_hour.csv"
MONTH_CSV = ROOT / "reports" / "final" / "final_test_error_by_month.csv"
QUANTILE_CSV = ROOT / "reports" / "final" / "final_test_error_by_price_quantile.csv"
SHAP_IMP = ROOT / "reports" / "explainability" / "shap_feature_importance.csv"
SHAP_GRP = ROOT / "reports" / "explainability" / "shap_feature_groups.csv"
SHAP_PNG = ROOT / "outputs" / "figures" / "shap_feature_importance.png"
FORECAST_AUDIT = ROOT / "reports" / "forecasting" / "forecasting_availability_audit.csv"
ENERGY_CSV = ROOT / "data" / "raw" / "energy_dataset.csv"

NAVY = "#1F4E9B"
NAVY_DARK = "#1F2A44"
MUTED = "#8AA0C2"
GRAY = "#8A93A3"

FRIENDLY = {
    "price_day_ahead_lag_24": "Fiyat gecikmesi 24s",
    "price_day_ahead_lag_48": "Fiyat gecikmesi 48s",
    "price_day_ahead_lag_168": "Fiyat gecikmesi 168s",
    "price_mean_lag24_lag48": "Fiyat ort. gecikme 24–48s",
    "total_load_forecast": "Toplam yük tahmini",
    "forecast_solar_day_ahead": "Güneş tahmini",
    "forecast_wind_onshore_day_ahead": "Rüzgâr tahmini",
    "renewable_forecast_total": "Yenilenebilir tahmin toplamı",
    "forecast_wind_share_of_load": "Rüzgârın yük payı",
    "forecast_solar_share_of_load": "Güneşin yük payı",
    "generation_wind_onshore_lag_24": "Rüzgâr üretimi gecikme 24s",
    "renewable_generation_lag_24": "Yenilenebilir üretim gecikme 24s",
    "generation_hydro_water_reservoir_lag_24": "Hidro rezervuar gecikme 24s",
    "generation_solar_lag_24": "Güneş üretimi gecikme 24s",
    "total_generation_lag_24": "Toplam üretim gecikme 24s",
    "clouds_all_national_mean_lag_24": "Ulusal bulutluluk gecikme 24s",
    "wind_speed_national_mean_lag_24": "Ulusal rüzgâr hızı gecikme 24s",
    "month": "Ay",
    "day_of_year": "Yılın günü",
    "day_of_month": "Ayın günü",
    "hour": "Saat",
    "calendar": "Takvim",
    "historical_target": "Geçmiş hedef fiyat",
    "historical_load": "Geçmiş yük",
    "historical_generation": "Geçmiş üretim",
    "historical_weather": "Geçmiş hava",
    "day_ahead_forecast": "Gün öncesi tahminler",
    "weather_aggregate": "Hava agregatları",
    "diğer": "Diğer (METHOD_B)",
}


def friendly(name: str) -> str:
    return FRIENDLY.get(name, str(name).replace("_", " "))


def missing(path: Path) -> None:
    rel = path.relative_to(ROOT) if path.is_absolute() else path
    st.warning(f"Gerekli çıktı bulunamadı: `{rel}`")


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame | None:
    for path in PRED_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        cols = {c.lower(): c for c in df.columns}
        need = {"timestamp_utc": None, "y_true": None, "y_pred": None}
        for key in list(need):
            if key in cols:
                need[key] = cols[key]
        if None in need.values():
            continue
        out = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(df[need["timestamp_utc"]], utc=True),
                "y_true": pd.to_numeric(df[need["y_true"]], errors="coerce"),
                "y_pred": pd.to_numeric(df[need["y_pred"]], errors="coerce"),
            }
        )
        if "residual" in cols:
            out["residual"] = pd.to_numeric(df[cols["residual"]], errors="coerce")
        else:
            out["residual"] = out["y_pred"] - out["y_true"]
        return out.sort_values("timestamp_utc").reset_index(drop=True)
    return None


@st.cache_data(show_spinner=False)
def load_energy_preview() -> pd.DataFrame | None:
    if not ENERGY_CSV.exists():
        return None
    keep = {
        "time",
        "total load forecast",
        "total load actual",
        "generation wind onshore",
        "generation solar",
        "forecast solar day ahead",
        "price day ahead",
    }
    df = pd.read_csv(ENERGY_CSV, usecols=lambda c: c in keep, nrows=6)
    return df.rename(
        columns={
            "time": "Zaman",
            "total load forecast": "Yük tahmini",
            "total load actual": "Yük (gerçek)",
            "generation wind onshore": "Rüzgâr üretimi",
            "generation solar": "Güneş üretimi",
            "forecast solar day ahead": "Güneş tahmini",
            "price day ahead": "Day-ahead fiyat",
        }
    )


def inject_css() -> None:
    st.markdown(
        """
<style>
footer {visibility: hidden;}
.block-container {padding-top: 1.35rem; padding-bottom: 2.2rem; max-width: 1180px;}
.kicker {font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-bottom: 0.4rem;}
.h1 {font-size: 2.4rem; font-weight: 750; color: #1F2A44; line-height: 1.15; margin: 0;}
.h2 {font-size: 1.7rem; font-weight: 720; color: #1F2A44; margin: 0 0 0.35rem 0;}
.sub {font-size: 1.15rem; color: #3d4a63; margin: 0.4rem 0 1rem 0;}
.lead {font-size: 1.05rem; color: #334155; line-height: 1.55; max-width: 46rem;}
.q {font-size: 1.45rem; font-weight: 700; color: #1F4E9B; line-height: 1.35; margin: 1.15rem 0 0.5rem 0;}
.card {border: 1px solid #d7deea; background: #fff; border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: 0.65rem;}
.card .l {font-size: 0.78rem; color: #64748b;}
.card .v {font-size: 1.7rem; font-weight: 750; color: #1F2A44; margin-top: 0.15rem;}
.card.hl {border-color: #1F4E9B; background: #f3f6fb;}
.card.hl .v {color: #1F4E9B;}
.mini {border: 1px solid #d7deea; border-radius: 12px; padding: 0.8rem 0.7rem; text-align: center; background: #fff;}
.mini .ico {font-size: 1.25rem;}
.mini .t {font-weight: 650; color: #1F2A44; margin-top: 0.2rem;}
.pass {background: #eaf6ee; border: 1px solid #b7ddc2; color: #166534; border-radius: 14px; padding: 0.95rem 1.1rem; font-size: 1.4rem; font-weight: 750; text-align: center;}
.stop {background: #fdecec; border: 1px solid #f0b4b4; color: #9b1c1c; border-radius: 14px; padding: 0.95rem 1.1rem; font-size: 1.35rem; font-weight: 750; text-align: center;}
.ok {margin: 0.28rem 0; font-size: 1.02rem; color: #1F2A44;}
.speak {font-size: 1.05rem; color: #334155; line-height: 1.65; margin: 0.65rem 0 1.1rem 0; max-width: 50rem;}
.tl {display: grid; grid-template-columns: 7fr 1.5fr 1.5fr; gap: 8px; margin: 0.9rem 0 1.1rem 0;}
.tl div {border-radius: 10px; padding: 0.85rem 0.9rem; color: #fff; font-weight: 650;}
.tl .a {background: #1F4E9B;}
.tl .b {background: #4a6aa8;}
.tl .c {background: #1F2A44;}
.tl span {display:block; font-size:0.78rem; font-weight:500; opacity:0.9; margin-top:0.15rem;}
</style>
""",
        unsafe_allow_html=True,
    )


def speak(text: str) -> None:
    st.markdown(f'<p class="speak">{text}</p>', unsafe_allow_html=True)


def card(label: str, value: str, *, highlight: bool = False) -> None:
    cls = "card hl" if highlight else "card"
    st.markdown(
        f'<div class="{cls}"><div class="l">{label}</div><div class="v">{value}</div></div>',
        unsafe_allow_html=True,
    )


def bar_chart(df: pd.DataFrame, x: str, y: str, *, highlight: str | None = None, height: int = 340) -> None:
    try:
        import altair as alt

        color = (
            alt.condition(alt.datum[x] == highlight, alt.value(NAVY), alt.value(MUTED))
            if highlight
            else alt.value(NAVY)
        )
        chart = (
            alt.Chart(df)
            .mark_bar(size=38)
            .encode(
                x=alt.X(f"{x}:N", sort=list(df[x]), title=""),
                y=alt.Y(f"{y}:Q", title="MAE (€/MWh)"),
                color=color,
            )
            .properties(height=height)
        )
        labels = (
            alt.Chart(df)
            .mark_text(dy=-8, fontSize=12, fontWeight=600)
            .encode(
                x=alt.X(f"{x}:N", sort=list(df[x])),
                y=y,
                text=alt.Text(f"{y}:Q", format=".2f"),
            )
        )
        st.altair_chart(chart + labels, use_container_width=True)
    except Exception:
        st.bar_chart(df.set_index(x)[[y]])


def page_problem() -> None:
    st.markdown('<div class="kicker">01  ·  Problem ve amaç</div>', unsafe_allow_html=True)
    st.markdown('<p class="h1">Elektrik Piyasası Fiyat Tahmini</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub">İspanya’nın saatlik gün öncesi elektrik fiyatını tahmin etmek</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="q">Model, sadece dünün aynı saatindeki fiyatı kullanmaktan daha iyi tahmin yapabilir mi?</p>',
        unsafe_allow_html=True,
    )
    card("Hedef", "Gün öncesi piyasa fiyatı (€/MWh)", highlight=True)
    a, b, c = st.columns(3)
    with a:
        card("Gözlem", "35.064 saat")
    with b:
        card("Dönem", "2014–2018")
    with c:
        card("Piyasa", "İspanya elektrik piyasası")
    speak(
        "Bu çalışmada İspanya elektrik piyasasının saatlik gün öncesi fiyatını tahmin ettim. "
        "Karşılaştırma olarak dünün aynı saatindeki fiyatı kullandım. "
        "Geçmiş fiyat, tüketim, üretim, hava ve gün öncesi tahminlerle sızıntısız bir zaman serisi modeli kurdum "
        "ve bu basit kopyalamadan daha iyi sonuç alıp almadığıma baktım."
    )


def page_data() -> None:
    st.markdown('<div class="kicker">02  ·  Veri</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Model hangi verilerle çalışıyor?</p>', unsafe_allow_html=True)
    speak(
        "Enerji piyasası ve hava verisini birleştirdim. "
        "Sadece gün öncesi ihalenin yapıldığı anda elde olan bilgileri kullandım. "
        "Aynı saatin gerçekleşen tüketimini ve gerçekleşen fiyatını özellik olarak almadım."
    )
    a, b, c = st.columns(3)
    with a:
        card("Enerji gözlemi", "35.064")
    with b:
        card("Hava kaydı", "178.396")
    with c:
        card("Sızıntısız özellik", "184+")

    cols = st.columns(5)
    for col, (ico, title) in zip(
        cols,
        (
            ("⚡", "Üretim"),
            ("🔌", "Yük / Tüketim"),
            ("🌤️", "Hava"),
            ("📈", "Gün öncesi tahmin"),
            ("💰", "Piyasa fiyatı"),
        ),
    ):
        with col:
            st.markdown(
                f'<div class="mini"><div class="ico">{ico}</div><div class="t">{title}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("**Özellik grupları:** Takvim · Geçmiş fiyat · Yük · Üretim · Hava · Gün öncesi tahmin · METHOD_B")
    st.markdown("**Hedef gecikmeleri:** t-24 · t-48 · t-168")
    speak(
        "Üretim ve yükü arz ile talebi taşıdığı için, havayı rüzgâr ve güneşi etkilediği için, "
        "resmi gün öncesi tahminleri de operatörün elindeki bilgi olduğu için kullandım. "
        "Fiyat gecikmesi olarak dünü, iki gün önceyi ve geçen haftanın aynı saatini aldım. "
        "Bir saat önceki fiyatı kullanmadım, çünkü ihale anında o fiyat henüz yoktu."
    )

    preview = load_energy_preview()
    if preview is None:
        missing(ENERGY_CSV)
    else:
        st.dataframe(preview, hide_index=True, use_container_width=True)

    with st.expander("Teknik ayrıntı"):
        st.write("Fit anında METHOD_B ile 3 kolon eklenir (187 model kolonu). `price actual` özellik değildir.")


def page_models() -> None:
    st.markdown('<div class="kicker">03  ·  Model seçimi</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Hangi modeli seçtim ve neden?</p>', unsafe_allow_html=True)
    speak(
        "Modelleri test setini açmadan walk-forward ortalama mutlak hata ile karşılaştırdım. "
        "Düşük çubuk daha iyidir. Naive tabanın yanında Ridge, LightGBM ve XGBoost’u denedim. "
        "En düşük hatayı Ridge tabanlı birleşik model verdiği için onu seçtim."
    )
    bar_chart(WF_BARS, "Model", "MAE", highlight="Ridge+B+AR(1)", height=380)
    st.success("Seçilen model: **Ridge(α = 0.001) + METHOD_B + AR(1)**")
    st.markdown(
        """
**Ridge (α = 0.001):** Ana regresyon modeli olarak kullandım. Çoklu doğrusal bağlantının etkisini azaltarak üretim, tüketim ve hava durumu gibi değişkenlerden fiyat tahmini yaptım.

**METHOD_B:** Ridge’in yakalayamadığı ek yapıyı modellemek için kullandım; tahmini bu bileşenle iyileştirdim.

**AR(1):** Ridge artığındaki bir önceki zaman adımını kullanarak zamansal bağımlılığı yakaladım.
"""
    )


def page_result() -> None:
    st.markdown('<div class="kicker">04  ·  Kilitli test</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Kilitli sonuç</p>', unsafe_allow_html=True)
    speak(
        "Test setini yalnızca bir kez açtım. Seçtiğim modelin ortalama mutlak hatası 3,99 euro, "
        "Naive tabanınki 6,05 euro çıktı; yani hata yaklaşık yüzde 34 daha düşük. "
        "Bu grafikte gerçek fiyatlarla tahminimi karşılaştırdım. "
        "Henüz oluşmamış gelecek saatler için tahmin üretmedim."
    )
    a, b, c = st.columns([1.25, 1, 1])
    with a:
        card("MAE", f"{LOCKED_MAE:.2f} €/MWh", highlight=True)
    with b:
        card("Naive Lag-24", f"{NAIVE_MAE:.2f} €/MWh")
    with c:
        card("MAE iyileşmesi", "≈ %34")

    st.markdown(
        f"""
| Metrik | Model | Naive |
|---|---:|---:|
| MAE | {LOCKED_MAE:.2f} | {NAIVE_MAE:.2f} |
| RMSE | {LOCKED_RMSE:.2f} | {NAIVE_RMSE:.2f} |
| R² | {LOCKED_R2:.3f} | {NAIVE_R2:.3f} |
| sMAPE | {LOCKED_SMAPE:.2f}% | {NAIVE_SMAPE:.2f}% |
| Sapma | +{LOCKED_BIAS:.2f} | ≈ 0 |
"""
    )

    pred = load_predictions()
    if pred is None:
        missing(PRED_CANDIDATES[0])
        return
    daily = (
        pred.set_index("timestamp_utc")[["y_true", "y_pred"]]
        .resample("D")
        .mean()
        .rename(columns={"y_true": "Gerçek", "y_pred": "Tahmin"})
        .reset_index()
    )
    st.markdown("**Gerçek ve tahmin** · kilitli test, günlük ortalama")
    try:
        import altair as alt

        long = daily.melt("timestamp_utc", var_name="Seri", value_name="€/MWh")
        chart = (
            alt.Chart(long)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("timestamp_utc:T", title=""),
                y=alt.Y("€/MWh:Q"),
                color=alt.Color(
                    "Seri:N",
                    scale=alt.Scale(domain=["Gerçek", "Tahmin"], range=[NAVY_DARK, NAVY]),
                    legend=alt.Legend(orient="top"),
                ),
            )
            .properties(height=430)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.line_chart(daily.set_index("timestamp_utc"))


def page_errors() -> None:
    st.markdown('<div class="kicker">05  ·  Hata</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Model nerede hata yapıyor?</p>', unsafe_allow_html=True)
    speak(
        "Artığı tahminden gerçeği çıkararak hesapladım. Pozitif değer aşırı tahmini, negatif değer eksik tahmini gösteriyor. "
        "Hatanın rastgele dağılmadığını gördüm. Test döneminde model ortalama olarak biraz yüksek tahmin ediyor "
        "ve yüksek fiyatlı saatlerde sapma devam ediyor. Bu yüzden modeli üretime hazır saymadım."
    )

    pred = load_predictions()
    if pred is None:
        missing(PRED_CANDIDATES[0])
        return
    st.markdown("**Zaman içinde artık** · kilitli test")
    st.line_chart(pred.set_index("timestamp_utc")[["residual"]].rename(columns={"residual": "Artık"}))

    left, right = st.columns(2)
    with left:
        st.markdown("**Artık dağılımı**")
        hist, edges = np.histogram(pred["residual"].dropna(), bins=40)
        hist_df = pd.DataFrame({"bin": 0.5 * (edges[:-1] + edges[1:]), "Sayı": hist})
        st.bar_chart(hist_df.set_index("bin"))
    with right:
        q = load_csv(str(QUANTILE_CSV))
        if q is not None and "quantile" in q.columns:
            st.markdown("**Fiyat dilimine göre sapma**")
            q_plot = q.rename(columns={"quantile": "Dilim", "bias": "Sapma"}).set_index("Dilim")
            if "Sapma" in q_plot.columns:
                st.bar_chart(q_plot[["Sapma"]])
        else:
            missing(QUANTILE_CSV)

    hour_csv = load_csv(str(HOUR_CSV))
    if hour_csv is not None and "hour" in hour_csv.columns and "MAE" in hour_csv.columns:
        st.markdown("**Saate göre MAE**")
        st.bar_chart(hour_csv.set_index("hour")[["MAE"]])

    h1, h2, h3 = st.columns(3)
    with h1:
        card("P75+ sapma", f"+{P75_BIAS:.2f}")
    with h2:
        card("P90+ sapma", f"+{P90_BIAS:.2f}")
    with h3:
        card("P95+ sapma", f"+{P95_BIAS:.2f}")

    speak(
        "Geliştirme döneminde yüksek fiyatlarda eksik tahmin gördüm. "
        "Kilitli testte sapma tersine döndü ve model biraz yüksek tahmin etmeye başladı. "
        "Yüksek fiyatlı saatlerdeki hatayı iki dönemde de çözemedim."
    )


def page_explain() -> None:
    st.markdown('<div class="kicker">06  ·  Açıklanabilirlik</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Model hangi değişkenlere bakıyor?</p>', unsafe_allow_html=True)
    speak(
        "SHAP ile modelin hangi değişkenlere daha çok yaslandığına baktım; bu neden-sonuç kanıtlamaz. "
        "En güçlü sinyaller olarak geçmiş fiyatları ve gün öncesi resmi tahminleri gördüm. "
        "Ay ve yılın günü gibi takvim değişkenlerini tek başına fiyat sürücüsü gibi okumadım, çünkü birbirine bağlılar."
    )
    imp = load_csv(str(SHAP_IMP))
    if imp is None:
        missing(SHAP_IMP)
        return
    wanted = [
        "price_day_ahead_lag_24",
        "price_day_ahead_lag_48",
        "total_load_forecast",
        "forecast_wind_share_of_load",
        "forecast_solar_day_ahead",
        "renewable_forecast_total",
        "generation_wind_onshore_lag_24",
        "price_mean_lag24_lag48",
        "price_day_ahead_lag_168",
        "forecast_wind_onshore_day_ahead",
    ]
    show = imp[imp["feature"].isin(wanted)].copy()
    show["Özellik"] = show["feature"].map(friendly)
    show = show.sort_values("mean_abs_shap", ascending=True)
    try:
        import altair as alt

        chart = (
            alt.Chart(show)
            .mark_bar()
            .encode(
                y=alt.Y("Özellik:N", sort="x", title=""),
                x=alt.X("mean_abs_shap:Q", title="Ort. |SHAP|"),
                color=alt.value(NAVY),
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(show.set_index("Özellik")[["mean_abs_shap"]])

    with st.expander("Teknik ayrıntı"):
        st.write(
            "`month` ve `day_of_year` neredeyse eşdoğrusaldır. Ham |SHAP| sıralamasında üstte görünebilir; "
            "tek tek fiyat sürücüsü gibi okunmamalıdır."
        )
        top = imp.head(10).copy()
        top["Özellik"] = top["feature"].map(friendly)
        st.dataframe(
            top[["Özellik", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "Ort. |SHAP|"}),
            hide_index=True,
            use_container_width=True,
        )
        grp = load_csv(str(SHAP_GRP))
        if grp is not None:
            g = grp.copy()
            g["Grup"] = g["feature_group"].map(friendly)
            st.bar_chart(g.set_index("Grup")[["mean_abs_shap_sum"]])
        if SHAP_PNG.exists():
            st.image(str(SHAP_PNG))


def page_leakage() -> None:
    st.markdown('<div class="kicker">07  ·  Güven</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Bu sonuca güvenebilir miyiz?</p>', unsafe_allow_html=True)
    speak(
        "Bu sonuca güvenmek için bilgi sızıntısını kapattım. Veriyi zamana göre böldüm ve satırları karıştırmadım. "
        "Aynı saatin gerçekleşen değerlerini ve gerçekleşen piyasa fiyatını özellik olarak kullanmadım. "
        "Modeli walk-forward ile seçtim; test setini ayar için kullanmadım ve yalnızca bir kez ölçtüm."
    )
    st.markdown(
        """
<div class="tl">
  <div class="a">EĞİTİM<span>%70 · 2014 → 2017</span></div>
  <div class="b">DOĞRULAMA<span>%15</span></div>
  <div class="c">TEST<span>%15 · kilitli</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pass">✓ Sızıntı denetimi: GEÇTİ</div>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Kronolojik bölme</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Rastgele bölme / karıştırma yok</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Gerçekleşen piyasa fiyatı kullanılmadı</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Aynı saatteki gerçekleşmeler kullanılmadı</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Gelecek bilgi kullanılmadı</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Test seti ayar için kullanılmadı</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Walk-forward doğrulama</p>', unsafe_allow_html=True)
    st.caption("Hedef gecikmeleri: t-24 / t-48 / t-168")

    with st.expander("Teknik ayrıntı"):
        audit = load_csv(str(FORECAST_AUDIT))
        if audit is None:
            missing(FORECAST_AUDIT)
        else:
            cols = [c for c in ("feature", "feature_group", "notes") if c in audit.columns]
            st.dataframe(audit[cols] if cols else audit, hide_index=True, use_container_width=True)


def page_close() -> None:
    st.markdown('<div class="kicker">08  ·  Kapanış</div>', unsafe_allow_html=True)
    st.markdown('<p class="h2">Sonuç</p>', unsafe_allow_html=True)
    speak(
        "Sızıntısız bir tahmin hattı kurdum ve dünün aynı saatini kopyalayan Naive tabanı geçtim. "
        "Model olarak Ridge, METHOD_B ve AR(1) birleşimini seçtim. "
        "Kilitli testte ortalama mutlak hatayı yaklaşık yüzde 34 düşürdüm. "
        "Yüksek fiyatlı saatlerdeki hatayı çözemedim; veri 2014–2018 aralığında ve testi tek bir kilitli dönemde ölçtüm."
    )
    st.markdown('<p class="ok">✓ Sızıntısız zaman serisi hattı</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Kronolojik bölme · walk-forward doğrulama</p>', unsafe_allow_html=True)
    st.markdown('<p class="ok">✓ Ridge + METHOD_B + AR(1)</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="ok">✓ MAE = {LOCKED_MAE:.2f} €/MWh  ·  Naive’e göre ≈ %34</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="ok">✓ Doğrusal SHAP açıklanabilirliği</p>', unsafe_allow_html=True)
    st.markdown("### Sınırlılıklar")
    st.warning("Yüksek fiyatlı saatlerde hata devam ediyor.")
    st.warning("Veri 2014–2018 aralığındadır ve test tek bir kilitli dönemdir.")
    st.caption("Yakıt fiyatları, arızalar ve sınır ötesi kısıtlar veri setinde yoktur.")
    st.markdown("### Özet")
    st.success(
        "Sızıntısız zaman serisi modeli kurdum ve dünün aynı saatini kopyalayan tabandan "
        "belirgin şekilde daha iyi gün öncesi elektrik fiyat tahmini aldım."
    )


PAGES = (
    ("🎯 Problem ve amaç", page_problem),
    ("📊 Veri ve özellikler", page_data),
    ("🤖 Model karşılaştırması", page_models),
    ("🏆 Kilitli sonuç", page_result),
    ("📉 Hata analizi", page_errors),
    ("🔍 Açıklanabilirlik", page_explain),
    ("🛡️ Sızıntı ve doğrulama", page_leakage),
    ("✅ Sonuç ve sınırlılıklar", page_close),
)


def main() -> None:
    st.set_page_config(page_title="Elektrik Piyasası Fiyat Tahmini", layout="wide")
    inject_css()
    labels = [p[0] for p in PAGES]
    choice = st.sidebar.radio("Sunum", labels, index=0)
    dict(PAGES)[choice]()


if __name__ == "__main__":
    main()
