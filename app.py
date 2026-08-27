"""Kilitli gün-öncesi fiyat pipeline'ı için salt-okunur Streamlit panosu.

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
P75_BIAS = 0.840776
P90_BIAS = 0.920483
P95_BIAS = 0.556602

# Sunum sırası (walk-forward, 4 kat, TRAIN+VAL). Test yalnızca kilitli finalde.
PRESENT_MODELS = (
    "Ridge",
    "LightGBM",
    "XGBoost",
    "ARIMA",
    "Ridge+B+AR(1)",
)
WF_COMPARE = pd.DataFrame(
    {
        "Sıra": [1, 2, 3, 4, 5],
        "Model": list(PRESENT_MODELS),
        "MAE": [5.796612, 5.916945, 5.945953, 4.529617, 4.529617],
        "RMSE": [7.414208, 7.852708, 7.905745, 6.070565, 6.070565],
        "R²": [0.622150, 0.606148, 0.601387, 0.747805, 0.747805],
        "sMAPE": [12.837611, 13.003515, 13.056313, 10.146126, 10.146126],
        "Sapma": [-1.896630, -3.153088, -3.069063, -0.712679, -0.712679],
        "Test MAE": [None, None, None, None, LOCKED_MAE],
        "Not": [
            "α=0.001, 184 özellik",
            "184 özellik, ayarsız 200 ağaç",
            "184 özellik, ayarsız 200 ağaç",
            "Ridge+METHOD_B artığına ARIMA(1,0,0), 24s blok",
            "Kilitli model (METHOD_B + AR(1))",
        ],
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
FORECAST_STRICT = ROOT / "reports" / "forecasting" / "forecasting_predictions.csv"
FORECAST_ASSUMED = ROOT / "reports" / "forecasting" / "forecasting_predictions_assumed.csv"
FORECAST_AUDIT = ROOT / "reports" / "forecasting" / "forecasting_availability_audit.csv"
FORECAST_FIG = ROOT / "outputs" / "figures" / "forecast_24h.png"

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
    "y_true": "Gerçek",
    "y_pred": "Tahmin",
    "residual": "Artık",
}


def friendly(name: str) -> str:
    if name in FRIENDLY:
        return FRIENDLY[name]
    return str(name).replace("_", " ")


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


def kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
<div style="border:1px solid rgba(125,125,125,0.25);border-radius:10px;padding:0.85rem 1rem;margin-bottom:0.6rem;">
<div style="font-size:0.78rem;opacity:0.7;letter-spacing:0.02em;">{label}</div>
<div style="font-size:1.45rem;font-weight:650;margin-top:0.15rem;">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def presentation_mae_chart() -> None:
    plot = WF_COMPARE[["Sıra", "Model", "MAE"]].copy()
    plot["etiket"] = plot["Sıra"].astype(str) + ". " + plot["Model"]
    try:
        import altair as alt

        chart = (
            alt.Chart(plot)
            .mark_bar()
            .encode(
                x=alt.X("etiket:N", sort=list(plot["etiket"]), title=""),
                y=alt.Y("MAE:Q", title="Walk-forward MAE (€/MWh)"),
                color=alt.condition(
                    alt.datum.etiket == "5. Ridge+B+AR(1)",
                    alt.value("#1f4e9b"),
                    alt.value("#8aa0c2"),
                ),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(plot.set_index("etiket")[["MAE"]])


def page_compare() -> None:
    st.subheader("Model karşılaştırması")
    st.write(
        "Sunum sırası sabittir: **Ridge → LightGBM → XGBoost → ARIMA → Ridge+B+AR(1)**. "
        "Sayılar 4 katlı walk-forward MAE’dir (TRAIN+VALIDATION). Test yalnızca kilitli finale bakıldı."
    )
    st.caption(
        "ARIMA, ham fiyata tek değişkenli ARIMA değildir. Ridge+METHOD_B artıklarına "
        "ARIMA(1,0,0) yani AR(1) oturtulmuştur; 24 saatlik gün öncesi blok protokolü."
    )

    presentation_mae_chart()

    show = WF_COMPARE.copy()
    show["MAE"] = show["MAE"].map(lambda x: f"{x:.2f}")
    show["RMSE"] = show["RMSE"].map(lambda x: f"{x:.2f}")
    show["R²"] = show["R²"].map(lambda x: f"{x:.3f}")
    show["sMAPE"] = show["sMAPE"].map(lambda x: f"{x:.1f}")
    show["Sapma"] = show["Sapma"].map(lambda x: f"{x:.2f}")
    show["Test MAE"] = show["Test MAE"].apply(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.success("Kilitli teslimat: **5. Ridge+B+AR(1)** · test MAE = 3,99 · naive = 6,05")


def page_overview() -> None:
    st.subheader("Bu proje ne yapıyor?")
    st.write(
        "Bu pano, İspanya saatlik **gün öncesi elektrik fiyatını** (`price day ahead`, €/MWh) "
        "sızıntısız ve kronolojik bir makine öğrenmesi hattıyla tahmin eden kilitli modeli gösterir. "
        "Bu sayfada model yeniden eğitilmez veya ayarlanmaz."
    )
    st.info(
        "Model, sızıntısız kronolojik doğrulama ile seçildi. "
        "Kilitli test seti model seçimi veya hiperparametre ayarı için kullanılmadı."
    )
    st.markdown("**MODEL DURUMU = KİLİTLİ**")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Model", "Ridge+B+AR(1)")
    with c2:
        kpi("Artık düzeltmesi", "METHOD_B + AR(1)")
    with c3:
        kpi("Test MAE", f"{LOCKED_MAE:.2f}")
    with c4:
        kpi("Test RMSE", f"{LOCKED_RMSE:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi("Test R²", f"{LOCKED_R2:.3f}")
    with c6:
        kpi("sMAPE", f"{LOCKED_SMAPE:.2f}%")
    with c7:
        kpi("Naive MAE", f"{NAIVE_MAE:.2f}")
    with c8:
        kpi("Naive'i geçti", "EVET")

    st.caption(
        "Kilitli test: 5.260 saat. Naive Lag-24'e göre MAE iyileşmesi yaklaşık %34. "
        "Düşük MAE, modelin yansız olduğu anlamına gelmez (test sapması = +1,42)."
    )
    st.warning(
        "24 saatlik operasyonel tahmin **üretime hazır değil**. "
        "Ayrıntı için «24 saatlik tahmin» sayfasına bakın."
    )


def page_performance() -> None:
    st.subheader("Kilitli test performansı")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        kpi("MAE", f"{LOCKED_MAE:.2f}")
    with m2:
        kpi("RMSE", f"{LOCKED_RMSE:.2f}")
    with m3:
        kpi("R²", f"{LOCKED_R2:.3f}")
    with m4:
        kpi("sMAPE", f"{LOCKED_SMAPE:.2f}%")
    with m5:
        kpi("Sapma", f"+{LOCKED_BIAS:.2f}")

    st.markdown("**Sunum sırası (walk-forward MAE)**")
    presentation_mae_chart()
    st.caption("Resmi kilitli skor yalnızca Ridge+B+AR(1) içindir. Naive Lag-24 test MAE = 6,05.")

    pred = load_predictions()
    if pred is None:
        missing(PRED_CANDIDATES[0])
        return

    show = pred.rename(columns={"y_true": "Gerçek", "y_pred": "Tahmin"}).set_index("timestamp_utc")
    st.markdown("**Gerçek vs tahmin**")
    st.line_chart(show[["Gerçek", "Tahmin"]])

    st.markdown("**Zaman içinde artık** (tahmin − gerçek)")
    st.line_chart(pred.set_index("timestamp_utc")[["residual"]].rename(columns={"residual": "Artık"}))

    left, right = st.columns(2)
    with left:
        st.markdown("**Artık dağılımı**")
        hist, edges = np.histogram(pred["residual"].dropna(), bins=40)
        hist_df = pd.DataFrame({"bin": 0.5 * (edges[:-1] + edges[1:]), "Sayı": hist})
        st.bar_chart(hist_df.set_index("bin"))
    with right:
        st.markdown("**Gerçek vs tahmin saçılımı**")
        scatter = pred[["y_true", "y_pred"]].rename(columns={"y_true": "Gerçek", "y_pred": "Tahmin"})
        st.scatter_chart(scatter, x="Gerçek", y="Tahmin")
    st.caption("Pozitif artık = aşırı tahmin. Kilitli test sapması +1,42.")


def page_errors() -> None:
    st.subheader("Hata nerede artıyor?")
    st.info(
        "Geliştirme walk-forward değerlendirmesinde pahalı saatlerde sistematik "
        "eksik tahmin vardı; kilitli testte sapma pozitife döndü. "
        "Bu işaret değişimi test görüldükten sonra düzeltilmedi, yalnızca raporlandı."
    )

    h1, h2, h3 = st.columns(3)
    with h1:
        kpi("P75+ sapma", f"+{P75_BIAS:.2f}")
    with h2:
        kpi("P90+ sapma", f"+{P90_BIAS:.2f}")
    with h3:
        kpi("P95+ sapma", f"+{P95_BIAS:.2f}")

    q = load_csv(str(QUANTILE_CSV))
    if q is not None:
        st.markdown("**Fiyat dilimine göre hata (kilitli test)**")
        q_plot = q.copy()
        if "quantile" in q_plot.columns:
            q_plot = q_plot.set_index("quantile")
        q_plot = q_plot.rename(columns={"MAE": "MAE", "bias": "Sapma"})
        cols = [c for c in ("MAE", "Sapma") if c in q_plot.columns]
        if cols:
            st.bar_chart(q_plot[cols])

    hp = pd.DataFrame(
        {
            "Rejim": ["P75+", "P90+", "P95+"],
            "Kilitli test sapması": [P75_BIAS, P90_BIAS, P95_BIAS],
        }
    )
    st.markdown("**Yüksek fiyat sapması (kilitli test, yalnızca referans)**")
    st.bar_chart(hp.set_index("Rejim"))

    pred = load_predictions()
    hour_csv = load_csv(str(HOUR_CSV))
    month_csv = load_csv(str(MONTH_CSV))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Saate göre MAE**")
        if hour_csv is not None and "hour" in hour_csv.columns:
            st.bar_chart(hour_csv.set_index("hour")[["MAE"]] if "MAE" in hour_csv.columns else hour_csv.set_index("hour"))
        else:
            missing(HOUR_CSV)
    with c2:
        st.markdown("**Aya göre MAE**")
        if month_csv is not None and "month" in month_csv.columns:
            st.bar_chart(month_csv.set_index("month")[["MAE"]] if "MAE" in month_csv.columns else month_csv.set_index("month"))
        else:
            missing(MONTH_CSV)

    if pred is not None:
        tmp = pred.copy()
        tmp["hour"] = tmp["timestamp_utc"].dt.hour
        tmp["month"] = tmp["timestamp_utc"].dt.month
        bias_h = tmp.groupby("hour", sort=True)["residual"].mean().rename("Sapma")
        bias_m = tmp.groupby("month", sort=True)["residual"].mean().rename("Sapma")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Saate göre sapma**")
            st.bar_chart(bias_h)
        with d2:
            st.markdown("**Aya göre sapma**")
            st.bar_chart(bias_m)
    st.caption("Bu grafikler tanı amaçlıdır. Kilitli model bunlara göre kalibre edilmedi.")


def page_explain() -> None:
    st.subheader("Kilitli model nelere bakıyor?")
    st.info("Ridge doğrusal bir modeldir. TreeSHAP yerine tam doğrusal SHAP katkıları kullanıldı.")
    st.warning(
        "Ay (`month`) ve yılın günü (`day_of_year`) birbirine çok bağlı takvim kodlarıdır. "
        "Büyük ve zıt katsayıları, fiyatın nedeni gibi tek tek yorumlanmamalıdır."
    )
    st.caption(
        "Hava özellikleri birçok kolona dağılır. Tek tek şehir gecikmeleri genellikle "
        "geçmiş fiyat ve yük/tahmin sinyallerinden daha zayıftır. "
        "Sayılar öngörü ilişkisidir; nedensellik kanıtı değildir."
    )

    imp = load_csv(str(SHAP_IMP))
    grp = load_csv(str(SHAP_GRP))
    if imp is None:
        missing(SHAP_IMP)
    else:
        top = imp.head(20).copy()
        top["label"] = top["feature"].map(friendly)
        st.markdown("**En önemli 20 özellik (ortalama |doğrusal SHAP|)**")
        st.bar_chart(top.set_index("label")[["mean_abs_shap"]].rename(columns={"mean_abs_shap": "Ortalama |SHAP|"}))
        highlight = [
            "price_day_ahead_lag_24",
            "price_day_ahead_lag_48",
            "total_load_forecast",
            "forecast_wind_share_of_load",
            "forecast_solar_day_ahead",
            "renewable_forecast_total",
        ]
        show = imp[imp["feature"].isin(highlight)].copy()
        if not show.empty:
            show["Özellik"] = show["feature"].map(friendly)
            st.markdown("**Seçilmiş piyasa / tahmin özellikleri**")
            st.dataframe(
                show[["Özellik", "mean_abs_shap", "mean_std_coef", "direction"]].rename(
                    columns={
                        "mean_abs_shap": "Ort. |SHAP|",
                        "mean_std_coef": "Ort. standart katsayı",
                        "direction": "Yön",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    if grp is None:
        missing(SHAP_GRP)
    else:
        g = grp.copy()
        g["group"] = g["feature_group"].map(friendly)
        st.markdown("**Özellik grubu katkısı (ortalama |SHAP| toplamı)**")
        st.bar_chart(g.set_index("group")[["mean_abs_shap_sum"]].rename(columns={"mean_abs_shap_sum": "Grup |SHAP|"}))
        st.caption("Takvim grubunun büyük toplamı, ay / yılın günü çiftinden gelir.")

    if SHAP_PNG.exists():
        st.image(str(SHAP_PNG), caption="Kayıtlı açıklanabilirlik figürü (walk-forward geliştirme seti).")


def _forecast_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        out["Zaman (UTC)"] = out["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M")
    if "y_pred" in out.columns:
        out["Tahmin"] = out["y_pred"].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}")
    keep = [c for c in ("Zaman (UTC)", "forecast_horizon", "Tahmin", "forecast_origin") if c in out.columns]
    return out[keep].rename(columns={"forecast_horizon": "Ufuk", "forecast_origin": "Başlangıç"})


def page_forecast() -> None:
    st.subheader("24 saatlik tahmin durumu")
    st.error("ÜRETİME HAZIR DEĞİL")
    st.markdown("**Mevcut veri setiyle 24 saatlik tahmin üretime hazır değildir.**")
    st.markdown(
        """
1. Gün öncesi tahmin yayın saatleri dosyada bağımsız doğrulanmamış.  
2. Bazı gerçekleşmiş gecikme özellikleri tahmin anında henüz yok.  
3. Bu yüzden katı (STRICT) mod sızıntılı tahmin üretmez.
"""
    )
    st.caption(
        "24 saatlik tahmin, ancak gerekli her özellik tahmin başlangıç anında biliniyorsa üretime hazır olur."
    )

    strict = load_csv(str(FORECAST_STRICT))
    st.markdown("**Katı (STRICT) üretim yolu**")
    if strict is None:
        missing(FORECAST_STRICT)
    else:
        st.dataframe(_forecast_table(strict), hide_index=True, use_container_width=True)
        st.caption("Boş tahminler kasıtlıdır. Eksik değerler 0 yazılmaz ve tahmin gibi çizilmez.")

    assumed = load_csv(str(FORECAST_ASSUMED))
    st.markdown("### Varsayımsal senaryo — ÜRETİME HAZIR DEĞİL")
    st.warning("Bu tahminler bağımsız doğrulanmamış varsayımlara dayanır.")
    if assumed is None:
        missing(FORECAST_ASSUMED)
        return
    st.dataframe(_forecast_table(assumed), hide_index=True, use_container_width=True)
    plot = assumed.copy()
    plot["timestamp_utc"] = pd.to_datetime(plot.get("timestamp_utc"), utc=True, errors="coerce")
    plot = plot.dropna(subset=["y_pred", "timestamp_utc"])
    if not plot.empty:
        st.line_chart(
            plot.set_index("timestamp_utc")[["y_pred"]].rename(
                columns={"y_pred": "Varsayımsal tahmin (üretim değil)"}
            )
        )
    if FORECAST_FIG.exists():
        st.image(str(FORECAST_FIG), caption="Varsayımsal senaryo figürü — üretim tahmini değildir.")


def page_audit() -> None:
    st.subheader("Veri bölmesi ve sızıntı kontrolleri")
    a, b, c = st.columns(3)
    with a:
        kpi("Veri satırı", "35.064")
    with b:
        kpi("Güvenli özellik", "184")
    with c:
        kpi("Bölme", "70 / 15 / 15")

    st.markdown(
        """
| Bölme | Pay | Dönem (UTC) |
|---|---|---|
| Eğitim | %70 | 2014-12-31 → 2017-10-19 |
| Doğrulama | %15 | 2017-10-19 → 2018-05-26 |
| Test | %15 | 2018-05-26 → 2018-12-31 |
"""
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        kpi("Sızıntı", "GEÇTİ")
    with b2:
        kpi("Test ile ayar", "HAYIR")
    with b3:
        kpi("Test ile seçim", "HAYIR")
    with b4:
        kpi("Korumalı dosyalar", "DEĞİŞMEDİ")

    b5, b6, b7, b8 = st.columns(4)
    with b5:
        kpi("Tekrarlanabilirlik", "GEÇTİ")
    with b6:
        kpi("Tahmin sızıntısı", "GEÇTİ")
    with b7:
        kpi("Üretim tahmini", "HAZIR DEĞİL")
    with b8:
        kpi("Karıştırma (shuffle)", "Yok")

    st.markdown("**Tahmin anında özellik uygunluğu (187 model kolonu)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Güvenli (SAFE)", "106")
    with c2:
        kpi("Bilinmiyor (UNKNOWN)", "6")
    with c3:
        kpi("Yasak (FORBIDDEN)", "75")

    audit = load_csv(str(FORECAST_AUDIT))
    if audit is None:
        missing(FORECAST_AUDIT)
        return
    show = st.multiselect(
        "Gösterilecek sınıflar",
        options=["SAFE", "UNKNOWN", "FORBIDDEN", "CONDITIONAL"],
        default=["UNKNOWN", "FORBIDDEN"],
        format_func=lambda x: {
            "SAFE": "Güvenli",
            "UNKNOWN": "Bilinmiyor",
            "FORBIDDEN": "Yasak",
            "CONDITIONAL": "Koşullu",
        }.get(x, x),
    )
    view = audit[audit["classification_strict"].isin(show)] if show else audit
    cols = [c for c in ("feature", "feature_group", "classification_strict", "notes") if c in view.columns]
    shown = view[cols].rename(
        columns={
            "feature": "Özellik",
            "feature_group": "Grup",
            "classification_strict": "Sınıf",
            "notes": "Not",
        }
    ) if cols else view
    st.dataframe(shown, hide_index=True, use_container_width=True)


def page_info() -> None:
    st.subheader("Proje bilgisi")
    st.write(
        "Bu proje, İspanya gün öncesi elektrik piyasası fiyatını (`price day ahead`) "
        "geçmiş ihale fiyatları, yük/yenilenebilir gün-öncesi tahminleri, "
        "üretim geçmişi ve hava durumu özellikleriyle tahmin eden sızıntısız "
        "bir zaman serisi ML hattıdır. Kripto veya ETH/USDT ile ilgisi yoktur."
    )
    st.markdown(
        """
**Hat**

Veri → Kalite → Birleştirme → Özellik mühendisliği → Zaman bölmesi → Baseline →
Walk-forward doğrulama → Hiperparametre → Artık analizi →
Açıklanabilirlik → Final model → Kilitli test değerlendirme → 24s tahmin denetimi
"""
    )
    status = pd.DataFrame(
        {
            "Aşama": [
                "Veri hattı",
                "Sızıntı denetimi",
                "Özellik mühendisliği",
                "Zaman serisi CV",
                "Model seçimi",
                "Açıklanabilirlik",
                "Final değerlendirme",
                "24s üretim tahmini",
                "Pano",
            ],
            "Durum": [
                "Tamam",
                "Tamam",
                "Tamam",
                "Tamam",
                "Tamam",
                "Tamam",
                "Tamam",
                "Hazır değil",
                "Tamam",
            ],
        }
    )
    st.dataframe(status, hide_index=True, use_container_width=True)
    st.markdown(
        """
**Karşılaştırılan modeller (sunum sırası)**

1. Ridge  
2. LightGBM  
3. XGBoost  
4. ARIMA (Ridge+METHOD_B artığı, ARIMA(1,0,0))  
5. Ridge+B+AR(1) — kilitli teslimat
"""
    )


def main() -> None:
    st.set_page_config(page_title="Elektrik Piyasası Fiyat Tahmini", layout="wide")
    st.title("Elektrik Piyasası Fiyat Tahmini")
    st.caption("İspanya gün öncesi piyasası · sızıntısız zaman serisi hattı")

    page = st.sidebar.radio(
        "Menü",
        (
            "Genel bakış",
            "Model karşılaştırması",
            "Model performansı",
            "Hata analizi",
            "Açıklanabilirlik",
            "24 saatlik tahmin",
            "Veri ve sızıntı denetimi",
            "Proje bilgisi",
        ),
    )
    st.sidebar.markdown("**Kilitli model**")
    st.sidebar.write("1 Ridge · 2 LightGBM · 3 XGBoost")
    st.sidebar.write("4 ARIMA · 5 Ridge+B+AR(1)")
    st.sidebar.write("Kilitli: Ridge+B+AR(1)")
    st.sidebar.write("Durum: KİLİTLİ")
    st.sidebar.error("24s tahmin: üretime hazır değil")

    pages = {
        "Genel bakış": page_overview,
        "Model karşılaştırması": page_compare,
        "Model performansı": page_performance,
        "Hata analizi": page_errors,
        "Açıklanabilirlik": page_explain,
        "24 saatlik tahmin": page_forecast,
        "Veri ve sızıntı denetimi": page_audit,
        "Proje bilgisi": page_info,
    }
    pages[page]()


if __name__ == "__main__":
    main()
