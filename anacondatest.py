import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# Streamlit Dashboard adaptado al dataset de Reclamos 2023-2025
# ==================================================

st.set_page_config(layout="wide", page_title="Dashboard Reclamos 2023-2025")

st.title("📊 Dashboard Reclamos (2023-2025) - Sociedad Eléctrica del Sur Oeste S.A")
st.write("Provincias del departamento de Arequipa")
# =====================
# 1. Cargar Dataset desde URL
# =====================
url = "https://datosabiertos.gob.pe/sites/default/files/DataSet_Reclamos2023-2025.csv"

@st.cache_data
def load_data(url):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    engines = ["c", "python"]
    for enc in encodings:
        for eng in engines:
            try:
                kwargs = {
                    "encoding": enc,
                    "engine": eng,
                    "on_bad_lines": "warn",
                    "dtype": {"CodigoReclamo": "string"}  # 🚨 bigint seguro como string
                }
                if eng != "python":
                    kwargs["low_memory"] = False  # solo válido con engine="c"

                df = pd.read_csv(url, **kwargs)

                st.write(f"✅ Cargado con encoding={enc}, engine={eng}")
                df.columns = [c.strip() for c in df.columns]

                # Convertir CodigoReclamo a Int64 si es posible
                if "CodigoReclamo" in df.columns:
                    df["CodigoReclamo"] = pd.to_numeric(
                        df["CodigoReclamo"], errors="coerce"
                    ).astype("Int64")

                # Parsear fechas
                for date_col in ["FechaCreacion", "FechaResolucion"]:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(
                            df[date_col], dayfirst=True, errors="coerce"
                        )

                # Calcular tiempo de resolución
                if "FechaCreacion" in df.columns and "FechaResolucion" in df.columns:
                    df["TiempoResolucionDias"] = (
                        (df["FechaResolucion"] - df["FechaCreacion"]).dt.total_seconds() / (3600 * 24)
                    )

                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.write(f"Error con encoding={enc}, engine={eng}: {e}")
                continue

    st.error("❌ No se pudo cargar el archivo con ninguno de los encodings probados.")
    return None



df = load_data(url)
if df is None:
    st.stop()

st.subheader("📌 Vista previa del dataset")
st.dataframe(df.head(20))

# =====================
# 2. Resumen rápido y limpieza
# =====================
st.subheader("🧾 Resumen rápido & Limpieza")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Registros", f"{len(df):,}")
    st.metric("Columnas", f"{df.shape[1]}")
with col2:
    missing_tot = df.isna().sum().sum()
    st.metric("Valores faltantes (tot)", f"{missing_tot:,}")
with col3:
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    st.write("**Columnas fecha detectadas:**")
    st.write(date_cols if date_cols else "Ninguna detectada")

summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(2)
})
st.dataframe(summary.sort_values("pct_missing", ascending=False))

# =====================
# 4. KPIs específicos para reclamos
# =====================
st.subheader("📈 KPIs - Reclamos")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Reclamos", f"{len(df):,}")
with col2:
    atendidos = df[df['NombreSituacionReclamo'].str.upper().str.contains('ATENDIDO', na=False)] if 'NombreSituacionReclamo' in df.columns else df
    st.metric("Atendidos (count)", f"{len(atendidos):,}")
with col3:
    # % resueltos según NombreTipoResolucionReclamo (si existe)
    resolved_mask = df['NombreTipoResolucionReclamo'].notna() if 'NombreTipoResolucionReclamo' in df.columns else pd.Series([False] * len(df))
    pct_resolved = (resolved_mask.sum() / len(df) * 100) if len(df) > 0 else 0
    st.metric("% con tipo de resolución", f"{pct_resolved:.2f}%")
with col4:
    if 'TiempoResolucionDias' in df.columns:
        st.metric("Tiempo resolución mediana (d)", f"{df['TiempoResolucionDias'].median():.2f}")
    else:
        st.write("Sin datos de tiempo")

# =====================
# 5. Series temporales y distribución por Periodo + FechaResolucion
# =====================
st.subheader("📅 Series temporales y distribución (Resolución vs Periodo)")

if "Periodo" in df.columns and "FechaResolucion" in df.columns:
    # Agrupamos por Periodo (AAAAMM) y contamos cuántos reclamos se resolvieron en ese periodo
    temp = df.copy()
    temp["Periodo"] = temp["Periodo"].astype(str)

    resol_por_periodo = (
        temp.dropna(subset=["FechaResolucion"])
        .groupby(["Periodo"])
        .agg({"FechaResolucion": "count"})
        .reset_index()
        .rename(columns={"FechaResolucion": "ReclamosResueltos"})
        .sort_values("Periodo")
    )

    fig_resol = px.bar(
        resol_por_periodo,
        x="Periodo",
        y="ReclamosResueltos",
        title="Reclamos resueltos por Periodo",
    )
    st.plotly_chart(fig_resol, use_container_width=True)

    # Evolución mensual de resoluciones (timeline por FechaResolucion)
    series = (
        temp.dropna(subset=["FechaResolucion"])
        .set_index("FechaResolucion")
        .resample("M")
        .size()
        .reset_index(name="count")
    )
    series["Mes"] = series["FechaResolucion"].dt.to_period("M").astype(str)

    fig_series = px.line(
        series,
        x="Mes",
        y="count",
        title="Evolución mensual de reclamos resueltos",
        markers=True,
    )
    st.plotly_chart(fig_series, use_container_width=True)

elif "Periodo" in df.columns:
    periodo_counts = df.groupby("Periodo").size().reset_index(name="count").sort_values("Periodo")
    fig_periodo = px.bar(periodo_counts, x="Periodo", y="count", title="Reclamos ingresados por Periodo")
    st.plotly_chart(fig_periodo, use_container_width=True)

elif "FechaResolucion" in df.columns:
    series = df.dropna(subset=["FechaResolucion"]).set_index("FechaResolucion").resample("M").size().reset_index(name="count")
    series["Mes"] = series["FechaResolucion"].dt.to_period("M").astype(str)
    fig_series = px.line(series, x="Mes", y="count", title="Reclamos resueltos por mes", markers=True)
    st.plotly_chart(fig_series, use_container_width=True)

else:
    st.info("No hay columna Periodo ni FechaResolucion para series temporales")

# =====================
# 6. Distribución por categorías (provincias, tipo, forma)
# =====================
st.subheader("📊 Distribución por categorías")
col1, col2 = st.columns(2)
with col1:
    if 'NombreProvincia' in df.columns:
        prov = df['NombreProvincia'].value_counts().reset_index()
        prov.columns = ['NombreProvincia', 'count']
        fig_prov = px.bar(prov.head(20), x='NombreProvincia', y='count', title='Top 20 Provincias por reclamos')
        st.plotly_chart(fig_prov, use_container_width=True)
    else:
        st.info('No hay columna NombreProvincia')
with col2:
    if 'NombreSituacionReclamo' in df.columns:
        sit = df['NombreSituacionReclamo'].value_counts().reset_index()
        sit.columns = ['NombreSituacionReclamo', 'count']
        fig_sit = px.pie(sit, names='NombreSituacionReclamo', values='count', title='Situación Reclamo')
        st.plotly_chart(fig_sit, use_container_width=True)
    else:
        st.info('No hay columna NombreSituacionReclamo')

# =====================
st.subheader("📋 Agrupaciones / Tablas Dinámicas")
if 'NombreProvincia' in df.columns and 'NombreClaseReclamo' in df.columns:
    pivot = df.groupby(['NombreProvincia', 'NombreClaseReclamo']).size().reset_index(name='n_reclamos')
    # mostrar top por provincia
    st.dataframe(pivot.sort_values('n_reclamos', ascending=False).head(200))
    top_piv = pivot.groupby('NombreProvincia')['n_reclamos'].sum().reset_index().sort_values('n_reclamos', ascending=False)
    fig_topprov = px.bar(top_piv.head(20), x='NombreProvincia', y='n_reclamos', title='Reclamos totales por Provincia')
    st.plotly_chart(fig_topprov, use_container_width=True)
else:
    st.info('Se requieren NombreProvincia y NombreClaseReclamo para tablas agrupadas')


# =====================
# Fin 
# =====================
