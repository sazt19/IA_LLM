"""
Dashboard de estadísticas con datos sintéticos
------------------------------------------------
Ejecutar con:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Datos Sintéticos",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------
# Generación de datos sintéticos (cacheada para no regenerar en cada
# interacción, salvo que cambien los parámetros)
# ------------------------------------------------------------------
@st.cache_data
def generar_datos(n_filas: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    categorias = ["Electrónica", "Ropa", "Hogar", "Deportes", "Alimentos"]
    regiones = ["Norte", "Sur", "Este", "Oeste", "Centro"]

    fechas = pd.date_range(end=pd.Timestamp.today(), periods=n_filas, freq="D")
    fechas = rng.choice(fechas, size=n_filas, replace=True)

    df = pd.DataFrame({
        "fecha": fechas,
        "categoria": rng.choice(categorias, size=n_filas, p=[0.25, 0.2, 0.2, 0.15, 0.2]),
        "region": rng.choice(regiones, size=n_filas),
        "ventas": rng.gamma(shape=2.0, scale=150, size=n_filas).round(2),
        "unidades": rng.poisson(lam=8, size=n_filas),
        "satisfaccion": rng.normal(loc=4.2, scale=0.6, size=n_filas).clip(1, 5).round(1),
        "costo": rng.gamma(shape=1.8, scale=90, size=n_filas).round(2),
    })

    df["ganancia"] = (df["ventas"] - df["costo"]).round(2)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Barra lateral: parámetros de generación y filtros
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración de datos")

n_filas = st.sidebar.slider("Cantidad de registros", 100, 20000, 3000, step=100)
semilla = st.sidebar.number_input("Semilla aleatoria", value=42, step=1)

df = generar_datos(n_filas, semilla)

st.sidebar.header("🔍 Filtros")

rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(df["fecha"].min().date(), df["fecha"].max().date()),
)

categorias_sel = st.sidebar.multiselect(
    "Categorías", options=sorted(df["categoria"].unique()),
    default=sorted(df["categoria"].unique()),
)

regiones_sel = st.sidebar.multiselect(
    "Regiones", options=sorted(df["region"].unique()),
    default=sorted(df["region"].unique()),
)

# Aplicar filtros
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    inicio, fin = rango_fechas
else:
    inicio, fin = df["fecha"].min().date(), df["fecha"].max().date()

df_filtrado = df[
    (df["fecha"].dt.date >= inicio)
    & (df["fecha"].dt.date <= fin)
    & (df["categoria"].isin(categorias_sel))
    & (df["region"].isin(regiones_sel))
]

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("📊 Dashboard de Estadísticas — Datos Sintéticos")
st.caption("Datos generados aleatoriamente con NumPy únicamente con fines demostrativos.")

if df_filtrado.empty:
    st.warning("No hay datos con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# ------------------------------------------------------------------
# KPIs principales
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Ventas totales", f"${df_filtrado['ventas'].sum():,.0f}")
col2.metric("Ganancia total", f"${df_filtrado['ganancia'].sum():,.0f}")
col3.metric("Unidades vendidas", f"{df_filtrado['unidades'].sum():,}")
col4.metric("Satisfacción promedio", f"{df_filtrado['satisfaccion'].mean():.2f} / 5")

st.markdown("---")

# ------------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Tendencia temporal", "🏷️ Por categoría", "🗺️ Por región", "📋 Datos y estadísticas"]
)

with tab1:
    st.subheader("Ventas y ganancia a lo largo del tiempo")
    df_diario = (
        df_filtrado.groupby(df_filtrado["fecha"].dt.date)[["ventas", "ganancia"]]
        .sum()
        .reset_index()
        .rename(columns={"fecha": "fecha"})
    )
    fig_tendencia = px.line(
        df_diario, x="fecha", y=["ventas", "ganancia"],
        labels={"value": "Monto ($)", "fecha": "Fecha", "variable": "Métrica"},
        title="Evolución diaria de ventas y ganancia",
    )
    st.plotly_chart(fig_tendencia, use_container_width=True)

    fig_unidades = px.bar(
        df_filtrado.groupby(df_filtrado["fecha"].dt.to_period("M").astype(str))["unidades"]
        .sum().reset_index().rename(columns={"fecha": "mes"}),
        x="fecha", y="unidades",
        labels={"fecha": "Mes", "unidades": "Unidades vendidas"},
        title="Unidades vendidas por mes",
    )
    st.plotly_chart(fig_unidades, use_container_width=True)

with tab2:
    st.subheader("Desempeño por categoría")
    c1, c2 = st.columns(2)

    resumen_cat = (
        df_filtrado.groupby("categoria")
        .agg(ventas=("ventas", "sum"), ganancia=("ganancia", "sum"), unidades=("unidades", "sum"))
        .reset_index()
        .sort_values("ventas", ascending=False)
    )

    with c1:
        fig_cat_ventas = px.bar(
            resumen_cat, x="categoria", y="ventas", color="categoria",
            title="Ventas totales por categoría",
        )
        st.plotly_chart(fig_cat_ventas, use_container_width=True)

    with c2:
        fig_cat_pie = px.pie(
            resumen_cat, names="categoria", values="unidades",
            title="Distribución de unidades vendidas",
        )
        st.plotly_chart(fig_cat_pie, use_container_width=True)

    fig_satisfaccion = px.box(
        df_filtrado, x="categoria", y="satisfaccion", color="categoria",
        title="Distribución de satisfacción por categoría",
    )
    st.plotly_chart(fig_satisfaccion, use_container_width=True)

with tab3:
    st.subheader("Desempeño por región")
    resumen_region = (
        df_filtrado.groupby("region")
        .agg(ventas=("ventas", "sum"), ganancia=("ganancia", "sum"), unidades=("unidades", "sum"))
        .reset_index()
        .sort_values("ventas", ascending=False)
    )

    fig_region = px.bar(
        resumen_region, x="region", y=["ventas", "ganancia"], barmode="group",
        title="Ventas y ganancia por región",
    )
    st.plotly_chart(fig_region, use_container_width=True)

    fig_heatmap = px.density_heatmap(
        df_filtrado, x="region", y="categoria", z="ventas", histfunc="sum",
        title="Ventas por región y categoría",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.subheader("Estadísticas descriptivas")
    st.dataframe(df_filtrado.describe(include="number").T, use_container_width=True)

    st.subheader("Datos filtrados")
    st.dataframe(df_filtrado, use_container_width=True, height=400)

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name="datos_sinteticos.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Generado con Streamlit · Datos 100% sintéticos, sin fuentes externas.")
