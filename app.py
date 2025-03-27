import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.utils import resample
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.figure_factory as ff

# ================================,
# Configuración inicial de la App
# ================================
st.set_page_config(
    page_title="🔥 Dashboard Acero-Aranceles 2025 👽",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# ================================
# Barra lateral - Configuración
# ================================
st.sidebar.header("⚙️ Configuración del Espacio")
with st.sidebar:
    start_date = st.date_input("📅 Fecha de inicio", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("📅 Fecha final", value=pd.to_datetime("today"))

    # Definir tickers
    tickers = {
        "Ternium México": "TX.MX",
        "Grupo Simec": "SIMECB.MX",
        "Industrias CH": "ICHB.MX",
        "Orbia Advance Corporation": "ORBIA.MX",
        "Grupo Industrial Saltillo": "GISSAA.MX"

    }

    # Selección de empresas
    selected_companies = st.multiselect(
        "🏭 Selecciona empresas",
        options=list(tickers.keys()),
        default=list(tickers.keys())
    )

    if len(selected_companies) == 0:
        st.error("⚠️ Por favor, selecciona al menos una acción para continuar ⚠️")
        st.stop()




# Descargar datos históricos
@st.cache_data
def load_data(tickers, start_date, end_date):
    data_dict = {}
    for empresa, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date)
        data_dict[empresa] = data
    return data_dict

data_dict = load_data(tickers, start_date, end_date)

# Fechas importantes para líneas de eventos
event_dates = {
    "Anuncio de aranceles (Trump)": "2025-02-09",
    "Aranceles impuestos (USA)": "2025-03-12",
    "Decisión pendiente (Sheinbaum)": "2025-04-02"
}

# ================================
# Función para líneas de eventos
# ================================
def add_event_lines(fig):
    offset = 0.0
    for label, date_str in event_dates.items():
        x_val = pd.to_datetime(date_str).to_pydatetime()
        fig.add_vline(x=x_val, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=x_val,
            y=1.05 + offset,
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="center",
            font=dict(color="red", size=10)
        )
        offset += 0.05

# ================================
# Contenido principal con pestañas
# ================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏠 Contexto",
    "📉 Retornos Diarios",
    "🚀 Retornos Acumulados",
    "💵 Cambio Absoluto",
    "📊 Distribución Retornos",
    "🧮 Estadísticas",
    "💼 Carteras 2025",
    "💰 Cartera Extra",
    "🧠 Pronóstico",
    "📅 Notas"

])

# ================================
# Pestaña 0: Contexto y Precios
# ================================
with tab0:
    st.markdown("""
    ## 📈 Bienvenido al Campo de Batalla Financiero ⚔️
    **Contexto clave de 2025:**
    📅 **9/feb**: Trump inicia guerra comercial 💣 → 📅 **12/mar**: EE.UU. impone 25% arancel ⚠️
    📅 **2/abr**: Sheinbaum contraataca 🇲🇽 → 📉📈 Acciones en modo montaña rusa 🎢
    """)

    # Gráfico de precios de cierre
    close_prices = pd.concat([data_dict[empresa]["Close"] for empresa in selected_companies], axis=1)
    close_prices.columns = selected_companies

    fig_price = go.Figure()
    for empresa in close_prices.columns:
        fig_price.add_trace(go.Scatter(x=close_prices.index, y=close_prices[empresa], mode='lines', name=empresa))
    add_event_lines(fig_price)
    fig_price.update_layout(title="🔥 Evolución del Precio de Cierre (MXN)", height=600)
    st.plotly_chart(fig_price, use_container_width=True)

    # Indicadores de últimos precios
    st.subheader("💰 Últimos Precios de Cierre (MXN)")
    cols = st.columns(len(selected_companies))
    for idx, empresa in enumerate(selected_companies):
        last_price = close_prices[empresa].iloc[-1]       #DADO QUE EL ACTIVO TARDA EN CAMBIAR DE PRECIO, NOS VAMOS A ILOC 1 PARA QUE NO MUESTRE VALORES NaN
        cols[idx].metric(
            label=f"{empresa} {'📈' if last_price > close_prices[empresa].iloc[-3] else '📉'}",
            value=f"${last_price:,.2f} MXN"
        )

    # Tabla de precios
    st.subheader("📋 Tabla Histórica de Precios")
    st.dataframe(close_prices.tail().style.format("${:,.2f}"), height=200)

# ================================
# Pestaña 1: Retornos Diarios
# ================================
with tab1:
    st.markdown("## 📉📈 Retornos Diarios: ¿Dónde latió fuerte el corazón bursátil? 💓")

    returns = close_prices.pct_change().dropna()

    # Gráfico de retornos
    fig_returns = go.Figure()
    for empresa in returns.columns:
        fig_returns.add_trace(go.Scatter(x=returns.index, y=returns[empresa], mode='lines', name=empresa))
    add_event_lines(fig_returns)
    fig_returns.update_layout(title="🎢 Evolución de Retornos Diarios (%)", height=600)
    st.plotly_chart(fig_returns, use_container_width=True)

    # Indicadores de últimos retornos
    st.subheader("🔄 Últimos Retornos Diarios")
    cols = st.columns(len(selected_companies))
    for idx, empresa in enumerate(selected_companies):
        last_return = returns[empresa].iloc[-1] * 100   #DADO QUE EL ACTIVO TARDA EN CAMBIAR DE PRECIO, NOS VAMOS A ILOC 2 PARA QUE NO MUESTRE VALORES NaN
        cols[idx].metric(
            label=f"{empresa} {'🟢' if last_return > 0 else '🔴'}",
            value=f"{last_return:.2f}%"
        )

    # Tabla de retornos
    st.subheader("📋 Tabla de Retornos Diarios")
    st.dataframe(returns.tail().style.format("{:.2%}"), height=200)

# ================================
# Pestaña 2: Retornos Acumulados
# ================================
with tab2:
    st.markdown("## 🚀 Retornos Acumulados: El viaje completo del capital 🧳")

    cumulative_returns = (1 + returns).cumprod() - 1

    # Gráfico acumulado
    fig_cum = go.Figure()
    for empresa in cumulative_returns.columns:
        fig_cum.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[empresa], mode='lines', name=empresa))
    add_event_lines(fig_cum)
    fig_cum.update_layout(title="📦 Retornos Acumulados desde Inicio (%)", height=600)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Indicadores acumulados
    st.subheader("🧮 Retorno Total Acumulado")
    cols = st.columns(len(selected_companies))
    for idx, empresa in enumerate(selected_companies):
        total_return = cumulative_returns[empresa].iloc[-1] * 100  #DADO QUE EL ACTIVO TARDA EN CAMBIAR DE PRECIO, NOS VAMOS A ILOC 1 PARA QUE NO MUESTRE VALORES NaN
        cols[idx].metric(
            label=f"{empresa} {'🚀' if total_return > 0 else '💣'}",
            value=f"{total_return:.2f}%"
        )

    # Tabla acumulada
    st.subheader("📋 Tabla de Retornos Acumulados")
    st.dataframe(cumulative_returns.tail().style.format("{:.2%}"), height=200)

# ================================
# Pestaña 3: Cambio Absoluto
# ================================
with tab3:
    st.markdown("## 💵 Cambio Absoluto: ¿Cuánto latió el precio cada día? 💓")

    price_change = close_prices.diff().dropna()

    # Gráfico de barras
    fig_price_change = go.Figure()
    for empresa in price_change.columns:
        fig_price_change.add_trace(go.Bar(
            x=price_change.index,
            y=price_change[empresa],
            name=empresa
        ))
    add_event_lines(fig_price_change)
    fig_price_change.update_layout(
        title="🔄 Cambio Diario Absoluto en Pesos",
        height=600,
        barmode='group'
    )
    st.plotly_chart(fig_price_change, use_container_width=True)

    # Indicadores de último cambio
    st.subheader("💸 Último Cambio Diario (MXN)")
    cols = st.columns(len(selected_companies))
    for idx, empresa in enumerate(selected_companies):
        last_change = price_change[empresa].iloc[-1]
        cols[idx].metric(
            label=f"{empresa} {'💹' if last_change > 0 else '🔻'}",
            value=f"${last_change:,.2f}"
        )

    # Tabla de cambios
    st.subheader("📋 Tabla de Cambios Absolutos")
    st.dataframe(price_change.tail().style.format("${:,.2f}"), height=200)

# ================================
# Pestaña 4: Distribución Retornos
# ================================
with tab4:
    st.markdown("## 📊 Distribución de Retornos: ¿Dónde se esconden las sorpresas? 🎭")

    # Histograma
    fig_hist = go.Figure()
    for empresa in returns.columns:
        fig_hist.add_trace(go.Histogram(
            x=returns[empresa],
            name=empresa,
            opacity=0.5,
            histnorm='percent'
        ))
    fig_hist.update_layout(
        title="📦 Distribución de Retornos Diarios (%)",
        height=600,
        barmode='overlay'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Indicadores de volatilidad
    st.subheader("🌪️ Volatilidad Reciente (Últimos 30 días)")
    cols = st.columns(len(selected_companies))
    for idx, empresa in enumerate(selected_companies):
        vol = returns[empresa].tail(30).std() * 100
        cols[idx].metric(
            label=f"{empresa}",
            value=f"{vol:.2f}%",
            delta="Alta volatilidad" if vol > 5 else "Estable"
        )

    # Tabla de retornos
    st.subheader("📋 Datos Crudos de Retornos")
    st.dataframe(returns.tail().style.format("{:.2%}"), height=200)

# ================================
# Pestaña 5: Estadísticas Rendimiento
# ================================
with tab5:
    st.markdown("## 🧮 Estadísticas Clave: El termómetro del rendimiento 🌡️")

    # Calcular estadísticas
    stats = pd.DataFrame(columns=[
        "Return Promedio Diario",
        "Volatilidad Diaria",
        "Sharpe Ratio",
        "Máxima Caída"
    ])

    for empresa in selected_companies:
        daily_returns = returns[empresa]
        stats.loc[empresa] = [
            daily_returns.mean() * 100,          # Return promedio
            daily_returns.std() * 100,           # Volatilidad
            daily_returns.mean() / daily_returns.std(),  # Sharpe
            (daily_returns.cummin()).min() * 100 # Máxima caída
        ]

    # Mostrar métricas en cards
    st.subheader("📌 Resumen Ejecutivo")
    cols = st.columns(len(selected_companies))
    for idx, empresa in enumerate(selected_companies):
        with cols[idx]:
            st.metric(label=f"🏆 Mejor día {empresa}",
                     value=f"{returns[empresa].max()*100:.2f}%")
            st.metric(label=f"💥 Peor día {empresa}",
                     value=f"{returns[empresa].min()*100:.2f}%")

    # Tabla de estadísticas
    st.subheader("📋 Tabla Completa de Métricas")
    st.dataframe(stats.style.format({
        "Return Promedio Diario": "{:.2f}%",
        "Volatilidad Diaria": "{:.2f}%",
        "Sharpe Ratio": "{:.2f}",
        "Máxima Caída": "{:.2f}%"
    }), height=400)





# ================================
# Pestaña 7: Optimización  NPEB de Portafolio
# ================================
with tab7:
    st.markdown("## 🧠 Portafolio Óptimo Bayesiano (NPEB)")
    st.markdown(r"""
    ### 📚 Modelo de Optimización Bayesiana
    Se estima el portafolio óptimo utilizando el método **No Parametric Empirical Bayes (NPEB)**, que incorpora la incertidumbre en los parámetros a través de bootstrap.

    **Estimación de parámetros:**

    Se generan $B$ muestras bootstrap de los retornos para calcular:

    - La media:
      $$\mu_n = \frac{1}{B}\sum_{b=1}^{B} \mu_b$$

    - La matriz de segundo momento:
      $$V_n = \frac{1}{B}\sum_{b=1}^{B} \Big(\Sigma_b + \mu_b \mu_b^T\Big)$$

    donde $\mu_b$ y $\Sigma_b$ son la media y covarianza obtenidas de cada muestra.

    **Problema de optimización:**

    $$\min_{w}\; \lambda\, w^T V_n w \;-\; \eta\, w^T \mu_n$$

    sujeto a:
    $$\sum_{i} w_i = 1,\quad w_i \geq 0.$$

    **Referencia:**
    Efron, B. (2013). *Bayesian inference and the parametric bootstrap*. arXiv preprint arXiv:1301.2936. [Disponible en](https://arxiv.org/abs/1301.2936)
    """)


    # Parámetros para el modelo (λ, B, etc.)
    st.markdown("### ⚙️ Selección de Parámetros del Modelo Bayesiano")
    col_par, col_boot = st.columns(2)
    with col_par:
        lam = st.slider("Aversión al riesgo (λ)", 0.1, 20.0, 10.0, 0.1,
                        help="Valores más altos = Mayor penalización a la volatilidad [Recomendado: 5-15]")
    with col_boot:
        B = st.slider("Muestras bootstrap (B)", 100, 2000, 500, 100,
                      help="Número de simulaciones para estimación [Recomendado: 500-1000]")

    # En el código de referencia, NO se permite venta en corto (w >= 0).
    allow_short = False

    # 1) Verificar datos históricos disponibles
    min_date = min([data.index[0] for data in data_dict.values()]).date()
    max_date = max([data.index[-1] for data in data_dict.values()]).date()

    # 2) Ajustar la fecha por defecto al rango [min_date, max_date]
    default_start = pd.to_datetime("2020-01-04").date()
    if default_start < min_date:
        default_start = min_date
    default_end = pd.to_datetime("today").date()
    if default_end > max_date:
        default_end = max_date

    st.warning(f"""
    📅 **Rango de datos disponible:**
    {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}
    """)

    # 3) Selección de fechas para entrenamiento del portafolio
    col1, col2 = st.columns(2)
    with col1:
        port_start = st.date_input(
            "Fecha inicial entrenamiento",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        port_end = st.date_input(
            "Fecha final entrenamiento",
            value=default_end,
            min_value=min_date,
            max_value=max_date
        )

    # 4) Cargar y preparar datos

    # Quita o comenta el decorador para evitar datos cacheados
    # @st.cache_data(hash_funcs={list: lambda l: tuple(l)})
    def prepare_portfolio_data(tickers, selected_companies, start_date, end_date):
        data_prices = yf.download([tickers[e] for e in selected_companies],
                                  start=start_date, end=end_date)["Close"]
        data_prices = data_prices.dropna(axis=0, how="any")
        returns = data_prices.pct_change().dropna()
        return returns




    try:
        #returns = prepare_portfolio_data(tickers, port_start, port_end)

        returns = prepare_portfolio_data(tickers, selected_companies, port_start, port_end)
       # returns = prepare_portfolio_data(tickers, tuple(selected_companies), port_start, port_end)

        if returns.empty:
            st.error("❌ No hay suficientes datos para el periodo seleccionado")
            st.stop()

        # === 5) Estimación de parámetros (NPEB) ===

        # @st.cache_data
        def estimate_parameters(_returns, B_samples):
            n, m = _returns.shape
            mu_boot = []
            Sigma_boot = []
            returns_np = _returns.values

            for _ in range(B_samples):
                sample = resample(returns_np, n_samples=n, replace=True)
                mu_b = np.mean(sample, axis=0)
                Sigma_b = np.cov(sample, rowvar=False)
                mu_boot.append(mu_b)
                Sigma_boot.append(Sigma_b)

            mu_n = np.mean(mu_boot, axis=0)
            V_n = np.mean([
                Sigma_boot[b] + np.outer(mu_boot[b], mu_boot[b])
                for b in range(B_samples)
            ], axis=0)
            return mu_n, V_n








        mu_n, V_n = estimate_parameters(returns, B)
        m = len(mu_n)

        # === 6) Función de optimización bayesiana (con w >= 0) ===
        def solve_bayesian_portfolio(mu, V, lam, eta):
            w = cp.Variable(m)
            objective = cp.Minimize(lam * cp.quad_form(w, V) - eta * (mu @ w))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP)
            if prob.status == "optimal":
                return w.value
            else:
                return None

        # === 7) Escaneo de η para replicar EXACTO el código de referencia ===
        eta_values = np.linspace(0.1, 5, 50)
        best_eta = None
        best_w = None
        best_utility = -np.inf

        for e in eta_values:
            w_try = solve_bayesian_portfolio(mu_n, V_n, lam, e)
            if w_try is not None:
                port_ret = returns.values @ w_try
                util = np.mean(port_ret) - lam * np.var(port_ret)
                if util > best_utility:
                    best_utility = util
                    best_eta = e
                    best_w = w_try

        if best_w is None:
            st.error("⚠️ No se pudo encontrar solución óptima. Ajusta los parámetros.")
            st.stop()

        # Comentario indicando la η óptima encontrada.
        st.success("✅ Cartera NPEB optimizada correctamente!")
        st.write(f"**η óptimo encontrado:** {best_eta:.2f}  *(En el paper encontraron: 1.00)*")

        # === 8) Mostrar Resultados Cartera NPEB ===
        port_ret = returns.values @ best_w
        mean_daily = np.mean(port_ret)
        std_daily = np.std(port_ret)
        mean_ann = (1 + mean_daily)**252 - 1
        std_ann = std_daily * np.sqrt(252)
        sharpe = mean_daily/std_daily if std_daily > 0 else float('nan')

        st.markdown("### Resultados Cartera NPEB")
        c1, c2, c3 = st.columns(3)
        c1.metric("Retorno anualizado", f"{mean_ann*100:.2f}%")
        c2.metric("Volatilidad anualizada", f"{std_ann*100:.2f}%")
        c3.metric("Sharpe ratio (diario)", f"{sharpe:.4f}")

        # === 9) Cartera Markowitz (Plug-in) para comparación (w >= 0) ===
        def markowitz_portafolio(mu, Sigma):
            w = cp.Variable(m)
            objective = cp.Minimize(cp.quad_form(w, Sigma))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP)
            if prob.status == "optimal":
                return w.value
            else:
                return None

        mu_sample = returns.mean().values
        Sigma_sample = returns.cov().values
        w_markowitz = markowitz_portafolio(mu_sample, Sigma_sample)

        if w_markowitz is not None:
            port_ret_mk = returns.values @ w_markowitz
            mean_daily_mk = np.mean(port_ret_mk)
            std_daily_mk = np.std(port_ret_mk)
            mean_ann_mk = (1 + mean_daily_mk)**252 - 1
            std_ann_mk = std_daily_mk * np.sqrt(252)
            sharpe_mk = mean_daily_mk/std_daily_mk if std_daily_mk > 0 else float('nan')

            st.markdown("### Cartera Markowitz (Plug-in)")
            cA, cB, cC = st.columns(3)
            cA.metric("Retorno anualizado", f"{mean_ann_mk*100:.2f}%")
            cB.metric("Volatilidad anualizada", f"{std_ann_mk*100:.2f}%")
            cC.metric("Sharpe ratio (diario)", f"{sharpe_mk:.4f}")
        else:
            st.warning("No se pudo resolver la cartera Markowitz con los parámetros actuales.")

        # === 10) Tabla comparativa de Pesos de ambas carteras ===
        st.markdown("### Comparación de Pesos de las Carteras")
        df_comparacion = pd.DataFrame({
            "Activo": selected_companies,
            "Peso NPEB (%)": np.round(best_w*100, 2),
            "Peso Markowitz (%)": np.round(w_markowitz*100, 2) if w_markowitz is not None else np.nan
        })
        st.dataframe(df_comparacion.style.format({
            "Peso NPEB (%)": "{:.2f}",
            "Peso Markowitz (%)": "{:.2f}"
        }), height=300)

        # === 11) Gráfico comparativo de barras de los pesos ===
        fig = go.Figure(data=[
            go.Bar(name="NPEB", x=list(selected_companies), y=best_w*100, marker_color='#1f77b4'),
            go.Bar(name="Markowitz", x=list(selected_companies), y=w_markowitz*100, marker_color='#ff7f0e')
        ])
        fig.update_layout(
            title="🔢 Comparación de Distribución de Pesos",
            yaxis_title="Peso (%)",
            barmode="group",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error en cálculo: {str(e)}")





# ================================
# Pestaña 6: Optimización Portafolios Extras
#   - Portafolio Bayesiano Tradicional (prior definido por el usuario)
#   - Portafolio Risk Parity
#   - Portafolio Sharpe (Máximo Ratio)
#   - Portafolio MinVar (Mínima Varianza)
# ================================
with tab6:
    st.markdown("## 🧮 Optimización Portafolios 2025: Modelos Interactivos 🧮")
    st.markdown("""
    **Contexto:**
    En esta pestaña podrás explorar cuatro nuevos modelos de optimización de portafolios. Cada uno te permite ajustar parámetros clave,
    incluyendo consideraciones de impacto arancelario.
    - **Bayesiano Tradicional:** Ajusta los priors para incorporar expectativas subjetivas (por ejemplo, el impacto de aranceles) y escoge la distribución previa (Normal o T-Student).
    - **Risk Parity:** Igualar la contribución al riesgo de cada activo; además, puedes decidir si incluir activos de cobertura como CETES u Oro.
    - **Sharpe (Máximo Ratio):** Permite personalizar los retornos esperados (ajustados por aranceles) y elegir entre datos históricos o proyecciones bayesianas.
    - **MinVar:** Optimiza para la mínima varianza permitiendo restringir el peso máximo asignado a cada activo y escoger el método de estimación de la matriz de covarianza.
    """)

    # ─────────────────────────────────────────────────────────────────────────────
    # Parámetros Generales para Portafolios Extras
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("🔧 Parámetros Generales para Portafolios Extras", expanded=True):
        # Modo arancelario global: explica cómo se ajustan retornos y volatilidades.
        arancel_mode = st.checkbox("Modo Arancelario 🛃",
                                    help="Si se activa, se reduce el retorno esperado de exportadores (por ejemplo, TX.MX en -15%) y se aumenta la volatilidad de empresas con deuda en USD (por ejemplo, ORBIA.MX +20%).")
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            port_start_ex = st.date_input("Fecha inicial entrenamiento",
                                          value=default_start, min_value=min_date, max_value=max_date, key="port_start_ex")
        with col_gen2:
            port_end_ex = st.date_input("Fecha final entrenamiento",
                                        value=default_end, min_value=min_date, max_value=max_date, key="port_end_ex")
        # Se cargan los datos para la optimización en Extras
       # returns_ex = prepare_portfolio_data(tickers, port_start_ex, port_end_ex)

        returns_ex = prepare_portfolio_data(tickers, selected_companies, port_start_ex, port_end_ex)
        if returns_ex.empty:
            st.error("❌ No hay suficientes datos para el periodo seleccionado en Extras")
            st.stop()

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Portafolio Bayesiano Tradicional 🧠
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 🧠 Portafolio Bayesiano Tradicional (Prior definido por el usuario)")
    with st.expander("Parámetros Bayesiano Tradicional", expanded=True):
        lam_bayes = st.slider("Aversión al riesgo (λ) [Bayesiano]", 0.1, 20.0, 10.0, step=0.1,
                              help="Valor mayor penaliza más la volatilidad. Ajusta este parámetro para controlar la aversión al riesgo.")
        st.markdown("#### 📊 Ajuste de Priors (en %)")
        st.markdown("""
        Los **priors** representan tus expectativas subjetivas sobre los retornos de cada activo.
        Por ejemplo, si crees que los aranceles impactarán negativamente a una empresa, puedes ajustar su prior a un valor negativo.
        *Modifica estos valores según tu análisis o intuición sobre el impacto de aranceles u otros factores.*
        """)
        prior_adjustments = {}
        for empresa in selected_companies:
            # Valor de referencia: para TX.MX se sugiere -15% si Modo Arancelario está activo
            default_prior = -15.0 if arancel_mode and tickers[empresa] == "TX.MX" else 0.0
            prior_adjustments[empresa] = st.number_input(f"Prior para {empresa}", value=default_prior, step=0.1, format="%.2f",
                                                         help="Modifica este valor para ajustar tu expectativa sobre el retorno del activo.")
        st.markdown("#### 🔢 Probabilidad de arancel prolongado")
        prob_arancel = st.slider("Probabilidad de arancel prolongado (%)", 0, 100, 50,
                                 help="Este valor ajusta la volatilidad. Una probabilidad mayor implica un mayor incremento en la incertidumbre.")
        st.markdown("#### 📈 Selección de Distribución Previa")
        prior_distribution = st.selectbox("Distribución previa", options=["Normal", "T-Student"],
                                          help="""
                                          - **Normal:** Asume una distribución simétrica de retornos.
                                          - **T-Student:** Permite colas más pesadas, útil en presencia de eventos extremos o incertidumbre elevada.
                                          """)
        st.info("💡 *Consejo:* Ajusta los priors y la distribución según tu percepción del entorno económico y el impacto de aranceles.")

    # Cálculos para el portafolio Bayesiano
    mu_sample = returns_ex.mean().values            # Promedio muestral
    Sigma_sample = returns_ex.cov().values            # Matriz de covarianza muestral
    # Ajustar los retornos esperados según los priors (convertidos de % a decimal)
    mu_adj = np.array([mu_sample[i] + (prior_adjustments[empresa] / 100.0)
                       for i, empresa in enumerate(selected_companies)])
    # Ajuste de la volatilidad: se incrementa según el slider de probabilidad
    vol_factor = 1 + (prob_arancel / 100.0)
    Sigma_adj = Sigma_sample * vol_factor
    # Ajuste extra: en modo arancelario, para ORBIA.MX se aumenta volatilidad en 20%
    for i, empresa in enumerate(selected_companies):
        if arancel_mode and tickers[empresa] == "ORBIA.MX":
            Sigma_adj[i, :] *= 1.20
            Sigma_adj[:, i] *= 1.20

    m_assets = len(selected_companies)
    def solve_bayes_portfolio(mu, Sigma, lam, eta):
        w = cp.Variable(m_assets)
        objective = cp.Minimize(lam * cp.quad_form(w, Sigma) - eta * (mu @ w))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        return w.value if prob.status == "optimal" else None

    eta_values = np.linspace(0.1, 5, 50)
    best_eta_bayes = None
    best_w_bayes = None
    best_util_bayes = -np.inf
    for eta in eta_values:
        w_try = solve_bayes_portfolio(mu_adj, Sigma_adj, lam_bayes, eta)
        if w_try is not None:
            port_ret = returns_ex.values @ w_try
            util = np.mean(port_ret) - lam_bayes * np.var(port_ret)
            if util > best_util_bayes:
                best_util_bayes = util
                best_eta_bayes = eta
                best_w_bayes = w_try

    if best_w_bayes is None:
        st.error("⚠️ No se pudo optimizar el Portafolio Bayesiano Tradicional. Ajusta los parámetros.")
    else:
        st.success("✅ Portafolio Bayesiano Tradicional optimizado correctamente!")
        st.write(f"**η óptimo encontrado:** {best_eta_bayes:.2f}")
        port_ret_bayes = returns_ex.values @ best_w_bayes
        mean_daily_bayes = np.mean(port_ret_bayes)
        std_daily_bayes = np.std(port_ret_bayes)
        mean_ann_bayes = (1 + mean_daily_bayes) ** 252 - 1
        std_ann_bayes = std_daily_bayes * np.sqrt(252)
        sharpe_bayes = mean_daily_bayes / std_daily_bayes if std_daily_bayes > 0 else float('nan')
        st.markdown("#### Resultados Portafolio Bayesiano Tradicional")
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Retorno anualizado", f"{mean_ann_bayes * 100:.2f}%")
        col_b2.metric("Volatilidad anualizada", f"{std_ann_bayes * 100:.2f}%")
        col_b3.metric("Sharpe ratio", f"{sharpe_bayes:.4f}")
        st.markdown("#### Pesos del Portafolio")
        df_bayes = pd.DataFrame({
            "Activo": selected_companies,
            "Peso (%)": np.round(best_w_bayes * 100, 2)
        })
        st.dataframe(df_bayes.style.format({"Peso (%)": "{:.2f}"}))
        fig_bayes = go.Figure(data=[
            go.Bar(name="Bayesiano Tradicional", x=selected_companies, y=best_w_bayes * 100, marker_color='purple')
        ])
        fig_bayes.update_layout(title="Distribución de Pesos - Portafolio Bayesiano",
                                yaxis_title="Peso (%)", xaxis_tickangle=-45)
        st.plotly_chart(fig_bayes, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Portafolio Risk Parity ⚖️
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### ⚖️ Portafolio Risk Parity")
    with st.expander("Parámetros Risk Parity", expanded=True):
        include_hedge = st.checkbox("Incluir activos de cobertura (CETES, Oro)", value=True,
                                    help="Al activarlo, se incluirán activos considerados como refugio, lo que puede disminuir la concentración de riesgo.")
        max_vol = st.slider("Volatilidad máxima permitida por activo (%)", 1, 20, 10,
                            help="Limita el riesgo individual de cada activo (por ejemplo, CETES u Oro podrían tener menor riesgo).")
        st.info("💡 *Consejo:* Si incluyes activos de cobertura, revisa que su baja volatilidad contribuya a una mejor diversificación del riesgo.")
    # Para Risk Parity se busca igualar la contribución al riesgo de cada activo.
    sigma_assets = np.sqrt(np.diag(Sigma_sample))
    sigma_assets_adj = sigma_assets.copy()
    for i, empresa in enumerate(selected_companies):
        if arancel_mode and tickers[empresa] == "ORBIA.MX":
            sigma_assets_adj[i] *= 1.20
    # Se arma la función objetivo: minimizar la diferencia entre las contribuciones al riesgo.
    w = cp.Variable(m_assets)
    risk_parity_obj = 0
    for i in range(m_assets):
        for j in range(i + 1, m_assets):
            risk_parity_obj += cp.square(w[i] * sigma_assets_adj[i] - w[j] * sigma_assets_adj[j])
    constraints = [cp.sum(w) == 1, w >= 0]
    # Restricción para limitar la volatilidad individual (w[i]*σ_i <= max_vol/100)
    for i in range(m_assets):
        constraints.append(w[i] * sigma_assets_adj[i] <= max_vol / 100)
    prob_rp = cp.Problem(cp.Minimize(risk_parity_obj), constraints)
    prob_rp.solve(solver=cp.OSQP)
    if prob_rp.status != "optimal":
        st.error("⚠️ No se pudo optimizar el Portafolio Risk Parity. Ajusta los parámetros.")
        risk_parity_weights = None
    else:
        risk_parity_weights = w.value
        st.success("✅ Portafolio Risk Parity optimizado correctamente!")
        port_ret_rp = returns_ex.values @ risk_parity_weights
        mean_daily_rp = np.mean(port_ret_rp)
        std_daily_rp = np.std(port_ret_rp)
        mean_ann_rp = (1 + mean_daily_rp) ** 252 - 1
        std_ann_rp = std_daily_rp * np.sqrt(252)
        sharpe_rp = mean_daily_rp / std_daily_rp if std_daily_rp > 0 else float('nan')
        st.markdown("#### Resultados Portafolio Risk Parity")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Retorno anualizado", f"{mean_ann_rp * 100:.2f}%")
        col_r2.metric("Volatilidad anualizada", f"{std_ann_rp * 100:.2f}%")
        col_r3.metric("Sharpe ratio", f"{sharpe_rp:.4f}")
        st.markdown("#### Pesos del Portafolio")
        df_rp = pd.DataFrame({
            "Activo": selected_companies,
            "Peso (%)": np.round(risk_parity_weights * 100, 2)
        })
        st.dataframe(df_rp.style.format({"Peso (%)": "{:.2f}"}))
        fig_rp = go.Figure(data=[
            go.Bar(name="Risk Parity", x=selected_companies, y=risk_parity_weights * 100, marker_color='orange')
        ])
        fig_rp.update_layout(title="Distribución de Pesos - Portafolio Risk Parity",
                             yaxis_title="Peso (%)", xaxis_tickangle=-45)
        st.plotly_chart(fig_rp, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Portafolio Sharpe (Máximo Ratio) 📈
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Portafolio Sharpe (Máximo Ratio)")
    with st.expander("Parámetros Portafolio Sharpe", expanded=True):
        st.markdown("#### 📊 Retornos Esperados Personalizados (en %)")
        st.markdown("""
        Ajusta los retornos esperados para cada activo.
        *Modificar estos valores afecta directamente la asignación del portafolio; por ejemplo, si se reducen los retornos de un activo (por aranceles), este podría recibir menor peso.*
        """)
        custom_returns = {}
        for empresa in selected_companies:
            # Valor base: el retorno muestral en porcentaje
            default_ret = mu_sample[list(selected_companies).index(empresa)] * 100
            # Ajuste de arancel: para TX.MX se reduce en 15% si está activo el modo arancelario
            if arancel_mode and tickers[empresa] == "TX.MX":
                default_ret -= 15
            custom_returns[empresa] = st.number_input(f"Retorno esperado para {empresa}", value=default_ret, step=0.1, format="%.2f",
                                                      help="Ajusta este valor según tus expectativas de rendimiento, considerando el posible impacto de aranceles.")
        use_historical = st.selectbox("Fuente de datos", options=["Históricos", "Proyecciones Bayesianas"],
                                      index=0, help="Selecciona la fuente para calcular los retornos: datos históricos o proyecciones basadas en análisis bayesiano.")
        st.markdown("#### 🎚️ Nivel de Aversión al Riesgo")
        risk_aversion_sharpe = st.slider("Nivel de Aversión al Riesgo", 1, 10, 5,
                                         help="Un valor mayor indica mayor aversión al riesgo, lo que influirá en la asignación del portafolio.")
    # Cálculos para el portafolio Sharpe
    lam_sharpe = risk_aversion_sharpe
    mu_custom = np.array([custom_returns[empresa] / 100.0 for empresa in selected_companies])
    Sigma_custom = Sigma_sample.copy()
    for i, empresa in enumerate(selected_companies):
        if arancel_mode and tickers[empresa] == "ORBIA.MX":
            Sigma_custom[i, :] *= 1.20
            Sigma_custom[:, i] *= 1.20
    def solve_sharpe_portfolio(mu, Sigma, lam, eta):
        w = cp.Variable(m_assets)
        objective = cp.Minimize(lam * cp.quad_form(w, Sigma) - eta * (mu @ w))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)
        return w.value if prob.status == "optimal" else None

    eta_values_sharpe = np.linspace(0.1, 5, 50)
    best_eta_sharpe = None
    best_w_sharpe = None
    best_util_sharpe = -np.inf
    for eta in eta_values_sharpe:
        w_try = solve_sharpe_portfolio(mu_custom, Sigma_custom, lam_sharpe, eta)
        if w_try is not None:
            port_ret = returns_ex.values @ w_try
            util = np.mean(port_ret) - lam_sharpe * np.var(port_ret)
            if util > best_util_sharpe:
                best_util_sharpe = util
                best_eta_sharpe = eta
                best_w_sharpe = w_try

    if best_w_sharpe is None:
        st.error("⚠️ No se pudo optimizar el Portafolio Sharpe. Ajusta los parámetros.")
    else:
        st.success("✅ Portafolio Sharpe optimizado correctamente!")
        st.write(f"**η óptimo encontrado:** {best_eta_sharpe:.2f}")
        port_ret_sharpe = returns_ex.values @ best_w_sharpe
        mean_daily_sharpe = np.mean(port_ret_sharpe)
        std_daily_sharpe = np.std(port_ret_sharpe)
        mean_ann_sharpe = (1 + mean_daily_sharpe) ** 252 - 1
        std_ann_sharpe = std_daily_sharpe * np.sqrt(252)
        sharpe_ratio_sharpe = mean_daily_sharpe / std_daily_sharpe if std_daily_sharpe > 0 else float('nan')
        st.markdown("#### Resultados Portafolio Sharpe")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Retorno anualizado", f"{mean_ann_sharpe * 100:.2f}%")
        col_s2.metric("Volatilidad anualizada", f"{std_ann_sharpe * 100:.2f}%")
        col_s3.metric("Sharpe ratio", f"{sharpe_ratio_sharpe:.4f}")
        st.markdown("#### Pesos del Portafolio")
        df_sharpe = pd.DataFrame({
            "Activo": selected_companies,
            "Peso (%)": np.round(best_w_sharpe * 100, 2)
        })
        st.dataframe(df_sharpe.style.format({"Peso (%)": "{:.2f}"}))
        fig_sharpe = go.Figure(data=[
            go.Bar(name="Portafolio Sharpe", x=selected_companies, y=best_w_sharpe * 100, marker_color='green')
        ])
        fig_sharpe.update_layout(title="Distribución de Pesos - Portafolio Sharpe",
                                 yaxis_title="Peso (%)", xaxis_tickangle=-45)
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Portafolio MinVar (Mínima Varianza) 🛡️
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("### 🛡️ Portafolio MinVar (Mínima Varianza)")
    with st.expander("Parámetros Portafolio MinVar", expanded=True):
        max_weight = st.slider("Restricción de peso máximo por activo (%)", 10, 100, 30, step=5,
                               help="Limita la asignación máxima a cada activo. Por ejemplo, restringir al 30% evita concentrar demasiado riesgo en un solo activo.")
        cov_method = st.selectbox("Método de estimación de matriz de covarianza", options=["Sample", "Ledoit-Wolf"],
                                  index=0, help="Elige entre la matriz de covarianza muestral o una versión robusta (Ledoit-Wolf) que puede mitigar efectos de muestras pequeñas.")
    if cov_method == "Sample":
        Sigma_minvar = Sigma_sample.copy()
    else:
        # Aquí se puede implementar Ledoit-Wolf; se usa Sample como placeholder.
        Sigma_minvar = Sigma_sample.copy()
        st.info("Uso de Sample covariance como aproximación para Ledoit-Wolf (placeholder)")
    w_min = cp.Variable(m_assets)
    objective_min = cp.Minimize(cp.quad_form(w_min, Sigma_minvar))
    constraints_min = [cp.sum(w_min) == 1, w_min >= 0, w_min <= max_weight / 100]
    prob_minvar = cp.Problem(objective_min, constraints_min)
    prob_minvar.solve(solver=cp.OSQP)
    if prob_minvar.status != "optimal":
        st.error("⚠️ No se pudo optimizar el Portafolio MinVar. Ajusta los parámetros.")
        w_minvar = None
    else:
        w_minvar = w_min.value
        st.success("✅ Portafolio MinVar optimizado correctamente!")
        port_ret_minvar = returns_ex.values @ w_minvar
        mean_daily_minvar = np.mean(port_ret_minvar)
        std_daily_minvar = np.std(port_ret_minvar)
        mean_ann_minvar = (1 + mean_daily_minvar) ** 252 - 1
        std_ann_minvar = std_daily_minvar * np.sqrt(252)
        sharpe_minvar = mean_daily_minvar / std_daily_minvar if std_daily_minvar > 0 else float('nan')
        st.markdown("#### Resultados Portafolio MinVar")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Retorno anualizado", f"{mean_ann_minvar * 100:.2f}%")
        col_m2.metric("Volatilidad anualizada", f"{std_ann_minvar * 100:.2f}%")
        col_m3.metric("Sharpe ratio", f"{sharpe_minvar:.4f}")
        st.markdown("#### Pesos del Portafolio")
        df_minvar = pd.DataFrame({
            "Activo": selected_companies,
            "Peso (%)": np.round(w_minvar * 100, 2)
        })
        st.dataframe(df_minvar.style.format({"Peso (%)": "{:.2f}"}))
        fig_minvar = go.Figure(data=[
            go.Bar(name="MinVar", x=selected_companies, y=w_minvar * 100, marker_color='blue')
        ])
        fig_minvar.update_layout(title="Distribución de Pesos - Portafolio MinVar",
                                 yaxis_title="Peso (%)", xaxis_tickangle=-45)
        st.plotly_chart(fig_minvar, use_container_width=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------------



# ================================
# Pestaña 8: Pronóstico Avanzado
# ================================

with tab8:
    st.markdown("## 🧠 Pronóstico Avanzado - Triple Método Predictivo")

    # Selector principal de método
    metodo_pronostico = st.selectbox("🔮 Seleccione Método Predictivo", options=[
        "🎲 Simulación Mejorada de Monte Carlo",
        "🤖 IA - Temporal Fusion Transformer",
        "📉 Modelo Econométrico ARIMA-GARCH"
    ], help="Elija entre métodos cuantitativos modernos para generar pronósticos")

    # =========================================================================
    # Sección 1: Simulación Mejorada de Monte Carlo (Modelo Heston)
    # =========================================================================
    if metodo_pronostico == "🎲 Simulación Mejorada de Monte Carlo":
        st.markdown("### 🎲 Simulación Mejorada de Monte Carlo (Modelo Heston)")
        with st.expander("⚙️ Parámetros del Modelo", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                n_sim = st.slider("Nº de simulaciones", 1000, 50000, 10000, 1000,
                                  help="Mayor número reduce varianza del pronóstico")
            with col2:
                dias_pronostico = st.slider("Días a pronosticar", 1, 365, 30,
                                            help="Horizonte temporal del pronóstico")
            with col3:
                fecha_inicio = st.date_input("Fecha inicio datos",
                                             value=pd.to_datetime("2010-01-04"),
                                             min_value=pd.to_datetime("2010-01-04"))

            col4, col5 = st.columns(2)
            with col4:
                incluir_saltos = st.checkbox("Incluir saltos de volatilidad (Black Swan)",
                                             help="Modela eventos extremos con distribución Poisson")
            with col5:
                lambda_jumps = st.slider("Intensidad saltos", 0.0, 1.0, 0.05, 0.01,
                                           disabled=not incluir_saltos,
                                           help="Frecuencia esperada de eventos extremos")

            # Control de semilla
            col6, col7 = st.columns(2)
            with col6:
                use_custom_seed = st.checkbox("🔒 Usar semilla fija", help="Habilita para resultados reproducibles")
            with col7:
                if use_custom_seed:
                    seed = st.number_input("Semilla personalizada", value=42, min_value=0)
                else:
                    import time
                    seed = int(time.time() * 1000) % (2**32 - 1)

        # Selección de acción
        accion = st.selectbox("📈 Seleccione acción para pronóstico", options=selected_companies)

        # Cargar datos
        data = data_dict[accion]["Close"].loc[pd.to_datetime(fecha_inicio):]
        returns = data.pct_change().dropna()

        if len(data) < 30:
            st.error("❌ Datos insuficientes para el período seleccionado")
            st.stop()

        # Parámetros del modelo Heston
        S0 = float(data.iloc[-1])
        mu = float(returns.mean() * 252)
        v0 = float(returns.var() * 252)
        kappa = 2.0    # Velocidad de reversión a la media
        theta = float(v0)
        sigma_v = 0.3  # Volatilidad de la volatilidad
        rho = -0.7     # Correlación entre precio y volatilidad
        T = dias_pronostico / 252
        n_steps = dias_pronostico  # Se asume un paso diario

        # Simulación Monte Carlo corregida
        @st.cache_data
        def run_heston_simulation(S0, mu, v0, kappa, theta, sigma_v, rho, T, n_steps, n_sim, lambda_jumps, incluir_saltos, seed):
            dt = T / n_steps
            prices = np.zeros((n_steps+1, n_sim))
            volatilities = np.zeros_like(prices)

            prices[0, :] = S0
            volatilities[0, :] = v0

            np.random.seed(seed)
            Z1 = np.random.normal(size=(n_steps, n_sim))
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(n_steps, n_sim))

            for t in range(1, n_steps+1):
                v_prev = volatilities[t-1, :]
                sqrt_v_prev = np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt)

                # Actualizar volatilidad
                new_vol = v_prev + kappa*(theta - v_prev)*dt + sigma_v*sqrt_v_prev*Z2[t-1, :]
                volatilities[t, :] = np.maximum(new_vol, 0)

                # Manejar saltos correctamente
                if incluir_saltos:
                    jump_counts = np.random.poisson(lambda_jumps * dt, size=n_sim)
                    total_jumps = jump_counts.sum()
                    if total_jumps > 0:
                        jump_sizes = np.random.normal(-0.1, 0.2, size=total_jumps)
                        index = np.repeat(np.arange(n_sim), jump_counts)
                        jumps = np.bincount(index, weights=jump_sizes, minlength=n_sim)
                    else:
                        jumps = np.zeros(n_sim)
                else:
                    jumps = 0

                prices[t, :] = prices[t-1, :] * np.exp(
                    (mu - 0.5*v_prev)*dt +
                    sqrt_v_prev*Z1[t-1, :] +
                    jumps
                )

            return prices, volatilities

        prices_sim, vol_sim = run_heston_simulation(S0, mu, v0, kappa, theta, sigma_v, rho, T, n_steps, n_sim, lambda_jumps, incluir_saltos, seed)

        # Mostrar semilla usada
        st.info(f"🔑 Semilla utilizada: `{seed}` - *Usa esta semilla para reproducir el escenario*")

        # Generar fechas para el pronóstico (excluyendo el último día histórico)
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=dias_pronostico, freq='B')

        # -----------------------------------------------------------------------------
        # Gráfica 1: Trayectorias Simuladas con colores aleatorios
        # -----------------------------------------------------------------------------
        price_mean = prices_sim.mean(axis=1)

        # Generar fechas completas (histórico + pronóstico)
        full_dates = pd.date_range(start=data.index[-1], periods=len(price_mean), freq='B')

        fig_tray = go.Figure()

        # 1. Trayectorias simuladas (solo primeras 100 para claridad)
        colors = px.colors.qualitative.Alphabet
        for i in range(min(100, n_sim)):
            fig_tray.add_trace(go.Scatter(
                x=forecast_dates,
                y=prices_sim[1:, i],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=1),
                opacity=0.2,
                hoverinfo='skip',
                showlegend=False
            ))

        # 2. Trayectorias destacadas
        fig_tray.add_trace(go.Scatter(
            x=forecast_dates,
            y=prices_sim[1:, 0],
            mode='lines',
            line=dict(color='#2ca02c', width=2),
            name="Trayectoria Inicial",
            hovertemplate="<b>Primera simulación</b><br>%{y:$,.2f}<extra></extra>"
        ))

        fig_tray.add_trace(go.Scatter(
            x=forecast_dates,
            y=prices_sim[1:, -1],
            mode='lines',
            line=dict(color='#7f7f7f', width=2, dash='dot'),
            name="Última Trayectoria",
            hovertemplate="<b>Última simulación</b><br>%{y:$,.2f}<extra></extra>"
        ))

        # 3. Media y bandas de percentiles
        lower_bound = np.percentile(prices_sim[1:], 5, axis=1)
        upper_bound = np.percentile(prices_sim[1:], 95, axis=1)

        fig_tray.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig_tray.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            name="Banda de Confianza (90%)",
            hovertemplate="<b>Rango 5-95%</b><br>%{y:$,.2f}<extra></extra>"
        ))

        fig_tray.add_trace(go.Scatter(
            x=forecast_dates,
            y=price_mean[1:],
            mode='lines',
            line=dict(color='#ff7f0e', width=3),
            name="Media Simulaciones",
            hovertemplate="<b>Promedio</b><br>%{y:$,.2f}<extra></extra>"
        ))

        # 4. Anotaciones importantes
        fig_tray.add_annotation(
            x=full_dates[-1],
            y=price_mean[-1],
            text=f"Pronóstico Final<br>${price_mean[-1]:,.2f}",
            showarrow=True,
            arrowhead=4,
            ax=-50,
            ay=-40,
            font=dict(size=12, color="#2ca02c")
        )

        fig_tray.update_layout(
            title=f"✅ Pronóstico Heston para {accion} - {n_sim} simulaciones",
            xaxis=dict(
                title="Fecha",
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis_title="Precio (MXN)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig_tray, use_container_width=True)

        # -----------------------------------------------------------------------------
        # Sección de Análisis de Riesgo
        # -----------------------------------------------------------------------------
        st.markdown("### 📊 Análisis de Riesgo Cuantitativo")

        final_prices = prices_sim[-1, :]
        VaR_95 = np.percentile(final_prices, 5)
        CVaR_95 = final_prices[final_prices <= VaR_95].mean()
        max_loss = (VaR_95 - S0)/S0
        prob_ganancia = (final_prices > S0).mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value at Risk (95%)", f"${VaR_95:,.2f}",
                     help="Pérdida máxima esperada en el peor 5% de casos")
        with col2:
            st.metric("Expected Shortfall", f"${CVaR_95:,.2f}",
                     help="Pérdida promedio en el peor 5% de escenarios")
        with col3:
            st.metric("Probabilidad de Ganancia", f"{prob_ganancia:.1%}",
                     help="Probabilidad de que el precio final supere el actual")

        # Gráfico de distribución de precios finales (CORREGIDO)
        fig_dist = ff.create_distplot(
            hist_data=[final_prices],
            group_labels=['Distribución Precios'],
            show_hist=True,  # Cambiado a True para mejor visualización
            show_rug=False,
            bin_size=0.5*(final_prices.max() - final_prices.min())/100  # Autoajuste de bins
        )

        fig_dist.add_vline(
            x=S0,
            line_dash="dash",
            line_color="green",
            annotation_text="Precio Actual",
            annotation_position="top right"
        )

        fig_dist.update_layout(
            title="📦 Distribución de Precios Finales Pronosticados",
            xaxis_title="Precio (MXN)",
            yaxis_title="Densidad",
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)



        # -----------------------------------------------------------------------------
        # Simulador financiero para el modelo Heston
        # Se utiliza el promedio de las trayectorias (price_mean) como precio pronosticado
        # -----------------------------------------------------------------------------
        st.markdown("### 💰 Simulador Financiero - Modelo Heston")
        monto = st.number_input("Monto a invertir (MXN)", min_value=0.0, value=10000.0, step=1000.0)
        precio_final = price_mean[-1]
        retorno = (precio_final / S0 - 1) * 100
        ganancia = monto * (precio_final / S0 - 1)
        monto_final = monto + ganancia
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio Actual", f"${S0:,.2f}")
        col2.metric("Precio Pronosticado", f"${precio_final:,.2f}", f"{retorno:.2f}%")
        col3.metric("Ganancia/Pérdida", f"${ganancia:,.2f}")
        col4.metric("Monto Final", f"${monto_final:,.2f}")


#-------------------------------------------------------------------------------------------------------













    # =========================================================================
    # Sección 2: IA - Temporal Fusion Transformer
    # =========================================================================
    elif metodo_pronostico == "🤖 IA - Temporal Fusion Transformer":
        st.markdown("### 🤖 IA - Temporal Fusion Transformer (TFT)")
        with st.expander("⚙️ Parámetros del Modelo", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ventana = st.slider("Ventana Temporal (días)", 30, 365, 90,
                                    help="Número de días históricos usados para cada predicción")
                epochs = st.slider("Épocas de entrenamiento", 10, 500, 50,
                                   help="Número de iteraciones para entrenar el modelo (hiperparámetro)")
            with col2:
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1,
                                          help="Cantidad de muestras procesadas antes de actualizar los pesos del modelo (hiperparámetro)")
                learning_rate = st.selectbox("Learning Rate", [1e-4, 3e-4, 1e-3], index=1,
                                             help="Tasa de aprendizaje que determina el tamaño de los pasos en la optimización (hiperparámetro)")
                st.markdown("#### Comentario:")
                st.markdown("ESTE MODELO ESTA PENDIENTE DE TERMINAR DE IMPLEMENTAR. Los hiperparámetros, como batch size y learning rate, influyen en cómo aprende el modelo. Un batch size mayor puede ayudar a estabilizar el entrenamiento, mientras que un learning rate adecuado es clave para lograr una buena convergencia.")

        # Selección de acción
        accion = st.selectbox("📈 Seleccione acción para pronóstico", options=selected_companies)

        # Cargar datos
        data = data_dict[accion]["Close"]
        returns = data.pct_change().dropna()

        # Preprocesamiento: división en datos de entrenamiento y prueba
        train_data = data.iloc[:-30]
        test_data = data.iloc[-30:]

        # Entrenamiento simulado (Placeholder: en producción se usaría PyTorch/TensorFlow)
        @st.cache_resource
        def train_tft_model(_data, window_size, epochs, batch_size):
            # Este modelo simulado ilustra el proceso de entrenamiento.
            class FakeModel:
                def predict(self, data):
                    last_value = data.iloc[-1]
                    trend = np.linspace(last_value, last_value * 1.1, 30)
                    noise = np.random.normal(0, data.std() * 0.1, 30)
                    return trend + noise
            return FakeModel()

        model = train_tft_model(train_data, ventana, epochs, batch_size)
        pronostico = model.predict(train_data.iloc[-ventana:])

        # Generar fechas para el pronóstico
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

        # Visualización del pronóstico TFT
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, name="Histórico", line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=pronostico, name="Pronóstico TFT", line=dict(color='#2ca02c', dash='dot')))
        fig.update_layout(title=f"Pronóstico TFT para {accion}", xaxis_title="Fecha", yaxis_title="Precio (MXN)")
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # Sección 3: Modelo Econométrico ARIMA-GARCH
    # =========================================================================
    else:
        st.markdown("### 📉 Modelo Econométrico ARIMA-GARCH (PENDIENTE DE TERMINAR DE IMPLEMENTAR)")

        # Selección de acción
        accion = st.selectbox("📈 Seleccione acción para pronóstico", options=selected_companies)

        # Cargar datos
        data = data_dict[accion]["Close"]
        returns = data.pct_change().dropna()

        st.markdown("#### 🧮 Análisis Estadístico Previo")
        # Prueba de Dickey-Fuller: evalúa la estacionariedad de la serie.
        adf_result = adfuller(data)
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        st.write(f"**Estadístico ADF:** {adf_stat:.4f}")
        st.write(f"**Valor p:** {p_value:.4f}")
        if p_value > 0.05:
            st.error("⚠️ La serie no es estacionaria. Se aplicará diferenciación automática.")
            data_diff = data.diff().dropna()
            st.write("Se ha aplicado la diferenciación a la serie para lograr estacionariedad.")
        else:
            st.success("✅ La serie es estacionaria. No se requiere diferenciación.")
            data_diff = data

        st.markdown("**Comentario:** La prueba ADF (Augmented Dickey-Fuller) determina si la serie tiene una raíz unitaria. Un valor p mayor a 0.05 indica que la serie no es estacionaria.")

        # Graficar ACF y PACF en dos gráficas pequeñas, una al lado de la otra
        fig_acf, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(data_diff, ax=ax1, lags=20)
        ax1.set_title("ACF")
        plot_pacf(data_diff, ax=ax2, lags=20, method='ywm')
        ax2.set_title("PACF")
        st.pyplot(fig_acf)

        # Selección automática del modelo basado en AIC
        st.markdown("#### Selección Automática del Modelo")
        best_aic = np.inf
        best_order = None
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model_temp = ARIMA(data, order=(p, d, q))
                        results = model_temp.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        continue
        st.write(f"Se ha seleccionado automáticamente el modelo ARIMA{best_order} con el menor AIC: {best_aic:.2f}.")
        st.markdown("**Nota:** El modelo óptimo se ha elegido automáticamente en función del AIC. Si lo deseas, puedes modificar manualmente los parámetros.")

        # Parámetros del modelo: opción manual
        with st.expander("⚙️ Configuración del Modelo (opcional)", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("Orden AR (p)", min_value=0, max_value=5, value=best_order[0])
            with col2:
                d = st.number_input("Orden de Diferenciación (d)", min_value=0, max_value=2, value=best_order[1])
            with col3:
                q = st.number_input("Orden MA (q)", min_value=0, max_value=5, value=best_order[2])

        # Entrenamiento del modelo econométrico
        @st.cache_resource
        def train_econometric_model(_data, order):
            model = ARIMA(_data, order=order)
            return model.fit()

        model = train_econometric_model(data, (p, d, q))

        # Pronóstico
        forecast_steps = st.slider("Días a pronosticar", 1, 365, 30)
        forecast = model.forecast(steps=forecast_steps)

        # Visualización del pronóstico
        fig_forecast_econ = go.Figure()
        fig_forecast_econ.add_trace(go.Scatter(x=data.index, y=data, name="Histórico", line=dict(color='#1f77b4')))
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
        fig_forecast_econ.add_trace(go.Scatter(x=forecast_index, y=forecast, name="Pronóstico", line=dict(color='#9467bd', dash='dot')))
        fig_forecast_econ.update_layout(title=f"Pronóstico ARIMA-GARCH para {accion}", xaxis_title="Fecha", yaxis_title="Precio (MXN)")
        st.plotly_chart(fig_forecast_econ, use_container_width=True)

        st.markdown("#### 📊 Resultados del Modelo")
        st.write(model.summary())




# ================================
# Pestaña 9: Distribución Retornos
# ================================
with tab9:
    st.markdown("""
    ## 🔍 **Análisis Detallado del Código** 🔍

    ### 🧮 1. Correctitud Matemática
    **📊 Portafolio NPEB (Pestaña 6):**
    - ✅ **Acierto:** Implementación correcta de μₙ y Vₙ con bootstrap
    - ✅ **Acierto:** Restricciones `sum(w)=1` y `w≥0` bien aplicadas
    - ⚠️ **Mejora:** Incorporar tasa libre de riesgo en cálculo de Sharpe ratio

    **🎲 Modelo Heston (Pestaña 8):**
    - ✅ **Acierto:** Ecuaciones diferenciales estocásticas bien implementadas
    - ⚠️ **Advertencia:** `np.maximum` para volatilidad podría causar inestabilidad numérica

    **📉 ARIMA-GARCH:**
    - ❌ **Error:** Falta componente GARCH completo
    - ✅ **Acierto:** Prueba ADF y diferenciación aplicadas correctamente

    ---

    ### 👨💻 2. Buenas Prácticas de Programación
    - ✅ **Excelente:** Uso eficiente de `@st.cache_data` para caching
    - ✅ **Modular:** Funciones bien estructuradas y reutilizables
    - 🔄 **Oportunidad:** Eliminar duplicados en carga de datos
    - ✅ **Robusto:** Manejo de errores con `try/except` en secciones críticas


    """)

# ================================
# Pie de página
# ================================
st.markdown("---")
st.markdown("""
🎓 **SEMINARIO DE INVERSIÓN Y MERCADOS FINANCIEROS - IPN**
🧑💻 Realizado por **J. Cruz Gómez** • 📧 josluigomez@gmail.com
🔮 *"Los datos son como el acero: en bruto no valen, procesados son invencibles"*
""")


#GPT
