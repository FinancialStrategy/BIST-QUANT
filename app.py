# ============================================================================
# 1. CORE IMPORTS & CONFIGURATION (STREAMLIT ADAPTED)
# ============================================================================
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# PyPortfolioOpt libraries
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import CLA
from pypfopt import EfficientCVaR
from pypfopt import HRPOpt

# ARCH: For Econometric Volatility Forecasting (GARCH)
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    
# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit Page Configuration
st.set_page_config(
    page_title="BIST Portfolio Risk Analytics",
    layout="wide",
    page_icon="📈"
)

# Turkish BIST 30 tickers
BIST30_TICKERS = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS',
    'EKGYO.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS',
    'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS',
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS',
    'SKBNK.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS',
    'TTKOM.IS', 'TUPRS.IS', 'ULKER.IS', 'VAKBN.IS', 'YKBNK.IS'
]

# Annualized risk-free rate for Turkey
RISK_FREE_RATE = 0.25

# ============================================================================
# 2. TURKISH PORTFOLIO OPTIMIZER CLASS (Modified for caching)
# ============================================================================

class TurkishPortfolioOptimizer:
    def __init__(self):
        self.tickers = BIST30_TICKERS
        self.benchmark_tickers = ['XU100.IS', 'XU030.IS']
        self.risk_free_rate = RISK_FREE_RATE
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None
        
    @st.cache_data(ttl=3600)
    def fetch_data(_self, start_date='2023-01-01', end_date=None):
        """Fetch data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # BIST 30 stocks
        data = yf.download(_self.tickers, start=start_date, end=end_date)['Adj Close']
        
        # Benchmark indices
        benchmark_data = yf.download(_self.benchmark_tickers, start=start_date, end=end_date)['Adj Close']
        
        # Fill NaN values with forward fill
        data = data.ffill()
        benchmark_data = benchmark_data.ffill()
        
        # Select stocks with at least 20 days of data
        valid_tickers = data.columns[data.notna().sum() > 20]
        data = data[valid_tickers]
        
        # Daily returns
        returns = data.pct_change().dropna()
        benchmark_returns = benchmark_data.pct_change().dropna()
        
        return data, returns, benchmark_data, benchmark_returns
    
    # NOTE: This method is now simplified to calculate metrics based on inputs 
    def calculate_portfolio_metrics(self, weights_series: pd.Series, returns: pd.DataFrame, benchmark_returns: pd.DataFrame, risk_free_rate: float):
        """Calculate portfolio performance metrics"""
        
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Ensure weights are aligned and normalized to assets present in returns
        aligned_weights = weights_series.reindex(returns.columns).fillna(0)
        
        # Portfolio return
        portfolio_returns = (returns * aligned_weights).sum(axis=1)
        
        # Mean return (daily)
        mean_return = portfolio_returns.mean()
        
        # Volatility (daily)
        volatility = portfolio_returns.std()
        
        # Annual return and volatility
        annual_return = (1 + mean_return) ** 252 - 1
        annual_volatility = volatility * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum Drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% daily)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Conditional VaR (Expected Shortfall - 95% daily)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Sortino Ratio (using downside volatility only)
        downside_returns = portfolio_returns[portfolio_returns < daily_rf]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Information Ratio (relative to benchmark - using XU100.IS)
        tracking_error = 0
        information_ratio = 0
        
        if 'XU100.IS' in benchmark_returns.columns:
            bench_returns_ts = benchmark_returns['XU100.IS'].reindex(portfolio_returns.index).fillna(0)
            if not bench_returns_ts.empty:
                benchmark_return_mean = bench_returns_ts.mean()
                benchmark_annual = (1 + benchmark_return_mean) ** 252 - 1
                active_return = annual_return - benchmark_annual
                
                tracking_error = (portfolio_returns - bench_returns_ts).std() * np.sqrt(252)
                information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        # Higher Moments
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis() + 3
        
        metrics = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Information Ratio': information_ratio,
            'Tracking Error': tracking_error,
            'Skewness': skewness, 
            'Kurtosis': kurtosis,
        }
        
        return metrics, portfolio_returns
    
    def optimize_portfolio(self, method, mu, S, returns, risk_free_rate, target_return=None, risk_aversion=1):
        """Portfolio optimization with different methods"""
        
        daily_rf = risk_free_rate/252
        weights = {}
        
        try:
            if method == 'max_sharpe':
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=daily_rf)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=daily_rf)
                
            elif method == 'min_volatility':
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=daily_rf)
                
            elif method == 'efficient_risk':
                ef = EfficientFrontier(mu, S)
                target_volatility = target_return if target_return is not None and target_return > 0 else mu.std().mean() * np.sqrt(252) * 1.1
                ef.efficient_risk(target_volatility=target_volatility/np.sqrt(252))
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=daily_rf)
                
            elif method == 'efficient_return':
                ef = EfficientFrontier(mu, S)
                target_ret = target_return if target_return is not None and target_return > 0 else mu.mean().mean() * np.sqrt(252) * 1.1
                daily_target_return = (1 + target_ret) ** (1/252) - 1
                ef.efficient_return(target_return=daily_target_return)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=daily_rf)
                
            elif method == 'max_quadratic_utility':
                ef = EfficientFrontier(mu, S)
                ef.max_quadratic_utility(risk_aversion=risk_aversion)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=daily_rf)
                
            elif method == 'hrp':
                hrp = HRPOpt(returns)
                weights = hrp.optimize()
                weights = hrp.clean_weights()
                
                # Performance calculation is manual for HRP
                weights_series = pd.Series(weights).reindex(returns.columns).fillna(0)
                metrics, _ = self.calculate_portfolio_metrics(weights_series, returns, None, risk_free_rate) # Simplified call
                performance = (metrics['Annual Return'], metrics['Annual Volatility'], metrics['Sharpe Ratio'])
                
            elif method == 'cvar':
                cvar = EfficientCVaR(mu, returns)
                cvar.min_cvar()
                weights = cvar.clean_weights()
                
                # Performance calculation is manual for CVaR
                weights_series = pd.Series(weights).reindex(returns.columns).fillna(0)
                metrics, _ = self.calculate_portfolio_metrics(weights_series, returns, None, risk_free_rate) # Simplified call
                performance = (metrics['Annual Return'], metrics['Annual Volatility'], metrics['Sharpe Ratio'])
                
            elif method == 'equal_weight':
                n_assets = len(returns.columns)
                weights = {ticker: 1/n_assets for ticker in returns.columns}
                
                # Performance calculation is manual for Equal Weight
                weights_series = pd.Series(weights).reindex(returns.columns).fillna(0)
                metrics, _ = self.calculate_portfolio_metrics(weights_series, returns, None, risk_free_rate) # Simplified call
                performance = (metrics['Annual Return'], metrics['Annual Volatility'], metrics['Sharpe Ratio'])
                
            else:
                raise ValueError(f"Unknown optimization method: {method}")
        
        except Exception as e:
            st.error(f"Optimization failed for {method}: {str(e)}. Using Equal Weight as fallback.")
            n_assets = len(returns.columns)
            weights = {ticker: 1/n_assets for ticker in returns.columns}
            weights_series = pd.Series(weights).reindex(returns.columns).fillna(0)
            metrics, _ = self.calculate_portfolio_metrics(weights_series, returns, None, risk_free_rate)
            performance = (metrics['Annual Return'], metrics['Annual Volatility'], metrics['Sharpe Ratio'])

        # Convert weights to DataFrame
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_df.index.name = 'Ticker'
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        
        return weights_df, performance, None # Metrics are recalculated fully later

    # Keep Plotly functions as they use native Plotly which works in Streamlit
    def plot_efficient_frontier(_self, mu, S, returns, method='max_sharpe'):
        # ... (Implementation kept the same, ensures Plotly output for Streamlit) ...
        cla = CLA(mu, S)
        ef_points = cla.efficient_frontier(points=100)
        
        optimizers = {
            'Max Sharpe': 'max_sharpe',
            'Min Volatility': 'min_volatility',
            'Equal Weight': 'equal_weight'
        }
        
        fig = go.Figure()
        
        # Efficient Frontier
        fig.add_trace(go.Scatter(
            x=[point[1] * np.sqrt(252) for point in ef_points], # Annualized volatility
            y=[point[0] * 252 for point in ef_points],          # Annualized return
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#60a5fa', width=2),
            fill='tozeroy',
            fillcolor='rgba(96, 165, 250, 0.1)'
        ))
        
        # Optimization points
        colors = ['#4ade80', '#f87171', '#fbbf24', '#a78bfa']
        for i, (label, method_name) in enumerate(optimizers.items()):
            try:
                # Use cached mu/S for quick plotting of fixed points
                weights_df, performance, _ = _self.optimize_portfolio(method_name, mu, S, returns, _self.risk_free_rate)
                ret, vol, sharpe = performance
                
                fig.add_trace(go.Scatter(
                    x=[vol],
                    y=[ret],
                    mode='markers+text',
                    name=label,
                    marker=dict(size=15, color=colors[i % len(colors)]),
                    text=[label],
                    textposition="top center",
                    hovertemplate=f"<b>{label}</b><br>Return: {ret:.2%}<br>Volatility: {vol:.2%}<br>Sharpe: {sharpe:.3f}"
                ))
            except:
                continue
        
        # Individual stocks
        individual_returns = mu * 252 # Daily mean return to annual
        individual_vols = np.sqrt(np.diag(S) * 252) # Daily std dev to annual vol
        
        fig.add_trace(go.Scatter(
            x=individual_vols,
            y=individual_returns,
            mode='markers',
            name='Individual Stocks',
            marker=dict(size=8, color='#94a3b8', opacity=0.6),
            text=returns.columns,
            hovertemplate="<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}"
        ))
        
        fig.update_layout(
            title='Efficient Frontier - BIST 30 Portfolio Optimization',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            hovermode='closest',
            template='plotly_dark',
            height=600,
            showlegend=True,
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b'
        )
        
        return fig

    def plot_portfolio_comparison(_self, mu, S, returns, benchmark_returns, strategies=None):
        # ... (Implementation kept the same, adapted to take mu, S as inputs) ...
        if strategies is None:
            strategies = ['max_sharpe', 'min_volatility', 'equal_weight', 'hrp']
        
        results = []
        portfolio_returns_dict = {}
        
        for strategy in strategies:
            try:
                weights_df, performance, _ = _self.optimize_portfolio(strategy, mu, S, returns, _self.risk_free_rate)
                ret, vol, sharpe = performance
                
                # Calculate full metrics
                weights_series = pd.Series(weights_df['Weight']).reindex(returns.columns).fillna(0)
                metrics, portfolio_returns = _self.calculate_portfolio_metrics(weights_series, returns, benchmark_returns, _self.risk_free_rate)

                portfolio_returns_dict[strategy] = portfolio_returns
                
                results.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Annual Return': ret,
                    'Annual Volatility': vol,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown': metrics['Max Drawdown'],
                    'VaR (95%)': metrics['VaR (95%)'],
                    'Sortino Ratio': metrics['Sortino Ratio']
                })
                
            except Exception as e:
                # st.warning(f"Comparison error for {strategy}: {e}")
                continue
        
        # Performance metrics table
        metrics_df = pd.DataFrame(results)
        
        # Performance plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Annual Returns Comparison',
                          'Risk-Return Tradeoff', 'Drawdown Analysis'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Cumulative Returns
        for strategy, returns in portfolio_returns_dict.items():
            cumulative = (1 + returns).cumprod()
            fig.add_trace(
                go.Scatter(x=cumulative.index, y=cumulative.values,
                          name=strategy.replace('_', ' ').title(),
                          mode='lines'),
                row=1, col=1
            )
        
        # 2. Annual Returns (Bar chart)
        fig.add_trace(
            go.Bar(x=metrics_df['Strategy'], y=metrics_df['Annual Return'],
                  name='Annual Return',
                  marker_color='#60a5fa'),
            row=1, col=2
        )
        
        # 3. Risk-Return Tradeoff
        fig.add_trace(
            go.Scatter(x=metrics_df['Annual Volatility'], y=metrics_df['Annual Return'],
                      mode='markers+text',
                      text=metrics_df['Strategy'],
                      marker=dict(size=metrics_df['Sharpe Ratio'].abs()*10, 
                                 color=metrics_df['Sharpe Ratio'],
                                 colorscale='Viridis',
                                 showscale=True,
                                 colorbar=dict(title="Sharpe Ratio")),
                      name='Risk-Return'),
            row=2, col=1
        )
        
        # 4. Drawdown analysis
        for strategy, returns in portfolio_returns_dict.items():
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          name=strategy.replace('_', ' ').title() + ' Drawdown',
                          mode='lines',
                          fill='tozeroy'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, template='plotly_dark',
                         title_text="Portfolio Strategy Comparison",
                         plot_bgcolor='#1e293b',
                         paper_bgcolor='#1e293b')
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Annual Return", row=1, col=2)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Annual Return", row=2, col=1)
        fig.update_xaxes(title_text="Annual Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=2)
        
        return fig, metrics_df

# ============================================================================
# 3. GARCH AND RISK EXTENSION FUNCTIONS (Kept the same)
# ============================================================================

def calculate_garch_metrics(returns_series):
    """Fit GARCH(1,1) model and calculate parameters."""
    if not HAS_ARCH:
        return None, None, None
    
    returns_scaled = returns_series * 100
    
    try:
        am = arch.arch_model(returns_scaled.dropna(), vol='Garch', p=1, q=1, rescale=False)
        res = am.fit(disp='off')
        
        conditional_volatility = res.conditional_volatility / 100
        
        garch_params = {
            'Constant (ω)': res.params.get('omega', 0),
            'ARCH Term (α)': res.params.get('alpha[1]', 0),
            'GARCH Term (β)': res.params.get('beta[1]', 0),
            'Log Likelihood': res.loglikelihood,
            'AIC': res.aic,
            'Persistence': res.params.get('alpha[1]', 0) + res.params.get('beta[1]', 0),
            'Next Day Forecast Volatility (Annualized)': res.forecast(horizon=1).variance.iloc[-1].values[0]**0.5 * np.sqrt(252) / 100
        }
        
        garch_stats = {
            'Constant (ω)': {'Estimate': garch_params['Constant (ω)'], 'Std. Error': res.std_err.get('omega', 0), 't-Statistic': res.tvalues.get('omega', 0), 'p-Value': res.pvalues.get('omega', 1)},
            'ARCH Term (α)': {'Estimate': garch_params['ARCH Term (α)'], 'Std. Error': res.std_err.get('alpha[1]', 0), 't-Statistic': res.tvalues.get('alpha[1]', 0), 'p-Value': res.pvalues.get('alpha[1]', 1)},
            'GARCH Term (β)': {'Estimate': garch_params['GARCH Term (β)'], 'Std. Error': res.std_err.get('beta[1]', 0), 't-Statistic': res.tvalues.get('beta[1]', 0), 'p-Value': res.pvalues.get('beta[1]', 1)},
        }
        
        return garch_params, conditional_volatility, garch_stats
    except Exception as e:
        return None, None, None

def calculate_var_cvar_levels(returns_series):
    """Calculate VaR and CVaR at different confidence levels."""
    levels = [0.95, 0.975, 0.99, 0.995]
    var_results = []
    cvar_results = []
    
    for level in levels:
        confidence = (1 - level) * 100
        var_return = np.percentile(returns_series, confidence)
        var_results.append(var_return * 100)
        
        cvar_return = returns_series[returns_series <= var_return].mean()
        cvar_results.append(cvar_return * 100)
        
    return var_results, cvar_results

# ============================================================================
# 4. STREAMLIT APPLICATION ENTRY POINT
# ============================================================================

def main_streamlit_app():
    optimizer = TurkishPortfolioOptimizer()

    st.title("🇹🇷 BIST Portfolio Risk & Optimization Terminal")
    st.markdown("---")

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Configuration")
    
    # Date Inputs
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*2))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    # Optimization Parameters
    st.sidebar.subheader("Optimization Parameters")
    
    risk_free_rate = st.sidebar.number_input(
        "Annual Risk-Free Rate (%)", 
        value=RISK_FREE_RATE * 100, 
        min_value=0.0, 
        max_value=50.0, 
        step=1.0
    ) / 100
    
    strategy = st.sidebar.selectbox(
        "Optimization Strategy",
        ['max_sharpe', 'min_volatility', 'efficient_risk', 
         'efficient_return', 'max_quadratic_utility', 
         'hrp', 'cvar', 'equal_weight'],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    target_return, risk_aversion = None, 1.0

    if strategy in ['efficient_risk', 'efficient_return']:
        target_return = st.sidebar.slider(
            "Target Annualized R/V (%)", 
            min_value=5.0, 
            max_value=80.0, 
            value=30.0,
            step=5.0
        ) / 100
        
    if strategy == 'max_quadratic_utility':
        risk_aversion = st.sidebar.slider(
            "Risk Aversion (Delta)", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0, 
            step=0.1
        )
    
    # --- DATA FETCHING & CORE CALCULATION ---
    
    with st.spinner("Fetching and processing BIST data..."):
        try:
            data, returns, benchmark_data, benchmark_returns = optimizer.fetch_data(start_date=start_date, end_date=end_date)
            
            if data.empty or returns.empty:
                st.error("Error: Could not retrieve data for the selected period. Check tickers or date range.")
                return

            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            weights_df, performance, _ = optimizer.optimize_portfolio(
                strategy, mu, S, returns, risk_free_rate, target_return, risk_aversion
            )
            
            # Recalculate full metrics using the final weights
            weights_series = pd.Series(weights_df['Weight']).reindex(returns.columns).fillna(0)
            metrics, portfolio_returns = optimizer.calculate_portfolio_metrics(weights_series, returns, benchmark_returns, risk_free_rate)

        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
            return

    st.success(f"Analysis completed successfully. Optimization method: **{strategy.replace('_', ' ').title()}**")
    st.markdown("---")

    # --- METRICS DASHBOARD (Streamlit Native) ---
    st.header("1. Core Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Annual Return", f"{metrics['Annual Return']:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{metrics['Annual Volatility']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}", delta=f"RFR: {risk_free_rate:.1%}")
    with col4:
        st.metric("Information Ratio (vs XU100)", f"{metrics['Information Ratio']:.3f}")
    with col5:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")

    # --- PORTFOLIO WEIGHTS ---
    st.header("2. Portfolio Weights")
    st.dataframe(
        weights_df.style.format({'Weight': '{:.2%}'}).background_gradient('Blues'),
        use_container_width=True, 
        height=min(400, len(weights_df) * 35 + 38)
    )

    # --- EFFICIENCY AND COMPARISON ---
    st.header("3. Optimization and Comparison")
    
    # Efficient Frontier
    ef_fig = optimizer.plot_efficient_frontier(mu, S, returns, strategy)
    st.plotly_chart(ef_fig, use_container_width=True)
    
    # Strategy Comparison
    comp_fig, comp_metrics_df = optimizer.plot_portfolio_comparison(mu, S, returns, benchmark_returns, ['max_sharpe', 'min_volatility', 'equal_weight', 'hrp'])
    st.plotly_chart(comp_fig, use_container_width=True)

    # --- ADVANCED RISK ---
    st.header("4. Advanced Risk Analysis")
    
    garch_params, conditional_volatility, garch_stats = calculate_garch_metrics(portfolio_returns)
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)

    # Risk Metrics Table (Risk Col 1)
    with risk_col1:
        st.subheader("VaR, CVaR & Moments")
        
        bench_returns = benchmark_returns['XU100.IS'].dropna() if 'XU100.IS' in benchmark_returns.columns else benchmark_returns.iloc[:, 0].dropna()
        bench_var95 = np.percentile(bench_returns, 5)
        bench_cvar95 = bench_returns[bench_returns <= bench_var95].mean()
        
        risk_df = pd.DataFrame({
            'Metric': ['VaR (95% Daily)', 'CVaR (95% Daily)', 'Skewness', 'Kurtosis'],
            'Portfolio': [metrics['VaR (95%)'], metrics['CVaR (95%)'], metrics['Skewness'], metrics['Kurtosis']],
            'Benchmark (XU100)': [bench_var95, bench_cvar95, bench_returns.skew(), bench_returns.kurtosis() + 3]
        }).set_index('Metric')
        
        st.dataframe(
            risk_df.style.format('{:.2%}', subset=pd.IndexSlice[['VaR (95% Daily)', 'CVaR (95% Daily)'], :])
            .format('{:.2f}', subset=pd.IndexSlice[['Skewness', 'Kurtosis'], :]),
            use_container_width=True
        )

    # GARCH Volatility Plot (Risk Col 2)
    with risk_col2:
        st.subheader("GARCH Conditional Volatility")
        if conditional_volatility is not None and not conditional_volatility.empty:
            garch_annualized = conditional_volatility * np.sqrt(252) * 100
            
            fig_garch = go.Figure()
            fig_garch.add_trace(go.Scatter(x=garch_annualized.index, y=garch_annualized.values, mode='lines', name='GARCH Volatility (Annualized)', line=dict(color='#f87171')))
            fig_garch.add_trace(go.Scatter(x=portfolio_returns.rolling(20).std().index, y=portfolio_returns.rolling(20).std() * np.sqrt(252) * 100, mode='lines', name='20-day Rolling Vol', line=dict(color='#60a5fa', dash='dot')))
            
            fig_garch.update_layout(
                template='plotly_dark',
                title_text="Conditional Volatility Trend (%)",
                yaxis_title="Annualized Volatility (%)",
                height=350,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_garch, use_container_width=True)
        else:
            st.info("ARCH library not found or GARCH model failed to fit.")

    # GARCH Model Diagnostics (Risk Col 3)
    with risk_col3:
        st.subheader("GARCH Model Parameters")
        if garch_stats:
            garch_df = pd.DataFrame(garch_stats).T[['Estimate', 't-Statistic', 'p-Value']]
            st.dataframe(
                garch_df.style.format({
                    'Estimate': '{:.6f}',
                    't-Statistic': '{:.2f}',
                    'p-Value': '{:.4f}'
                }),
                use_container_width=True
            )
            
            st.metric("Persistence ($\alpha+\beta$)", f"{garch_params['Persistence']:.4f}")
        else:
            st.info("GARCH parameters not available.")

# --- Execution ---
if __name__ == "__main__":
    main_streamlit_app()
