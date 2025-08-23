import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import os

# LangChain imports
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configuration with fallback to mitigate rate limits
PRIMARY_MODEL = "openai/gpt-oss-20b"
FALLBACK_MODEL = "compound-beta"

# Safe formatting helpers to avoid format errors on non-numeric values
def fmt_f(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"

def fmt_commas(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"

def get_llm(temperature: float = 0.1, prefer_fallback: bool = False) -> ChatGroq:
    """Return a ChatGroq instance, trying primary first then fallback (or vice versa if prefer_fallback)."""
    model_order = (FALLBACK_MODEL, PRIMARY_MODEL) if prefer_fallback else (PRIMARY_MODEL, FALLBACK_MODEL)
    last_error = None
    for model_name in model_order:
        try:
            return ChatGroq(model=model_name, temperature=temperature, api_key=GROQ_API_KEY)
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"AI service unavailable. Last error: {last_error}")

def invoke_with_fallback(prompt: str, temperature: float = 0.1) -> str:
    """Invoke completion with primary then fallback model on failure."""
    last_error = None
    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            llm = ChatGroq(model=model_name, temperature=temperature, api_key=GROQ_API_KEY)
            return llm.invoke(prompt).content
        except Exception as e:
            last_error = e
            continue
    return f"Please try again later. AI service error: {last_error}"

# ---------------- STOCK APP ----------------
class StockAnalysisApp:
    def __init__(self):
        """Initialize the Streamlit Stock Analysis Application"""
        st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
        st.title("⚡Comprehensive Stock Analysis Dashboard")

        # Sidebar for stock selection
        self.setup_sidebar()

    def setup_sidebar(self):
        """Setup sidebar for stock and date selection with user input"""
        st.sidebar.header("Stock Selection")
        
        predefined_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
        if 'ticker' not in st.session_state:
            st.session_state.ticker = "RELIANCE.NS"
        if 'start_date' not in st.session_state:
            st.session_state.start_date = pd.to_datetime('2020-01-01')
        if 'end_date' not in st.session_state:
            st.session_state.end_date = pd.to_datetime('today')

        ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", value=st.session_state.ticker)
        st.sidebar.markdown(f"**Try these:** {', '.join(predefined_tickers)}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date_input = st.date_input("Start Date", value=st.session_state.start_date)
        with col2:
            end_date_input = st.date_input("End Date", value=st.session_state.end_date)
        
        if st.sidebar.button("Analyze Stock"):
            st.session_state.ticker = ticker_input
            st.session_state.start_date = pd.to_datetime(start_date_input)
            st.session_state.end_date = pd.to_datetime(end_date_input)
            try:
                stock_data = yf.download(
                    st.session_state.ticker, start=st.session_state.start_date, end=st.session_state.end_date
                )
                
                if stock_data.empty:
                    st.error("No data available for the selected stock and date range. Please check the ticker symbol and date range.")
                    return
                
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)
                
                stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
                stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
                delta = stock_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                stock_data['RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                stock_data['BB_Mid'] = stock_data['SMA_20']
                stock_data['BB_Std'] = stock_data['Close'].rolling(window=20).std()
                stock_data['BB_Upper'] = stock_data['BB_Mid'] + 2 * stock_data['BB_Std']
                stock_data['BB_Lower'] = stock_data['BB_Mid'] - 2 * stock_data['BB_Std']
                
                # Donchian Channel
                stock_data['Don_Upper'] = stock_data['High'].rolling(window=20).max()
                stock_data['Don_Lower'] = stock_data['Low'].rolling(window=20).min()
                stock_data['Don_Mid'] = (stock_data['Don_Upper'] + stock_data['Don_Lower']) / 2
                
                # 20-day high/low
                stock_data['20D_High'] = stock_data['High'].rolling(window=20).max()
                stock_data['20D_Low'] = stock_data['Low'].rolling(window=20).min()
                
                # For Darvas Box (simplified)
                stock_data['Darvas_High'] = stock_data['High'].rolling(window=5).max()
                stock_data['Darvas_Low'] = stock_data['Low'].rolling(window=5).min()
                
                # For Turtle Trader (based on Donchian)
                stock_data['Turtle_Buy'] = stock_data['Close'] > stock_data['20D_High'].shift(1)
                stock_data['Turtle_Sell'] = stock_data['Close'] < stock_data['20D_Low'].shift(1)
                
                # Market Index for comparison
                market_data = yf.download('^NSEI', start=st.session_state.start_date, end=st.session_state.end_date)
                stock_data['Market_Close'] = market_data['Close']

                # Additional indicators for trading signals
                stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()  # For long-term
                stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()  # For short-term/swing
                stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()  # For short-term/swing
                stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']  # MACD for swing/intraday
                stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()  # Signal line for MACD

                st.session_state.stock_data = stock_data
                st.session_state.stock_info = yf.Ticker(st.session_state.ticker).info

            except Exception as e:
                st.error(f"Error fetching stock data: {e}. Please check the ticker symbol and your internet connection.")
                return

        if 'stock_data' in st.session_state:
            self.stock_data = st.session_state.stock_data
            self.stock_info = st.session_state.stock_info
            self.create_analysis_tabs()

    def create_analysis_tabs(self):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Exploratory Data Analysis",
            "Fundamental Analysis",
            "Statistical Analysis", 
            "Risk & Probability", 
            "ML Price Prediction",
            "Stock Advisor Chat",
            "Trading Strategies",
            "Trading Signals"
        ])
        
        with tab1: self.exploratory_data_analysis()
        with tab2: self.fundamental_analysis()
        with tab3: self.statistical_analysis()
        with tab4: self.risk_analysis()
        with tab5: self.ml_price_prediction()
        with tab6: self.chat_advisor()
        with tab7: self.trading_strategies()
        with tab8: self.trading_signals()

    def fundamental_analysis(self):
        st.subheader("Fundamental Analysis")
        
        # Display key fundamental metrics
        st.write("### Key Financial Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Cap", fmt_commas(self.stock_info.get('marketCap')))
            st.metric("Trailing P/E", fmt_f(self.stock_info.get('trailingPE')))
            st.metric("Forward P/E", fmt_f(self.stock_info.get('forwardPE')))
        with col2:
            dy = self.stock_info.get('dividendYield')
            st.metric("Dividend Yield", f"{fmt_f((dy or 0) * 100)}%")
            st.metric("Price to Book", fmt_f(self.stock_info.get('priceToBook')))
            st.metric("Enterprise Value", fmt_commas(self.stock_info.get('enterpriseValue')))
        with col3:
            st.metric("Beta", fmt_f(self.stock_info.get('beta')))
            st.metric("52 Week High", fmt_f(self.stock_info.get('fiftyTwoWeekHigh')))
            st.metric("52 Week Low", fmt_f(self.stock_info.get('fiftyTwoWeekLow')))

        # Display company information
        st.write("### Company Information")
        st.write(f"**{self.stock_info.get('longName', 'N/A')}**")
        st.write(f"**Sector:** {self.stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {self.stock_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {self.stock_info.get('website', 'N/A')}")
        
        st.write("### Business Summary")
        st.write(self.stock_info.get('longBusinessSummary', 'N/A'))

        # AI-Powered Fundamental Insights
        st.write("### AI-Powered Fundamental Insights")
        self.fundamental_insights = self.generate_ai_fundamental_insights()
        st.markdown(self.fundamental_insights)

    def generate_ai_fundamental_insights(self):
        if not GROQ_API_KEY:
            return "Please set your GROQ_API_KEY to enable AI-powered insights."
        
        context = f'''
        Act as a senior financial analyst for {self.stock_info.get('longName', 'N/A')}.
        
        Analyze the following fundamental data:
        - Market Cap: {self.stock_info.get('marketCap', 'N/A')}
        - Trailing P/E: {self.stock_info.get('trailingPE', 'N/A')}
        - Forward P/E: {self.stock_info.get('forwardPE', 'N/A')}
        - Dividend Yield: {self.stock_info.get('dividendYield', 'N/A')}
        - Price to Book: {self.stock_info.get('priceToBook', 'N/A')}
        - Beta: {self.stock_info.get('beta', 'N/A')}
        - Enterprise Value: {self.stock_info.get('enterpriseValue', 'N/A')}
        - Business Summary: {self.stock_info.get('longBusinessSummary', 'N/A')}

        Provide a detailed fundamental analysis in bullet points:
        - Evaluate the company's valuation based on P/E ratios and Price to Book.
        - Assess the company's financial health and stability using metrics like dividend yield and enterprise value.
        - Analyze the stock's risk profile based on its Beta.
        - Provide a summary of the company's business model and competitive advantages from the summary.
        - Offer a data-driven investment thesis (buy/hold/sell) based on the fundamental analysis.
        - Suggest what kind of investor this stock would be suitable for (e.g., value, growth, income).
        
        Use a professional, analyst-like tone.
        '''
        return invoke_with_fallback(context, temperature=0.1)

    def exploratory_data_analysis(self):
        st.subheader("Exploratory Data Analysis & Technical Indicators")
        col1, col2 = st.columns(2)

        with col1:
            current_price = self.stock_data['Close'].iloc[-1]
            previous_price = self.stock_data['Close'].iloc[-2] if len(self.stock_data) > 1 else "N/A"
            change = ((current_price - previous_price) / previous_price * 100) if isinstance(previous_price, (int, float)) else 0
            st.metric("Current Price", f"{current_price:.2f}")
            st.metric("Previous Close", f"{previous_price:.2f}" if isinstance(previous_price, (int, float)) else previous_price)
            st.metric("Daily Change", f"{change:.2f}%")
            
            st.write("### Summary Statistics")
            st.dataframe(self.stock_data.describe().round(2))

        with col2:
            st.write("### Price Trend with Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Closing Price'))
            fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='orange', width=2)))
            fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='purple', width=2)))
            st.plotly_chart(fig)
            
        st.write("### Candlestick Chart")
        fig_candle = go.Figure(data=[go.Candlestick(x=self.stock_data.index,
                                                    open=self.stock_data['Open'],
                                                    high=self.stock_data['High'],
                                                    low=self.stock_data['Low'],
                                                    close=self.stock_data['Close'])])
        st.plotly_chart(fig_candle)

        st.write("### Volume Chart")
        fig_vol = px.bar(self.stock_data, x=self.stock_data.index, y='Volume')
        st.plotly_chart(fig_vol)
        
        st.write("### Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        st.plotly_chart(fig_rsi)
        
        st.write("### Bollinger Bands")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Closing Price'))
        fig_bb.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='red')))
        fig_bb.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='green')))
        fig_bb.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['BB_Mid'], mode='lines', name='Middle Band', line=dict(color='blue', dash='dash')))
        st.plotly_chart(fig_bb)

    def statistical_analysis(self):
        st.subheader("Statistical Analysis")
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Descriptive Statistics of Daily Returns")
            st.write(f"**Mean Return:** {daily_returns.mean() * 100:.4f}%")
            st.write(f"**Median Return:** {daily_returns.median() * 100:.4f}%")
            st.write(f"**Standard Deviation:** {daily_returns.std() * 100:.4f}%")
            st.write(f"**Skewness:** {daily_returns.skew():.4f}")
            st.write(f"**Kurtosis:** {daily_returns.kurtosis():.4f}")
            
            # Additional features
            st.write(f"**Autocorrelation (lag 1):** {daily_returns.autocorr(lag=1):.4f}")
            shapiro_stat, shapiro_p = stats.shapiro(daily_returns)
            st.write(f"**Shapiro-Wilk Normality Test:** Stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
            jb_stat, jb_p = stats.jarque_bera(daily_returns)
            st.write(f"**Jarque-Bera Normality Test:** Stat={jb_stat:.4f}, p-value={jb_p:.4f}")
            
        with col2:
            st.write("### Daily Returns Distribution")
            fig = px.histogram(x=daily_returns, title='Daily Returns Distribution', labels={'x': 'Daily Returns'}, nbins=50, marginal='box')
            st.plotly_chart(fig)
            
            st.write("### Rolling Mean and Std (30 days)")
            rolling_mean = daily_returns.rolling(window=30).mean()
            rolling_std = daily_returns.rolling(window=30).std()
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(x=daily_returns.index, y=rolling_mean, mode='lines', name='Rolling Mean'))
            fig_rolling.add_trace(go.Scatter(x=daily_returns.index, y=rolling_std, mode='lines', name='Rolling Std'))
            st.plotly_chart(fig_rolling)
            
            # Comparison with market
            market_returns = self.stock_data['Market_Close'].pct_change().dropna()
            correlation = daily_returns.corr(market_returns)
            st.write(f"**Correlation with Market (Nifty 50):** {correlation:.4f}")
            
        st.write("### AI-Powered Statistical Insights")
        self.statistical_insights = self.generate_ai_statistical_insights(daily_returns)
        st.markdown(self.statistical_insights)

    def generate_ai_statistical_insights(self, daily_returns):
        if not GROQ_API_KEY:
            return "Please set your GROQ_API_KEY to enable AI-powered insights."
        
        market_returns = self.stock_data['Market_Close'].pct_change().dropna()
        correlation = daily_returns.corr(market_returns)
        shapiro_stat, shapiro_p = stats.shapiro(daily_returns)
        jb_stat, jb_p = stats.jarque_bera(daily_returns)
        
        context = f"""
        Stock: {st.session_state.ticker}
        Company: {self.stock_info.get('longName', 'N/A')}
        Sector: {self.stock_info.get('sector', 'N/A')}
        Industry: {self.stock_info.get('industry', 'N/A')}

        Analyze the following statistical data for the stock.
        - Mean Daily Return: {daily_returns.mean() * 100:.4f}%
        - Daily Returns Standard Deviation (Volatility): {daily_returns.std() * 100:.4f}%
        - Skewness: {daily_returns.skew():.4f}
        - Kurtosis: {daily_returns.kurtosis():.4f}
        - Autocorrelation (lag 1): {daily_returns.autocorr(lag=1):.4f}
        - Shapiro-Wilk Stat: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}
        - Jarque-Bera Stat: {jb_stat:.4f}, p-value: {jb_p:.4f}
        - Correlation with Market: {correlation:.4f}

        Provide a detailed analysis in bullet points:
        - Interpret each statistic in the context of the company's sector and industry.
        - Explain implications for investors, such as volatility level, potential for asymmetric returns, risk of extreme events, serial dependence, and normality.
        - Compare to typical market values.
        - Offer data-driven investment recommendations: buy, hold, or sell, with reasoning.
        - Suggest suitability for different investor types.

        Use a professional, analyst-like tone to aid data-driven decision-making.
        """
        return invoke_with_fallback(context, temperature=0.1)

    def risk_analysis(self):
        st.subheader("Risk & Probability Analysis")
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        confidence_levels = [0.90, 0.95, 0.99]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Value at Risk (VaR)")
            var_results = []
            for conf in confidence_levels:
                var = np.percentile(daily_returns, (1 - conf) * 100)
                var_results.append({'Confidence Level': f"{conf * 100}%", 'VaR': f"{-var * 100:.2f}%"})
            st.dataframe(pd.DataFrame(var_results))
            st.info("VaR represents the maximum expected loss over a specific period at a given confidence level.")
            
            # Additional: CVaR
            var_95 = np.percentile(daily_returns, 5)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            st.metric("Conditional VaR (95%)", f"{cvar_95 * 100:.2f}%")
            
            # Maximum Drawdown
            cum_returns = (1 + daily_returns).cumprod()
            peak = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns / peak) - 1
            max_drawdown = drawdown.min()
            st.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
            
        with col2:
            st.write("### Probability of Price Movements")
            mean = daily_returns.mean()
            std = daily_returns.std()
            
            prob_increase = 1 - stats.norm.cdf(0.01, loc=mean, scale=std)
            st.metric("Prob. of >1% Price Increase", f"{prob_increase * 100:.2f}%")
            
            prob_decrease = stats.norm.cdf(-0.01, loc=mean, scale=std)
            st.metric("Prob. of >1% Price Decrease", f"{prob_decrease * 100:.2f}%")
            
            prob_positive = 1 - stats.norm.cdf(0, loc=mean, scale=std)
            st.metric("Prob. of Positive Return", f"{prob_positive * 100:.2f}%")
            
            # Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else 0
            sortino_ratio = (mean - 0.0) / downside_std if downside_std != 0 else 0
            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
            
        st.write("### AI-Powered Risk Insights")
        self.risk_insights = self.generate_ai_risk_insights(daily_returns)
        st.markdown(self.risk_insights)

    def generate_ai_risk_insights(self, daily_returns):
        if not GROQ_API_KEY:
            return "Please set your GROQ_API_KEY to enable AI-powered insights."
        
        annualized_return = daily_returns.mean() * 252
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        risk_free_rate = 0.04
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        try:
            nifty_data = yf.download('^NSEI', start=st.session_state.start_date, end=st.session_state.end_date)
            nifty_returns = nifty_data['Close'].pct_change().dropna()
            covariance = daily_returns.cov(nifty_returns)
            market_variance = nifty_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 0
        except Exception:
            beta = "N/A"
            
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        cum_returns = (1 + daily_returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else 0
        sortino_ratio = (daily_returns.mean() - 0.0) / downside_std if downside_std != 0 else 0
        
        context = f"""
        Stock: {st.session_state.ticker}
        Company: {self.stock_info.get('longName', 'N/A')}
        Sector: {self.stock_info.get('sector', 'N/A')}
        Industry: {self.stock_info.get('industry', 'N/A')}

        Analyze the following data for the stock.
        - Annualized Return: {annualized_return:.2f}
        - Annualized Volatility: {annualized_volatility:.2f}
        - Sharpe Ratio: {sharpe_ratio:.2f}
        - Beta (vs Nifty 50): {beta if isinstance(beta, str) else f'{beta:.2f}'}
        - Max Daily Loss: {daily_returns.min():.2f}
        - Probability of >1% Daily Gain: {1 - stats.norm.cdf(0.01, loc=daily_returns.mean(), scale=daily_returns.std()):.2f}
        - Probability of >1% Daily Loss: {stats.norm.cdf(-0.01, loc=daily_returns.mean(), scale=daily_returns.std()):.2f}
        - Conditional VaR (95%): {cvar_95:.2f}
        - Maximum Drawdown: {max_drawdown:.2f}
        - Sortino Ratio: {sortino_ratio:.2f}
        
        Provide a detailed, professional risk assessment in bullet points:
        - Interpret each metric in the context of the stock's sector and industry.
        - Explain the stock's risk profile (e.g., high beta means more market-sensitive).
        - Discuss potential as an investment, including pros/cons.
        - Offer data-driven recommendations for buy/hold/sell.
        - Suggest risk management strategies and suitability for portfolio diversification.

        Focus on aiding data-driven investment decisions.
        """
        return invoke_with_fallback(context, temperature=0.1)

    def prepare_data_for_ml(self):
        data = self.stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 100  # Increased to 100 for better pattern capture
        X, y = create_sequences(scaled_data, seq_length)
        return train_test_split(X, y, test_size=0.2, random_state=42), scaler, seq_length

    def train_lstm(self, X_train, y_train, X_test, y_test, scaler, seq_length):
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)), 
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        with st.spinner("Training LSTM model..."):
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_original = scaler.inverse_transform(y_test)
        mse = mean_squared_error(y_test_original, predictions)
        return model, y_test_original, predictions, mse

    def train_linear_regression(self, X_train, y_train, X_test, y_test, scaler):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        model = LinearRegression()
        with st.spinner("Training Linear Regression model..."):
            model.fit(X_train_flat, y_train)
        predictions = model.predict(X_test_flat)
        predictions = scaler.inverse_transform(predictions)
        y_test_original = scaler.inverse_transform(y_test)
        mse = mean_squared_error(y_test_original, predictions)
        return model, y_test_original, predictions, mse

    def ml_price_prediction(self):
        st.subheader("ML Price Prediction Comparison")
        try:
            (X_train, X_test, y_train, y_test), scaler, seq_length = self.prepare_data_for_ml()
            
            st.info("Training both LSTM and Linear Regression models on the historical data to compare their performance.")

            # Custom loading animation for LSTM
            with st.spinner("Initializing LSTM model... (Step 1/3)"):
                import time; time.sleep(0.5)
            with st.spinner("Training LSTM model... (Step 2/3)"):
                lstm_model, y_lstm, pred_lstm, mse_lstm = self.train_lstm(X_train, y_train, X_test, y_test, scaler, seq_length)
            with st.spinner("Finalizing LSTM predictions... (Step 3/3)"):
                time.sleep(0.5)
            st.success("LSTM model trained successfully!")

            # Custom loading animation for Linear Regression
            with st.spinner("Training Linear Regression model..."):
                lr_model, y_lr, pred_lr, mse_lr = self.train_linear_regression(X_train, y_train, X_test, y_test, scaler)
            st.success("Linear Regression model trained successfully!")

            st.write("### Model Performance (Mean Squared Error)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("LSTM MSE", f"{mse_lstm:.4f}")
            with col2:
                st.metric("Linear Regression MSE", f"{mse_lr:.4f}")

            st.write("### Price Prediction Visualization (Test Set)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("LSTM Prediction")
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x=np.arange(len(y_lstm)), y=y_lstm.flatten(), mode='lines', name='Actual Price'))
                fig_lstm.add_trace(go.Scatter(x=np.arange(len(pred_lstm)), y=pred_lstm.flatten(), mode='lines', name='LSTM Predicted Price'))
                fig_lstm.update_layout(title=f'LSTM Price Prediction for {st.session_state.ticker}', xaxis_title='Time', yaxis_title='Price')
                st.plotly_chart(fig_lstm)
            
            with col2:
                st.write("Linear Regression Prediction")
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(x=np.arange(len(y_lr)), y=y_lr.flatten(), mode='lines', name='Actual Price'))
                fig_lr.add_trace(go.Scatter(x=np.arange(len(pred_lr)), y=pred_lr.flatten(), mode='lines', name='LR Predicted Price'))
                fig_lr.update_layout(title=f'LR Price Prediction for {st.session_state.ticker}', xaxis_title='Time', yaxis_title='Price')
                st.plotly_chart(fig_lr)

            # Future prediction with LSTM - 30 days
            st.write("### Future Price Prediction (LSTM - Next 30 Days)")
            data = self.stock_data['Close'].values.reshape(-1, 1)
            scaled_data = scaler.transform(data)
            last_sequence = scaled_data[-100:]  # Use last 100 days for better context
            future_days = 30
            future_predictions = []
            current_seq = last_sequence.copy()
            with st.spinner("Predicting next 30 days..."):
                for _ in range(future_days):
                    pred = lstm_model.predict(current_seq[-seq_length:].reshape(1, seq_length, 1))[0][0]
                    future_predictions.append(pred)
                    current_seq = np.append(current_seq[1:], [[pred]], axis=0)

            future_prices_lstm = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(self.stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)

            fig_future_lstm = go.Figure()
            fig_future_lstm.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['Close'][-100:], mode='lines', name='Historical Price'))
            fig_future_lstm.add_trace(go.Scatter(x=future_dates, y=future_prices_lstm.flatten(), mode='lines', name='LSTM Future Prediction'))
            fig_future_lstm.update_layout(title=f'Future Price Prediction (LSTM - 30 Days) for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_future_lstm)

            # Long-term predictions - 2 years
            st.write("### Long-Term Future Price Prediction (LSTM - Next 2 Years)")
            future_days_2yr = 730
            future_predictions_2yr = []
            current_seq = last_sequence.copy()
            with st.spinner("Predicting next 2 years..."):
                for _ in range(future_days_2yr):
                    pred = lstm_model.predict(current_seq[-seq_length:].reshape(1, seq_length, 1))[0][0]
                    future_predictions_2yr.append(pred)
                    current_seq = np.append(current_seq[1:], [[pred]], axis=0)

            future_prices_2yr = scaler.inverse_transform(np.array(future_predictions_2yr).reshape(-1, 1))
            future_dates_2yr = pd.date_range(self.stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days_2yr)

            fig_2yr = go.Figure()
            fig_2yr.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['Close'][-100:], mode='lines', name='Historical Price'))
            fig_2yr.add_trace(go.Scatter(x=future_dates_2yr, y=future_prices_2yr.flatten(), mode='lines', name='LSTM 2-Year Prediction'))
            fig_2yr.update_layout(title=f'2-Year Price Prediction (LSTM) for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_2yr)

            # Optional 5-year prediction
            if st.button("Show 5-Year Prediction"):
                st.write("### Long-Term Future Price Prediction (LSTM - Next 5 Years)")
                future_days_5yr = 1825
                future_predictions_5yr = []
                current_seq = last_sequence.copy()
                with st.spinner("Predicting next 5 years..."):
                    for _ in range(future_days_5yr):
                        pred = lstm_model.predict(current_seq[-seq_length:].reshape(1, seq_length, 1))[0][0]
                        future_predictions_5yr.append(pred)
                        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

                future_prices_5yr = scaler.inverse_transform(np.array(future_predictions_5yr).reshape(-1, 1))
                future_dates_5yr = pd.date_range(self.stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days_5yr)

                fig_5yr = go.Figure()
                fig_5yr.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['Close'][-100:], mode='lines', name='Historical Price'))
                fig_5yr.add_trace(go.Scatter(x=future_dates_5yr, y=future_prices_5yr.flatten(), mode='lines', name='LSTM 5-Year Prediction'))
                fig_5yr.update_layout(title=f'5-Year Price Prediction (LSTM) for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig_5yr)

            st.info("Note: Long-term predictions are highly uncertain and for illustrative purposes only.")

            next_day_pred_lstm = future_prices_lstm[0][0]
            pred_change_lstm = (next_day_pred_lstm - self.stock_data['Close'].iloc[-1]) / self.stock_data['Close'].iloc[-1] * 100
            next_day_pred_lr = pred_lr[-1][0] if len(pred_lr) > 0 else 0
            pred_change_lr = (next_day_pred_lr - self.stock_data['Close'].iloc[-1]) / self.stock_data['Close'].iloc[-1] * 100
            
            st.write("### AI-Powered ML Performance Insight")
            self.ml_insights = self.generate_ai_ml_insights(mse_lstm, mse_lr, next_day_pred_lstm, pred_change_lstm, next_day_pred_lr, pred_change_lr)
            st.markdown(self.ml_insights)

        except ValueError as e:
            st.error(f"Prediction models could not be trained. This may happen if the selected date range is too short or the data is insufficient. Error: {e}")

    def generate_ai_ml_insights(self, mse_lstm, mse_lr, next_day_pred_lstm, pred_change_lstm, next_day_pred_lr, pred_change_lr):
        if not GROQ_API_KEY:
            return "Please set your GROQ_API_KEY to enable AI-powered insights."
        
        context = f"""
        Act as a personal stock assistant for {st.session_state.ticker} ({self.stock_info.get('longName', 'N/A')}), in the {self.stock_info.get('sector', 'N/A')} sector.

        Analyze the ML prediction models:
        - LSTM MSE: {mse_lstm:.4f}, Next Day Pred: {next_day_pred_lstm:.2f} (Change: {pred_change_lstm:.2f}%)
        - Linear Regression MSE: {mse_lr:.4f}, Next Day Pred: {next_day_pred_lr:.2f} (Change: {pred_change_lr:.2f}%)
        - Current Price: {self.stock_data['Close'].iloc[-1]:.2f}
        - RSI: {self.stock_data['RSI'].iloc[-1]:.2f}
        - 20-Day SMA: {self.stock_data['SMA_20'].iloc[-1]:.2f}
        - 50-Day SMA: {self.stock_data['SMA_50'].iloc[-1]:.2f}

        Provide detailed insights in bullet points as a personal assistant:
        - Compare model performances, highlighting LSTM's strength in capturing non-linear trends and LR's simplicity for short-term linear patterns.
        - Assess reliability based on MSE, suggesting higher confidence in the model with lower error.
        - Analyze short-term (30-day) and long-term (2-year) trends, integrating RSI and SMA crossovers for momentum insights.
        - Provide a buy/sell/hold recommendation with reasoning, considering predicted changes and technical indicators.
        - Estimate prediction accuracy qualitatively, factoring in MSE, training data length, and market volatility.
        - Offer advice on using predictions for decisions, including limitations (e.g., long-term uncertainty) and synergy with statistical/risk analyses.
        - Suggest monitoring key levels (e.g., support/resistance from SMA or Bollinger Bands) for validation.

        Use an engaging, professional tone like a trusted advisor.
        """
        return invoke_with_fallback(context, temperature=0.1)

    # ---------------- CHAT ADVISOR ----------------
    def chat_advisor(self):
        st.subheader("💬 Ask the AI Advisor About This Stock")
        st.info("I'm a data-driven personal stock advisor. Ask me anything, like 'Is this a good long-term investment?' or 'What are the risks of this stock?'")
        
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Please set it in your .env file to use the AI advisor.")
            return

        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory()
        
        llm = get_llm(temperature=0)
        conversation = ConversationChain(llm=llm, memory=st.session_state.memory)
        
        stock_context = self.get_stock_context()
        
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != st.session_state.ticker:
            st.session_state.messages = [{"role": "assistant", "content": f"Hello! I am your AI stock advisor. I have analyzed the data for {st.session_state.ticker}. What would you like to know about its trends, risks, and potential as an investment?"}]
            st.session_state.memory.clear()
            st.session_state.memory.save_context({"input": stock_context}, {"output": "Understood, ready to advise."})
            st.session_state.last_ticker = st.session_state.ticker

        chat_container = st.container(height=400)
        
        for message in st.session_state.messages:
            with chat_container:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if user_query := st.chat_input("Ask about this stock..."):
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        try:
                            response = conversation.predict(input=user_query)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            # Retry once with explicit fallback model while preserving memory
                            try:
                                llm_fallback = ChatGroq(model=FALLBACK_MODEL, temperature=0, api_key=GROQ_API_KEY)
                                conversation_fallback = ConversationChain(llm=llm_fallback, memory=st.session_state.memory)
                                response = conversation_fallback.predict(input=user_query)
                                st.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            except Exception as e2:
                                st.error(f"Failed to get a response from the AI. Error: {e2}")

    def get_stock_context(self):
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        risk_free_rate = 0.04
        annualized_return = daily_returns.mean() * 252
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        try:
            nifty_data = yf.download('^NSEI', start=st.session_state.start_date, end=st.session_state.end_date)
            nifty_returns = nifty_data['Close'].pct_change().dropna()
            covariance = daily_returns.cov(nifty_returns)
            market_variance = nifty_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 0
        except Exception:
            beta = "N/A"
            
        stock_context = f"""
        You are a top financial advisor and stock analyst. Your responses should be professional, data-driven, and provide actionable, well-reasoned advice. Always reference the provided data in your analysis. Structure responses with bullet points where appropriate for clarity. Offer balanced views with pros/cons and specific recommendations to aid investment decisions.
        
        Company Name: {self.stock_info.get('longName', 'N/A')}
        Sector: {self.stock_info.get('sector', 'N/A')}
        Industry: {self.stock_info.get('industry', 'N/A')}
        Description: {self.stock_info.get('longBusinessSummary', 'N/A')[:1000]}

        Here is the latest data for {st.session_state.ticker}:
        - Latest Closing Price: {self.stock_data['Close'].iloc[-1]:.2f}
        - 20-Day SMA: {self.stock_data['SMA_20'].iloc[-1]:.2f}
        - 50-Day SMA: {self.stock_data['SMA_50'].iloc[-1]:.2f}
        - RSI (14-day): {self.stock_data['RSI'].iloc[-1]:.2f}
        - Annualized Return: {annualized_return * 100:.2f}%
        - Annualized Volatility: {annualized_volatility * 100:.2f}%
        - Sharpe Ratio: {sharpe_ratio:.2f}
        - Beta (vs Nifty 50): {beta if isinstance(beta, str) else f'{beta:.2f}'}
        - Value at Risk (95% conf.): {-np.percentile(daily_returns, 5) * 100:.2f}%
        - Probability of >1% daily loss: {stats.norm.cdf(-0.01, loc=daily_returns.mean(), scale=daily_returns.std()) * 100:.2f}%
        - Historical Price Range: {self.stock_data['Close'].min():.2f} to {self.stock_data['Close'].max():.2f}
        - Daily Returns Skewness: {daily_returns.skew():.4f}
        - Daily Returns Kurtosis: {daily_returns.kurtosis():.4f}
        
        Based on this data, provide personalized advice and analysis for the user's query. Interpret the numbers in a human-friendly way. For example, explain what a specific RSI value or SMA cross-over means for the stock's momentum.
        """
        return stock_context

    def trading_strategies(self):
        st.subheader("Trading Strategies")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Donchian Channel")
            fig_don = go.Figure()
            fig_don.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Closing Price'))
            fig_don.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Don_Upper'], mode='lines', name='Upper Channel'))
            fig_don.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Don_Lower'], mode='lines', name='Lower Channel'))
            fig_don.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Don_Mid'], mode='lines', name='Mid Channel', line=dict(dash='dash')))
            st.plotly_chart(fig_don)
            
            st.info("Buy when price breaks above upper channel, sell when below lower.")

        with col2:
            st.write("### 20-Day High/Low")
            fig_20d = go.Figure()
            fig_20d.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Closing Price'))
            fig_20d.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['20D_High'], mode='lines', name='20D High'))
            fig_20d.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['20D_Low'], mode='lines', name='20D Low'))
            st.plotly_chart(fig_20d)
            
            st.info("Breakout above 20-day high may signal buy, below low signal sell.")

        st.write("### Turtle Trader Signals")
        turtle_buy_dates = self.stock_data[self.stock_data['Turtle_Buy']].index
        turtle_sell_dates = self.stock_data[self.stock_data['Turtle_Sell']].index
        fig_turtle = go.Figure()
        fig_turtle.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Closing Price'))
        fig_turtle.add_trace(go.Scatter(x=turtle_buy_dates, y=self.stock_data.loc[turtle_buy_dates, 'Close'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up')))
        fig_turtle.add_trace(go.Scatter(x=turtle_sell_dates, y=self.stock_data.loc[turtle_sell_dates, 'Close'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))
        st.plotly_chart(fig_turtle)

        st.write("### Darvas Box (Simplified)")
        fig_darvas = go.Figure()
        fig_darvas.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'], mode='lines', name='Closing Price'))
        fig_darvas.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Darvas_High'], mode='lines', name='Darvas High'))
        fig_darvas.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data['Darvas_Low'], mode='lines', name='Darvas Low'))
        st.plotly_chart(fig_darvas)
        
        st.info("Buy when price breaks above Darvas high, with stop below low.")

        st.write("### AI-Powered Strategy Insights")
        self.strategy_insights = self.generate_ai_strategy_insights()
        st.markdown(self.strategy_insights)

    def generate_ai_strategy_insights(self):
        if not GROQ_API_KEY:
            return "Please set your GROQ_API_KEY to enable AI-powered insights."
        
        context = f"""
        Analyze trading strategies for {st.session_state.ticker}:
        - Donchian Channel: Upper {self.stock_data['Don_Upper'].iloc[-1]:.2f}, Lower {self.stock_data['Don_Lower'].iloc[-1]:.2f}
        - 20-Day High: {self.stock_data['20D_High'].iloc[-1]:.2f}, Low: {self.stock_data['20D_Low'].iloc[-1]:.2f}
        - Turtle Buy Signals: {self.stock_data['Turtle_Buy'].sum()}, Sell: {self.stock_data['Turtle_Sell'].sum()}
        - Darvas High: {self.stock_data['Darvas_High'].iloc[-1]:.2f}, Low: {self.stock_data['Darvas_Low'].iloc[-1]:.2f}

        Provide insights in bullet points:
        - Explain each strategy and current signals.
        - Suggest buy/sell/hold based on strategies.
        - Integrate with RSI ({self.stock_data['RSI'].iloc[-1]:.2f}) and other indicators.
        - Discuss risks and suitability.
        """
        return invoke_with_fallback(context, temperature=0.1)

    def trading_signals(self):
        st.subheader("Trading Signals")
        st.info("This section provides data-driven buy/sell signals for different trading horizons, with reasons and explanations backed by technical indicators.")

        # Helper function to generate AI insights for signals
        def generate_signal_insights(signal_type, signal_data):
            if not GROQ_API_KEY:
                return "Please set your GROQ_API_KEY to enable AI-powered insights."

            llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1, api_key=GROQ_API_KEY)
            context = f"""
            Act as a professional trading analyst for {st.session_state.ticker} ({self.stock_info.get('longName', 'N/A')}).
            Analyze the following data for {signal_type} trading signals:
            - Latest Closing Price: {self.stock_data['Close'].iloc[-1]:.2f}
            - RSI (14-day): {self.stock_data['RSI'].iloc[-1]:.2f}
            - 20-Day SMA: {self.stock_data['SMA_20'].iloc[-1]:.2f}
            - 50-Day SMA: {self.stock_data['SMA_50'].iloc[-1]:.2f}
            - 200-Day SMA: {fmt_f(self.stock_data['SMA_200'].iloc[-1]) if 'SMA_200' in self.stock_data else 'N/A'}
            - Bollinger Upper: {self.stock_data['BB_Upper'].iloc[-1]:.2f}
            - Bollinger Lower: {self.stock_data['BB_Lower'].iloc[-1]:.2f}
            - Donchian Upper: {self.stock_data['Don_Upper'].iloc[-1]:.2f}
            - Donchian Lower: {self.stock_data['Don_Lower'].iloc[-1]:.2f}
            - MACD: {fmt_f(self.stock_data['MACD'].iloc[-1]) if 'MACD' in self.stock_data else 'N/A'}
            - MACD Signal: {fmt_f(self.stock_data['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in self.stock_data else 'N/A'}
            - Additional Data: {signal_data}

            Provide a concise analysis in bullet points:
            - Current signal (Buy/Sell/Hold) based on the data.
            - Reason for the signal, referencing specific indicators.
            - Short explanation of why this signal is relevant for {signal_type} trading.
            - Data-backed confidence level (e.g., high/medium/low based on indicator alignment).
            - Risk considerations and suggested stop-loss levels.
            Use a professional, data-driven tone.
            """
            return llm.invoke(context).content

        # Long-Term Signals (based on 200-day SMA, RSI, and fundamental data)
        st.write("### Long-Term Signals (6+ Months)")
        long_term_data = f"""
        - Price vs 200-day SMA: {'Above' if self.stock_data['Close'].iloc[-1] > self.stock_data['SMA_200'].iloc[-1] else 'Below'}
        - Fundamental Metrics: Trailing P/E: {self.stock_info.get('trailingPE', 'N/A')}, Forward P/E: {self.stock_info.get('forwardPE', 'N/A')}
        """
        long_term_insights = generate_signal_insights("Long-Term", long_term_data)
        st.markdown(long_term_insights)

        # Short-Term Signals (based on 20-day/50-day SMA crossover, RSI)
        st.write("### Short-Term Signals (1-3 Months)")
        short_term_data = f"""
        - 20-day vs 50-day SMA: {'Golden Cross' if self.stock_data['SMA_20'].iloc[-1] > self.stock_data['SMA_50'].iloc[-1] else 'Death Cross'}
        - RSI Trend: {'Overbought' if self.stock_data['RSI'].iloc[-1] > 70 else 'Oversold' if self.stock_data['RSI'].iloc[-1] < 30 else 'Neutral'}
        """
        short_term_insights = generate_signal_insights("Short-Term", short_term_data)
        st.markdown(short_term_insights)

        # Swing Trade Signals (based on Bollinger Bands, MACD)
        st.write("### Swing Trade Signals (1-4 Weeks)")
        swing_trade_data = f"""
        - Price vs Bollinger Bands: {'Near Upper' if self.stock_data['Close'].iloc[-1] >= self.stock_data['BB_Upper'].iloc[-1] * 0.95 else 'Near Lower' if self.stock_data['Close'].iloc[-1] <= self.stock_data['BB_Lower'].iloc[-1] * 1.05 else 'Neutral'}
        - MACD vs Signal: {'Bullish' if self.stock_data['MACD'].iloc[-1] > self.stock_data['MACD_Signal'].iloc[-1] else 'Bearish'}
        """
        swing_trade_insights = generate_signal_insights("Swing Trade", swing_trade_data)
        st.markdown(swing_trade_insights)

        # Intraday Signals (based on Donchian Channels, RSI, MACD)
        st.write("### Intraday Signals (Same Day)")
        intraday_data = f"""
        - Price vs Donchian Channels: {'Breakout' if self.stock_data['Close'].iloc[-1] > self.stock_data['Don_Upper'].iloc[-1] else 'Breakdown' if self.stock_data['Close'].iloc[-1] < self.stock_data['Don_Lower'].iloc[-1] else 'Neutral'}
        - MACD vs Signal: {'Bullish' if self.stock_data['MACD'].iloc[-1] > self.stock_data['MACD_Signal'].iloc[-1] else 'Bearish'}
        """
        intraday_insights = generate_signal_insights("Intraday", intraday_data)
        st.markdown(intraday_insights)

        # Visualization of key indicators for signals
        st.write("### Signal Indicators Visualization")
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['Close'].iloc[-100:], mode='lines', name='Closing Price'))
        fig_signals.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['SMA_20'].iloc[-100:], mode='lines', name='20-Day SMA', line=dict(color='orange')))
        fig_signals.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['SMA_50'].iloc[-100:], mode='lines', name='50-Day SMA', line=dict(color='purple')))
        fig_signals.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['BB_Upper'].iloc[-100:], mode='lines', name='Bollinger Upper', line=dict(color='red')))
        fig_signals.add_trace(go.Scatter(x=self.stock_data.index[-100:], y=self.stock_data['BB_Lower'].iloc[-100:], mode='lines', name='Bollinger Lower', line=dict(color='green')))
        fig_signals.update_layout(title=f'Technical Indicators for {st.session_state.ticker}', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_signals)

# ---------------- MAIN ----------------
def main():
    StockAnalysisApp()

if __name__ == "__main__":
    main()