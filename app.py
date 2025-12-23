"""
Enhanced Trading Intelligence Platform v4.0
Professional-grade personal trading dashboard with advanced features
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Application configuration"""
    DEFAULT_TICKER: str = "AAPL"
    CACHE_TTL: int = 60
    MAX_NEWS_ITEMS: int = 5
    RSI_WINDOW: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    DEFAULT_WATCHLIST: List[str] = None
    MIN_PASSWORD_LENGTH: int = 8
    MOMENTUM_DAYS: int = 14
    MAX_WORKERS: int = 5
    
    def __post_init__(self):
        if self.DEFAULT_WATCHLIST is None:
            self.DEFAULT_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "AMD", "META", "GOOG", "NFLX"]

class RiskTier(Enum):
    """Risk classification"""
    LOW = (0.8, 0.15, "🟢")
    MEDIUM = (1.2, 0.10, "🟡")
    HIGH = (float('inf'), 0.05, "🔴")
    
    @classmethod
    def from_beta(cls, beta: float) -> 'RiskTier':
        if beta < 0.8:
            return cls.LOW
        elif beta < 1.2:
            return cls.MEDIUM
        return cls.HIGH

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Consumer Staples": "XLP",
    "Communication Services": "XLC"
}

SCREENER_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "WMT",
    "JNJ", "PG", "MA", "HD", "BAC", "DIS", "NFLX", "INTC", "AMD", "PYPL",
    "CSCO", "PFE", "KO", "PEP", "ABBV", "MRK", "TMO", "COST", "AVGO", "LLY"
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Trading Intelligence Platform v4.0",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Trading Intelligence Platform v4.0 - Professional Edition"
    }
)

# ============================================================================
# AUTHENTICATION WITH HASHING
# ============================================================================

class AuthManager:
    """Secure authentication with password hashing"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def load_credentials() -> Dict[str, str]:
        """Load stored credentials"""
        try:
            if os.path.exists('credentials.json'):
                with open('credentials.json', 'r') as f:
                    return json.load(f)
        except:
            pass
        # Default credentials (hashed "mytrader2024")
        default_hash = "3c9909afec25354d551dae21590bb26e38d53f2173b8d3dc3eee4c047e7ab1c1"
        return {"admin": default_hash}
    
    @staticmethod
    def save_credentials(creds: Dict[str, str]):
        """Save credentials"""
        try:
            with open('credentials.json', 'w') as f:
                json.dump(creds, f)
        except Exception as e:
            logger.error(f"Error saving credentials: {str(e)}")
    
    @staticmethod
    def check_password() -> bool:
        """Check authentication with improved security"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.username = None
        
        if st.session_state.authenticated:
            return True
        
        with st.container():
            st.title("🔐 Trading Intelligence Platform")
            st.caption("Secure access to your personal trading hub")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                username = st.text_input("Username", value="admin")
                password = st.text_input("Password", type="password", help="Default: mytrader2024")
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("🔓 Login", use_container_width=True):
                        creds = AuthManager.load_credentials()
                        hashed = AuthManager.hash_password(password)
                        
                        if username in creds and creds[username] == hashed:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                
                with col_btn2:
                    if st.button("🔑 Change Password", use_container_width=True):
                        st.session_state.show_change_password = True
                
                # Change password dialog
                if st.session_state.get('show_change_password', False):
                    st.divider()
                    st.subheader("Change Password")
                    old_pass = st.text_input("Current Password", type="password", key="old_pass")
                    new_pass = st.text_input("New Password", type="password", key="new_pass")
                    confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
                    
                    if st.button("Update Password"):
                        creds = AuthManager.load_credentials()
                        old_hash = AuthManager.hash_password(old_pass)
                        
                        if username in creds and creds[username] == old_hash:
                            if new_pass == confirm_pass and len(new_pass) >= 8:
                                creds[username] = AuthManager.hash_password(new_pass)
                                AuthManager.save_credentials(creds)
                                st.success("✅ Password updated successfully!")
                                st.session_state.show_change_password = False
                            else:
                                st.error("❌ Passwords don't match or too short (min 8 chars)")
                        else:
                            st.error("❌ Current password incorrect")
        
        return False
    
    @staticmethod
    def logout():
        """Logout user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# ============================================================================
# ENHANCED DATA FETCHER WITH THREADING
# ============================================================================

class DataFetcher:
    """Enhanced data fetching with parallel processing"""
    
    @staticmethod
    @st.cache_data(ttl=Config().CACHE_TTL)
    def fetch_stock_data(ticker: str, period: str = '1y') -> Tuple[Optional[yf.Ticker], Optional[Dict], Optional[pd.DataFrame], Optional[str]]:
        """Fetch comprehensive stock data with fallback"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'currentPrice' not in info:
                hist = stock.history(period=period)
                if hist.empty:
                    return None, None, None, f"No data for '{ticker}'"
                
                info = {
                    'currentPrice': hist['Close'].iloc[-1],
                    'longName': ticker,
                    'symbol': ticker
                }
            
            hist = stock.history(period=period)
            
            if hist.empty:
                return None, None, None, f"No historical data for '{ticker}'"
            
            return stock, info, hist, None
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            return None, None, None, f"Error: {str(e)}"
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_news(ticker: str, limit: int = Config().MAX_NEWS_ITEMS) -> Tuple[List, Optional[str]]:
        """Fetch stock news"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            return news[:limit], None
        except:
            return [], "News unavailable"
    
    @staticmethod
    def fetch_multiple_stocks_parallel(tickers: List[str], period: str = '1mo') -> Dict:
        """Fetch multiple stocks in parallel for better performance"""
        results = {}
        
        def fetch_single(ticker):
            try:
                _, info, hist, _ = DataFetcher.fetch_stock_data(ticker, period=period)
                if hist is not None and not hist.empty:
                    return ticker, {'info': info, 'hist': hist}
            except:
                pass
            return ticker, None
        
        with ThreadPoolExecutor(max_workers=Config().MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_single, ticker): ticker for ticker in tickers}
            
            for future in as_completed(futures):
                ticker, data = future.result()
                if data:
                    results[ticker] = data
        
        return results
    
    @staticmethod
    def analyze_sentiment(headlines: List) -> Tuple[str, str, float]:
        """Enhanced sentiment analysis with score"""
        if not headlines:
            return "Neutral", "⚖️", 0.0
        
        pos_words = ['growth', 'profit', 'surge', 'beat', 'upgraded', 'buy', 'bullish', 'gains', 
                     'rally', 'soar', 'increase', 'positive', 'strong', 'record', 'outperform']
        neg_words = ['fall', 'loss', 'decline', 'miss', 'downgraded', 'sell', 'bearish', 'crash', 
                     'plunge', 'drop', 'warn', 'cut', 'lower', 'negative', 'weak', 'underperform']
        
        score = 0
        total_words = 0
        all_text = " ".join([h.get('title', '').lower() for h in headlines])
        
        for w in pos_words:
            count = all_text.count(w)
            score += count * 2
            total_words += count
            
        for w in neg_words:
            count = all_text.count(w)
            score -= count * 2
            total_words += count
        
        # Normalize score
        normalized_score = score / max(total_words, 1)
        
        if score > 3:
            return "Very Bullish", "🚀", normalized_score
        elif score > 1:
            return "Bullish", "📈", normalized_score
        elif score < -3:
            return "Very Bearish", "💥", normalized_score
        elif score < -1:
            return "Bearish", "📉", normalized_score
        return "Neutral", "⚖️", normalized_score
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_earnings_calendar(ticker: str) -> Optional[Dict]:
        """Fetch earnings date information"""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and not calendar.empty:
                earnings_date = calendar.get('Earnings Date')
                if earnings_date is not None:
                    if isinstance(earnings_date, pd.Series):
                        earnings_date = earnings_date.iloc[0] if len(earnings_date) > 0 else None
                    
                    return {
                        'earnings_date': earnings_date,
                        'has_earnings': True
                    }
            
            return {'has_earnings': False}
        except:
            return {'has_earnings': False}

# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================

class TechnicalAnalysis:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = Config().RSI_WINDOW) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = data.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(data), index=data.index)
    
    @staticmethod
    def calculate_macd(data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        try:
            exp1 = data.ewm(span=12, adjust=False).mean()
            exp2 = data.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            return macd, signal
        except:
            zeros = pd.Series([0] * len(data), index=data.index)
            return zeros, zeros
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = data.rolling(window=window).mean()
            std = data.rolling(window=window).std()
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            return sma, upper, lower
        except:
            return data, data, data
    
    @staticmethod
    def get_rsi_signal(rsi_value: float) -> Tuple[str, str]:
        """Interpret RSI value"""
        if pd.isna(rsi_value):
            return "Neutral", "⚪"
        elif rsi_value >= Config().RSI_OVERBOUGHT:
            return "Overbought", "🔴"
        elif rsi_value <= Config().RSI_OVERSOLD:
            return "Oversold", "🟢"
        else:
            return "Neutral", "🟡"
    
    @staticmethod
    def calculate_simple_moving_average(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_momentum(hist: pd.DataFrame, days: int = Config().MOMENTUM_DAYS) -> float:
        """Calculate price momentum over specified days"""
        try:
            if len(hist) < days:
                return 0
            return ((hist['Close'].iloc[-1] - hist['Close'].iloc[-days]) / hist['Close'].iloc[-days]) * 100
        except:
            return 0

# ============================================================================
# PORTFOLIO MANAGER WITH VALIDATION
# ============================================================================

class PortfolioManager:
    """Enhanced portfolio tracking with validation"""
    
    @staticmethod
    def initialize():
        """Initialize portfolio in session state"""
        if 'portfolio' not in st.session_state:
            portfolio_data = PortfolioManager.load_from_file()
            st.session_state.portfolio = portfolio_data.get('portfolio', [])
            st.session_state.transactions = portfolio_data.get('transactions', [])
    
    @staticmethod
    def validate_transaction(ticker: str, shares: float, price: float, 
                           trans_type: str) -> Tuple[bool, str]:
        """Validate transaction before adding"""
        if not ticker or shares <= 0 or price <= 0:
            return False, "Invalid transaction data"
        
        if trans_type == "SELL":
            # Check if user has enough shares
            current_shares = 0
            for holding in st.session_state.portfolio:
                if holding['ticker'] == ticker:
                    current_shares = holding['shares']
                    break
            
            if shares > current_shares:
                return False, f"Cannot sell {shares} shares. You only have {current_shares} shares of {ticker}"
        
        return True, "Valid"
    
    @staticmethod
    def add_transaction(ticker: str, shares: float, price: float, 
                       trans_type: str, date: datetime):
        """Add a buy/sell transaction with validation"""
        valid, message = PortfolioManager.validate_transaction(ticker, shares, price, trans_type)
        
        if not valid:
            return False, message
        
        transaction = {
            'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
            'ticker': ticker.upper(),
            'shares': shares,
            'price': price,
            'type': trans_type,
            'total': shares * price
        }
        
        st.session_state.transactions.append(transaction)
        PortfolioManager.update_portfolio()
        PortfolioManager.save_to_file()
        
        return True, "Transaction added successfully"
    
    @staticmethod
    def update_portfolio():
        """Recalculate portfolio from transactions"""
        portfolio = {}
        
        for trans in st.session_state.transactions:
            ticker = trans['ticker']
            
            if ticker not in portfolio:
                portfolio[ticker] = {'shares': 0, 'cost_basis': 0}
            
            if trans['type'] == 'BUY':
                portfolio[ticker]['shares'] += trans['shares']
                portfolio[ticker]['cost_basis'] += trans['total']
            else:  # SELL
                portfolio[ticker]['shares'] -= trans['shares']
                portfolio[ticker]['cost_basis'] -= trans['total']
        
        st.session_state.portfolio = [
            {'ticker': k, **v} for k, v in portfolio.items() 
            if v['shares'] > 0
        ]
    
    @staticmethod
    def get_portfolio_value() -> Tuple[float, float, float]:
        """Calculate portfolio metrics"""
        if not st.session_state.portfolio:
            return 0.0, 0.0, 0.0
        
        total_value = 0.0
        total_cost = 0.0
        
        for holding in st.session_state.portfolio:
            try:
                _, info, _, _ = DataFetcher.fetch_stock_data(holding['ticker'], period='1d')
                if info and 'currentPrice' in info:
                    current_price = info['currentPrice']
                    total_value += current_price * holding['shares']
                    total_cost += holding['cost_basis']
            except:
                continue
        
        return total_value, total_cost, total_value - total_cost
    
    @staticmethod
    def calculate_realized_gains() -> float:
        """Calculate realized gains/losses from sold positions"""
        realized = 0.0
        positions = {}
        
        for trans in st.session_state.transactions:
            ticker = trans['ticker']
            
            if ticker not in positions:
                positions[ticker] = {'shares': 0, 'total_cost': 0}
            
            if trans['type'] == 'BUY':
                positions[ticker]['shares'] += trans['shares']
                positions[ticker]['total_cost'] += trans['total']
            else:  # SELL
                if positions[ticker]['shares'] > 0:
                    avg_cost = positions[ticker]['total_cost'] / positions[ticker]['shares']
                    realized += (trans['price'] - avg_cost) * trans['shares']
                    positions[ticker]['shares'] -= trans['shares']
                    positions[ticker]['total_cost'] -= avg_cost * trans['shares']
        
        return realized
    
    @staticmethod
    def save_to_file():
        """Save portfolio to local file"""
        try:
            data = {
                'portfolio': st.session_state.portfolio,
                'transactions': st.session_state.transactions
            }
            with open('my_portfolio.json', 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Portfolio saved successfully")
        except Exception as e:
            logger.error(f"Error saving portfolio: {str(e)}")
    
    @staticmethod
    def load_from_file() -> Dict:
        """Load portfolio from file"""
        try:
            if os.path.exists('my_portfolio.json'):
                with open('my_portfolio.json', 'r') as f:
                    data = json.load(f)
                    logger.info("Portfolio loaded successfully")
                    return {
                        'portfolio': data.get('portfolio', []),
                        'transactions': data.get('transactions', [])
                    }
        except Exception as e:
            logger.error(f"Error loading portfolio: {str(e)}")
        
        return {'portfolio': [], 'transactions': []}
    
    @staticmethod
    def export_for_taxes() -> pd.DataFrame:
        """Export transaction history formatted for tax reporting"""
        if not st.session_state.transactions:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.transactions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate gains for each sell transaction
        tax_records = []
        holdings = {}
        
        for _, trans in df.iterrows():
            ticker = trans['ticker']
            
            if trans['type'] == 'BUY':
                if ticker not in holdings:
                    holdings[ticker] = []
                holdings[ticker].append({
                    'date': trans['date'],
                    'shares': trans['shares'],
                    'price': trans['price']
                })
            else:  # SELL
                if ticker in holdings and holdings[ticker]:
                    # FIFO method
                    shares_to_sell = trans['shares']
                    total_cost_basis = 0
                    
                    while shares_to_sell > 0 and holdings[ticker]:
                        lot = holdings[ticker][0]
                        
                        if lot['shares'] <= shares_to_sell:
                            total_cost_basis += lot['shares'] * lot['price']
                            shares_to_sell -= lot['shares']
                            holdings[ticker].pop(0)
                        else:
                            total_cost_basis += shares_to_sell * lot['price']
                            lot['shares'] -= shares_to_sell
                            shares_to_sell = 0
                    
                    proceeds = trans['shares'] * trans['price']
                    gain_loss = proceeds - total_cost_basis
                    
                    tax_records.append({
                        'Date Sold': trans['date'].strftime('%Y-%m-%d'),
                        'Ticker': ticker,
                        'Shares': trans['shares'],
                        'Proceeds': proceeds,
                        'Cost Basis': total_cost_basis,
                        'Gain/Loss': gain_loss,
                        'Holding Period': 'Long' if (trans['date'] - lot['date']).days > 365 else 'Short'
                    })
        
        return pd.DataFrame(tax_records)

# ============================================================================
# ALERTS MANAGER
# ============================================================================

class AlertsManager:
    """Manage price alerts"""
    
    @staticmethod
    def check_alerts():
        """Check if any price alerts have been triggered"""
        if 'price_alerts' not in st.session_state:
            st.session_state.price_alerts = {}
        if 'alerts_triggered' not in st.session_state:
            st.session_state.alerts_triggered = set()
            
        triggered_now = []
        
        for ticker, alert_info in st.session_state.price_alerts.items():
            try:
                _, info, _, _ = DataFetcher.fetch_stock_data(ticker, period='1d')
                if not info:
                    continue
                
                current_price = info.get('currentPrice', 0)
                alert_type = alert_info['type']
                alert_price = alert_info['price']
                alert_key = f"{ticker}_{alert_type}_{alert_price}"
                
                if alert_key in st.session_state.alerts_triggered:
                    continue
                
                if alert_type == "Above" and current_price > alert_price:
                    st.toast(f"🚨 {ticker} above ${alert_price:.2f}! Currently ${current_price:.2f}", icon="📈")
                    st.session_state.alerts_triggered.add(alert_key)
                    triggered_now.append(ticker)
                    
                elif alert_type == "Below" and current_price < alert_price:
                    st.toast(f"🚨 {ticker} below ${alert_price:.2f}! Currently ${current_price:.2f}", icon="📉")
                    st.session_state.alerts_triggered.add(alert_key)
                    triggered_now.append(ticker)
                    
            except Exception as e:
                logger.error(f"Error checking alert for {ticker}: {str(e)}")
                continue
        
        return triggered_now
    
    @staticmethod
    def save_alerts():
        """Save alerts to file"""
        try:
            with open('my_alerts.json', 'w') as f:
                json.dump(st.session_state.price_alerts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alerts: {str(e)}")
    
    @staticmethod
    def load_alerts():
        """Load alerts from file"""
        try:
            if os.path.exists('my_alerts.json'):
                with open('my_alerts.json', 'r') as f:
                    st.session_state.price_alerts = json.load(f)
        except Exception as e:
            logger.error(f"Error loading alerts: {str(e)}")

# ============================================================================
# EARNINGS SENTRY
# ============================================================================

class EarningsSentry:
    """Monitor upcoming earnings dates"""
    
    @staticmethod
    def get_upcoming_earnings(watchlist: List[str]) -> List[Dict]:
        """Get earnings dates for watchlist"""
        earnings_data = []
        
        for ticker in watchlist:
            try:
                earnings_info = DataFetcher.fetch_earnings_calendar(ticker)
                
                if earnings_info and earnings_info.get('has_earnings'):
                    earnings_date = earnings_info['earnings_date']
                    
                    if earnings_date:
                        days_until = (pd.to_datetime(earnings_date) - pd.Timestamp.now()).days
                        
                        if -5 <= days_until <= 30:  # Show earnings from 5 days ago to 30 days ahead
                            earnings_data.append({
                                'ticker': ticker,
                                'date': earnings_date,
                                'days_until': days_until,
                                'status': '✅ Reported' if days_until < 0 else '🔔 Upcoming'
                            })
            except Exception as e:
                logger.error(f"Error fetching earnings for {ticker}: {str(e)}")
                continue
        
        return sorted(earnings_data, key=lambda x: x['days_until'])
    
    @staticmethod
    def render_earnings_widget(watchlist: List[str]):
        """Render earnings calendar widget"""
        earnings = EarningsSentry.get_upcoming_earnings(watchlist)
        
        if not earnings:
            st.info("📅 No upcoming earnings in the next 30 days")
            return
        
        st.subheader("📅 Earnings Sentry")
        
        for item in earnings[:10]:
            col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
            
            with col1:
                st.write(f"**{item['ticker']}**")
            
            with col2:
                if isinstance(item['date'], pd.Timestamp):
                    date_str = item['date'].strftime('%Y-%m-%d')
                else:
                    date_str = str(item['date'])[:10]
                st.write(date_str)
            
            with col3:
                days = item['days_until']
                if days < 0:
                    st.write(f"{abs(days)} days ago")
                elif days == 0:
                    st.write("**TODAY** 🔥")
                else:
                    st.write(f"in {days} days")
            
            with col4:
                st.write(item['status'])

# ============================================================================
# CORRELATION ANALYZER
# ============================================================================

class CorrelationAnalyzer:
    """Analyze correlations between stocks"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_correlation_matrix(tickers: List[str], period: str = '1y') -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for given tickers"""
        try:
            # Fetch all data
            price_data = {}
            
            for ticker in tickers:
                _, _, hist, error = DataFetcher.fetch_stock_data(ticker, period)
                if hist is not None and not hist.empty:
                    price_data[ticker] = hist['Close']
            
            if len(price_data) < 2:
                return None
            
            # Create dataframe and calculate returns
            df = pd.DataFrame(price_data)
            returns = df.pct_change().dropna()
            
            # Calculate correlation
            correlation = returns.corr()
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return None
    
    @staticmethod
    def render_correlation_heatmap(tickers: List[str]):
        """Render correlation heatmap"""
        st.subheader("📊 Portfolio Correlation Matrix")
        st.caption("Understanding how your holdings move together")
        
        if len(tickers) < 2:
            st.warning("⚠️ Need at least 2 stocks to calculate correlation")
            return
        
        with st.spinner("Calculating correlations..."):
            corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(tickers)
        
        if corr_matrix is None:
            st.error("❌ Unable to calculate correlation matrix")
            return
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdYlGn',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap (1 Year Returns)",
            template="plotly_dark",
            height=600,
            xaxis_title="",
            yaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("💡 Correlation Insights")
        
        # Find highly correlated pairs
        high_corr = []
        low_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                pair = f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}"
                
                if corr_val > 0.7:
                    high_corr.append((pair, corr_val))
                elif corr_val < 0.3:
                    low_corr.append((pair, corr_val))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔗 Highly Correlated (>0.7)**")
            if high_corr:
                for pair, corr in sorted(high_corr, key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"• {pair}: {corr:.2f}")
                st.caption("These stocks tend to move together - consider diversification")
            else:
                st.write("No highly correlated pairs found")
        
        with col2:
            st.write("**🔀 Low Correlation (<0.3)**")
            if low_corr:
                for pair, corr in sorted(low_corr, key=lambda x: x[1])[:5]:
                    st.write(f"• {pair}: {corr:.2f}")
                st.caption("These stocks provide good diversification")
            else:
                st.write("No low correlation pairs found")

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """Simple backtesting for strategies"""
    
    @staticmethod
    def simple_ma_crossover(ticker: str, short_window: int = 20, long_window: int = 50, 
                           initial_capital: float = 10000, period: str = '2y') -> Dict:
        """Backtest simple moving average crossover strategy"""
        try:
            _, _, hist, error = DataFetcher.fetch_stock_data(ticker, period)
            
            if error or hist is None or hist.empty:
                return None
            
            # Calculate moving averages
            hist['SMA_short'] = hist['Close'].rolling(window=short_window).mean()
            hist['SMA_long'] = hist['Close'].rolling(window=long_window).mean()
            
            # Generate signals
            hist['Signal'] = 0
            hist.loc[hist['SMA_short'] > hist['SMA_long'], 'Signal'] = 1
            hist['Position'] = hist['Signal'].diff()
            
            # Calculate returns
            capital = initial_capital
            shares = 0
            trades = []
            
            for idx, row in hist.iterrows():
                if row['Position'] == 1:  # Buy signal
                    shares = capital / row['Close']
                    trades.append({
                        'date': idx,
                        'type': 'BUY',
                        'price': row['Close'],
                        'shares': shares
                    })
                elif row['Position'] == -1 and shares > 0:  # Sell signal
                    capital = shares * row['Close']
                    trades.append({
                        'date': idx,
                        'type': 'SELL',
                        'price': row['Close'],
                        'value': capital
                    })
                    shares = 0
            
            # Final value
            final_value = capital if shares == 0 else shares * hist['Close'].iloc[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            
            # Buy and hold comparison
            buy_hold_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[long_window]) / 
                              hist['Close'].iloc[long_window]) * 100
            
            return {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'trades': trades,
                'num_trades': len(trades),
                'hist': hist
            }
            
        except Exception as e:
            logger.error(f"Backtesting error: {str(e)}")
            return None

# ============================================================================
# UTILITIES
# ============================================================================

def safe_get(info: Dict, key: str, default='N/A', formatter=None):
    """Safely extract and format data from info dict"""
    value = info.get(key, default)
    
    if value == default or value is None:
        return default
    
    if formatter:
        try:
            return formatter(value)
        except:
            return default
    
    return value

def format_large_number(num: float) -> str:
    """Format large numbers with B/M/K suffixes"""
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def calculate_position_size(capital: float, risk_pct: float, entry_price: float, 
                           stop_loss_price: float) -> Tuple[int, float]:
    """Calculate position size based on risk management"""
    if stop_loss_price <= 0 or entry_price <= 0 or stop_loss_price >= entry_price:
        return 0, 0.0
    
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share == 0:
        return 0, 0.0
    
    risk_amount = capital * (risk_pct / 100)
    shares = int(risk_amount / risk_per_share)
    total_cost = shares * entry_price
    
    return shares, total_cost

# ============================================================================
# JOURNAL MANAGER
# ============================================================================

class JournalManager:
    """Manage trading journal"""
    
    @staticmethod
    def get_today_filename():
        """Get today's journal filename"""
        return f"trading_journal_{datetime.now().strftime('%Y%m%d')}.txt"
    
    @staticmethod
    def load_today_notes():
        """Load today's journal notes"""
        try:
            filename = JournalManager.get_today_filename()
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading journal: {str(e)}")
        return ""
    
    @staticmethod
    def save_notes(notes: str):
        """Save journal notes"""
        try:
            filename = JournalManager.get_today_filename()
            with open(filename, 'w') as f:
                f.write(notes)
            logger.info(f"Journal saved: {filename}")
        except Exception as e:
            logger.error(f"Error saving journal: {str(e)}")

# ============================================================================
# SIDEBAR UTILITIES
# ============================================================================

def quick_tips():
    """Personal trading reminders and tools"""
    
    AlertsManager.load_alerts()
    
    # Trading Journal
    with st.sidebar.expander("📝 Trading Journal", expanded=False):
        if 'daily_notes' not in st.session_state:
            st.session_state.daily_notes = JournalManager.load_today_notes()
        
        notes = st.text_area(
            "Today's observations",
            value=st.session_state.daily_notes,
            height=120,
            key="notes_input"
        )
        
        if notes != st.session_state.daily_notes:
            st.session_state.daily_notes = notes
            JournalManager.save_notes(notes)
        
        if st.button("💾 Save", use_container_width=True):
            JournalManager.save_notes(notes)
            st.success("Saved!")
    
    # Price Alerts
    with st.sidebar.expander("🔔 Price Alerts", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            alert_ticker = st.text_input("Ticker", key="alert_ticker").upper()
        with col2:
            alert_type = st.selectbox("Alert", ["Above", "Below"], key="alert_type")
        
        alert_price = st.number_input("Price $", value=0.0, step=1.0, key="alert_price")
        
        if st.button("➕ Set Alert", use_container_width=True):
            if alert_ticker and alert_price > 0:
                st.session_state.price_alerts[alert_ticker] = {
                    'type': alert_type,
                    'price': alert_price,
                    'set_at': datetime.now().isoformat()
                }
                AlertsManager.save_alerts()
                st.success(f"Alert set for {alert_ticker}")
        
        if st.session_state.price_alerts:
            st.write("**Active:**")
            for ticker, alert in list(st.session_state.price_alerts.items()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"{ticker}: {alert['type']} ${alert['price']:.2f}")
                with col2:
                    if st.button("❌", key=f"del_{ticker}"):
                        del st.session_state.price_alerts[ticker]
                        AlertsManager.save_alerts()
                        st.rerun()
    
    # Quick Actions
    with st.sidebar.expander("⚡ Quick Actions", expanded=False):
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Refreshed!")
            st.rerun()
        
        if st.button("📊 Export Portfolio", use_container_width=True):
            if st.session_state.portfolio:
                df = pd.DataFrame(st.session_state.portfolio)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download",
                    csv,
                    "portfolio.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.info("Portfolio empty")

# ============================================================================
# PAGE RENDERERS
# ============================================================================

def render_dashboard(watchlist):
    """Enhanced dashboard with earnings sentry"""
    st.title("🏠 Trading Dashboard")
    
    # Market sentiment
    with st.spinner("Loading market data..."):
        spy_news, _ = DataFetcher.fetch_news("SPY", limit=10)
        if spy_news:
            sentiment, emoji, score = DataFetcher.analyze_sentiment(spy_news)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Market Sentiment", f"{emoji} {sentiment}")
            with col2:
                st.metric("Sentiment Score", f"{score:.2f}")
            with col3:
                st.metric("News Items", len(spy_news))
            with col4:
                st.metric("Username", st.session_state.username)
    
    st.divider()
    
    # Earnings Sentry
    EarningsSentry.render_earnings_widget(watchlist)
    
    st.divider()
    
    # Quick Watch
    st.subheader("📈 Quick Watch")
    
    all_data = DataFetcher.fetch_multiple_stocks_parallel(watchlist[:10])
    
    if not all_data:
        st.warning("No data available")
        return
    
    cols = st.columns(min(5, len(all_data)))
    
    for idx, (ticker, data) in enumerate(list(all_data.items())[:5]):
        with cols[idx % 5]:
            try:
                hist = data['hist']
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    momentum = TechnicalAnalysis.calculate_momentum(hist)
                    momentum_emoji = "🔥" if momentum > 5 else "❄️" if momentum < -5 else "➡️"
                    
                    rsi_text = ""
                    if len(hist) >= 14:
                        rsi = TechnicalAnalysis.calculate_rsi(hist['Close']).iloc[-1]
                        signal, emoji = TechnicalAnalysis.get_rsi_signal(rsi)
                        rsi_text = f"{emoji} RSI: {rsi:.1f}"
                    
                    st.metric(
                        label=f"{ticker} {momentum_emoji}",
                        value=f"${current:.2f}",
                        delta=f"{change:+.2f}%",
                        help=rsi_text
                    )
                    
                    if len(hist) > 5:
                        fig = go.Figure(go.Scatter(
                            x=hist.index[-20:],
                            y=hist['Close'].iloc[-20:],
                            mode='lines',
                            line=dict(color='green' if change >= 0 else 'red', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0,255,0,0.1)' if change >= 0 else 'rgba(255,0,0,0.1)'
                        ))
                        fig.update_layout(
                            height=80,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            xaxis_visible=False,
                            yaxis_visible=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            except:
                st.metric(ticker, "N/A")
    
    st.divider()
    
    # Watchlist table
    st.subheader("📋 Full Watchlist")
    
    table_data = []
    for ticker in watchlist[:15]:
        if ticker in all_data:
            try:
                data = all_data[ticker]
                info = data['info']
                hist = data['hist']
                
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    volume = hist['Volume'].iloc[-1]
                    
                    if len(hist) >= 14:
                        rsi = TechnicalAnalysis.calculate_rsi(hist['Close']).iloc[-1]
                        rsi_signal, rsi_emoji = TechnicalAnalysis.get_rsi_signal(rsi)
                        rsi_display = f"{rsi_emoji} {rsi:.1f}"
                    else:
                        rsi_display = "N/A"
                    
                    news, _ = DataFetcher.fetch_news(ticker, limit=3)
                    sentiment, sent_emoji, _ = DataFetcher.analyze_sentiment(news)
                    
                    table_data.append({
                        'Ticker': ticker,
                        'Price': f"${current:.2f}",
                        'Change': f"{change:+.2f}%",
                        'RSI': rsi_display,
                        'Sentiment': f"{sent_emoji} {sentiment}",
                        'Volume': f"{volume:,.0f}"
                    })
            except:
                continue
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        def color_change(val):
            if '+' in str(val):
                return 'color: #00ff00; font-weight: bold'
            elif '-' in str(val):
                return 'color: #ff4b4b; font-weight: bold'
            return ''
        
        styled_df = df.style.applymap(color_change, subset=['Change'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

def render_deep_analysis():
    """Enhanced analysis with more indicators"""
    st.title("📊 Deep Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5)
    
    if not ticker:
        st.info("Enter a ticker to begin")
        return
    
    with st.spinner(f"Analyzing {ticker}..."):
        stock, info, hist, error = DataFetcher.fetch_stock_data(ticker, timeframe)
    
    if error:
        st.error(f"❌ {error}")
        return
    
    st.header(f"{safe_get(info, 'longName', default=ticker)} ({ticker})")
    st.caption(f"Sector: {safe_get(info, 'sector')} | Industry: {safe_get(info, 'industry')}")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        price = safe_get(info, 'currentPrice', formatter=lambda x: f"${x:.2f}")
        st.metric("Price", price)
    
    with col2:
        change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
        st.metric("Change", f"{change:+.2f}%")
    
    with col3:
        market_cap = safe_get(info, 'marketCap', formatter=format_large_number)
        st.metric("Market Cap", market_cap)
    
    with col4:
        pe = safe_get(info, 'trailingPE', formatter=lambda x: f"{x:.2f}")
        st.metric("P/E", pe)
    
    with col5:
        beta = safe_get(info, 'beta', formatter=lambda x: f"{x:.2f}")
        st.metric("Beta", beta)
    
    st.divider()
    
    # Chart with indicators
    st.subheader("📈 Price Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Price'
    ))
    
    if len(hist) > 50:
        hist['SMA_20'] = TechnicalAnalysis.calculate_simple_moving_average(hist['Close'], 20)
        hist['SMA_50'] = TechnicalAnalysis.calculate_simple_moving_average(hist['Close'], 50)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA_20'],
            line=dict(color='orange', width=1),
            name='SMA 20'
        ))
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA_50'],
            line=dict(color='blue', width=1),
            name='SMA 50'
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    st.subheader("📊 Technical Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if len(hist) >= 14:
            rsi = TechnicalAnalysis.calculate_rsi(hist['Close']).iloc[-1]
            signal, emoji = TechnicalAnalysis.get_rsi_signal(rsi)
            st.metric("RSI (14)", f"{rsi:.1f}")
            st.caption(f"{emoji} {signal}")
    
    with col2:
        if len(hist) >= 26:
            macd, signal_line = TechnicalAnalysis.calculate_macd(hist['Close'])
            macd_val = macd.iloc[-1]
            trend = "Bullish 🟢" if macd_val > 0 else "Bearish 🔴"
            st.metric("MACD", f"{macd_val:.2f}")
            st.caption(trend)
    
    with col3:
        if len(hist) >= 20:
            sma, upper, lower = TechnicalAnalysis.calculate_bollinger_bands(hist['Close'])
            current = hist['Close'].iloc[-1]
            if current > upper.iloc[-1]:
                position = "Above 🔴"
            elif current < lower.iloc[-1]:
                position = "Below 🟢"
            else:
                position = "Within 🟡"
            st.metric("BB Position", position)
    
    with col4:
        volume_today = hist['Volume'].iloc[-1] if len(hist) > 0 else 0
        volume_avg = hist['Volume'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else volume_today
        volume_ratio = volume_today / volume_avg if volume_avg > 0 else 1
        st.metric("Volume", f"{volume_today:,.0f}")
        st.caption(f"{volume_ratio:.1f}x avg")
    
    # News
    st.divider()
    st.subheader("📰 News & Sentiment")
    
    news, _ = DataFetcher.fetch_news(ticker, limit=5)
    
    if news:
        sentiment, emoji, score = DataFetcher.analyze_sentiment(news)
        st.info(f"**Sentiment:** {emoji} {sentiment} (Score: {score:.2f})")
        
        for item in news[:3]:
            with st.expander(f"📰 {item.get('title', 'No title')}"):
                st.write(f"**Publisher:** {item.get('publisher', 'Unknown')}")
                if 'link' in item:
                    st.write(f"[Read more]({item['link']})")

def render_portfolio():
    """Enhanced portfolio with tax reporting"""
    st.title("💼 Portfolio Management")
    
    # Add transaction
    with st.expander("➕ Add Transaction", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trans_ticker = st.text_input("Ticker", key="trans_ticker").upper()
            trans_type = st.selectbox("Type", ["BUY", "SELL"], key="trans_type")
        
        with col2:
            trans_shares = st.number_input("Shares", min_value=0.0, step=1.0, key="trans_shares")
            trans_price = st.number_input("Price ($)", min_value=0.0, step=0.01, key="trans_price")
        
        with col3:
            trans_date = st.date_input("Date", value=datetime.now().date(), key="trans_date")
            st.write("")
            
            if st.button("✅ Add", use_container_width=True):
                success, message = PortfolioManager.add_transaction(
                    trans_ticker, trans_shares, trans_price, trans_type, trans_date
                )
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    # Portfolio summary
    if st.session_state.portfolio:
        total_value, total_cost, total_gain = PortfolioManager.get_portfolio_value()
        realized_gains = PortfolioManager.calculate_realized_gains()
        gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Portfolio Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col3:
            st.metric("Unrealized P/L", f"${total_gain:,.2f}", f"{gain_pct:+.2f}%")
        with col4:
            st.metric("Realized P/L", f"${realized_gains:,.2f}")
        with col5:
            st.metric("Holdings", len(st.session_state.portfolio))
        
        st.divider()
        
        # Holdings
        st.subheader("📊 Current Holdings")
        
        holdings_data = []
        for holding in st.session_state.portfolio:
            try:
                _, info, _, _ = DataFetcher.fetch_stock_data(holding['ticker'], period='1d')
                if not info:
                    continue
                
                current_price = info.get('currentPrice', 0)
                current_value = current_price * holding['shares']
                avg_cost = holding['cost_basis'] / holding['shares'] if holding['shares'] > 0 else 0
                gain_loss = current_value - holding['cost_basis']
                gain_loss_pct = (gain_loss / holding['cost_basis'] * 100) if holding['cost_basis'] > 0 else 0
                
                holdings_data.append({
                    'Ticker': holding['ticker'],
                    'Shares': f"{holding['shares']:.2f}",
                    'Avg Cost': f"${avg_cost:.2f}",
                    'Current': f"${current_price:.2f}",
                    'Value': f"${current_value:,.2f}",
                    'P/L': f"${gain_loss:,.2f}",
                    'Return': f"{gain_loss_pct:+.2f}%"
                })
            except:
                continue
        
        if holdings_data:
            df = pd.DataFrame(holdings_data)
            
            def color_return(val):
                if '+' in str(val):
                    return 'color: #00ff00; font-weight: bold'
                elif '-' in str(val):
                    return 'color: #ff4b4b; font-weight: bold'
                return ''
            
            styled = df.style.applymap(color_return, subset=['Return'])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            
            # Tax export
            st.divider()
            st.subheader("📄 Tax Reporting")
            
            if st.button("📥 Generate Tax Report", use_container_width=False):
                tax_df = PortfolioManager.export_for_taxes()
                if not tax_df.empty:
                    st.dataframe(tax_df, use_container_width=True)
                    
                    csv = tax_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Tax Report CSV",
                        csv,
                        f"tax_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
                else:
                    st.info("No realized gains/losses to report")
    else:
        st.info("Portfolio empty. Add your first transaction!")

def render_correlation():
    """Correlation analysis page"""
    st.title("📊 Correlation Matrix")
    st.caption("Analyze how your holdings move together")
    
    # Ticker selection
    if st.session_state.portfolio:
        portfolio_tickers = [h['ticker'] for h in st.session_state.portfolio]
        default_tickers = ", ".join(portfolio_tickers)
    else:
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA"
    
    tickers_input = st.text_input(
        "Enter tickers (comma-separated)",
        value=default_tickers
    )
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if len(tickers) >= 2:
        if st.button("📊 Calculate Correlation", type="primary"):
            CorrelationAnalyzer.render_correlation_heatmap(tickers)
    else:
        st.warning("Enter at least 2 tickers")

def render_backtest():
    """Backtesting page"""
    st.title("⏮️ Strategy Backtester")
    st.caption("Test your trading strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Ticker to test", value="AAPL").upper()
        short_ma = st.number_input("Short MA", value=20, min_value=5, max_value=100)
    
    with col2:
        initial_capital = st.number_input("Initial Capital ($)", value=10000.0, step=1000.0)
        long_ma = st.number_input("Long MA", value=50, min_value=10, max_value=200)
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            results = BacktestEngine.simple_ma_crossover(
                ticker, short_ma, long_ma, initial_capital
            )
        
        if results:
            st.success("✅ Backtest complete!")
            
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Initial Capital", f"${results['initial_capital']:,.2f}")
            with col2:
                st.metric("Final Value", f"${results['final_value']:,.2f}")
            with col3:
                delta_color = "normal" if results['total_return'] >= 0 else "inverse"
                st.metric("Strategy Return", f"{results['total_return']:+.2f}%")
            with col4:
                st.metric("Buy & Hold", f"{results['buy_hold_return']:+.2f}%")
            
            st.divider()
            
            # Performance comparison
            st.subheader("📊 Strategy Performance")
            
            outperformance = results['total_return'] - results['buy_hold_return']
            
            if outperformance > 0:
                st.success(f"✅ Strategy outperformed buy & hold by {outperformance:.2f}%")
            else:
                st.warning(f"⚠️ Strategy underperformed buy & hold by {abs(outperformance):.2f}%")
            
            st.write(f"**Number of trades:** {results['num_trades']}")
            
            # Trade log
            if results['trades']:
                st.divider()
                st.subheader("📜 Trade Log")
                
                trades_df = pd.DataFrame(results['trades'])
                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            # Chart with signals
            st.divider()
            st.subheader("📈 Backtest Chart")
            
            hist = results['hist']
            
            fig = go.Figure()
            
            # Price
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Price',
                line=dict(color='white', width=2)
            ))
            
            # Moving averages
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['SMA_short'],
                mode='lines',
                name=f'SMA {short_ma}',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['SMA_long'],
                mode='lines',
                name=f'SMA {long_ma}',
                line=dict(color='blue', width=1)
            ))
            
            # Buy/Sell signals
            buy_signals = hist[hist['Position'] == 1]
            sell_signals = hist[hist['Position'] == -1]
            
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=600,
                hovermode='x unified',
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Backtest failed. Check ticker and try again.")

def render_screener():
    """Enhanced stock screener"""
    st.title("🔎 Stock Screener")
    st.caption("Find stocks matching your criteria")
    
    # Filters
    st.sidebar.subheader("🎯 Filters")
    
    st.sidebar.write("**Price Range**")
    min_price = st.sidebar.number_input("Min Price ($)", value=0.0, step=10.0)
    max_price = st.sidebar.number_input("Max Price ($)", value=5000.0, step=10.0)
    
    st.sidebar.write("**Valuation**")
    min_pe = st.sidebar.number_input("Min P/E", value=0.0, step=1.0)
    max_pe = st.sidebar.number_input("Max P/E", value=100.0, step=5.0)
    
    st.sidebar.write("**Market Cap (B)**")
    min_mcap = st.sidebar.number_input("Min MCap", value=0.0, step=10.0)
    max_mcap = st.sidebar.number_input("Max MCap", value=3000.0, step=50.0)
    
    st.sidebar.write("**Technical**")
    min_rsi = st.sidebar.slider("Min RSI", 0, 100, 0)
    max_rsi = st.sidebar.slider("Max RSI", 0, 100, 100)
    
    # Quick presets
    st.subheader("⚡ Quick Presets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    preset = None
    
    with col1:
        if st.button("💎 Value", use_container_width=True):
            preset = {'min_price': 0, 'max_price': 5000, 'min_pe': 0, 'max_pe': 15,
                     'min_mcap': 10, 'max_mcap': 3000, 'min_rsi': 0, 'max_rsi': 100}
    
    with col2:
        if st.button("🚀 Growth", use_container_width=True):
            preset = {'min_price': 50, 'max_price': 5000, 'min_pe': 20, 'max_pe': 100,
                     'min_mcap': 50, 'max_mcap': 3000, 'min_rsi': 50, 'max_rsi': 80}
    
    with col3:
        if st.button("🟢 Oversold", use_container_width=True):
            preset = {'min_price': 0, 'max_price': 5000, 'min_pe': 0, 'max_pe': 100,
                     'min_mcap': 0, 'max_mcap': 3000, 'min_rsi': 0, 'max_rsi': 30}
    
    with col4:
        if st.button("🔴 Overbought", use_container_width=True):
            preset = {'min_price': 0, 'max_price': 5000, 'min_pe': 0, 'max_pe': 100,
                     'min_mcap': 0, 'max_mcap': 3000, 'min_rsi': 70, 'max_rsi': 100}
    
    filters = preset or {
        'min_price': min_price, 'max_price': max_price,
        'min_pe': min_pe, 'max_pe': max_pe,
        'min_mcap': min_mcap, 'max_mcap': max_mcap,
        'min_rsi': min_rsi, 'max_rsi': max_rsi
    }
    
    st.divider()
    
    # Run screener
    if st.button("🔍 Run Screener", type="primary", use_container_width=True):
        results = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for idx, ticker in enumerate(SCREENER_UNIVERSE):
            status.text(f"Screening {ticker}... ({idx+1}/{len(SCREENER_UNIVERSE)})")
            progress.progress((idx + 1) / len(SCREENER_UNIVERSE))
            
            try:
                _, info, hist, error = DataFetcher.fetch_stock_data(ticker, period='1mo')
                
                if error or not info or hist is None or hist.empty:
                    continue
                
                price = info.get('currentPrice', 0)
                pe = info.get('trailingPE', 0)
                mcap = info.get('marketCap', 0) / 1e9
                div_yield = (info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0
                
                if not (filters['min_price'] <= price <= filters['max_price']):
                    continue
                if pe and not (filters['min_pe'] <= pe <= filters['max_pe']):
                    continue
                if not (filters['min_mcap'] <= mcap <= filters['max_mcap']):
                    continue
                
                if len(hist) >= Config().RSI_WINDOW:
                    rsi = TechnicalAnalysis.calculate_rsi(hist['Close']).iloc[-1]
                    if not (filters['min_rsi'] <= rsi <= filters['max_rsi']):
                        continue
                else:
                    rsi = None
                
                results.append({
                    'Ticker': ticker,
                    'Price': f"${price:.2f}",
                    'P/E': f"{pe:.2f}" if pe else "N/A",
                    'MCap': f"${mcap:.1f}B",
                    'RSI': f"{rsi:.1f}" if rsi else "N/A",
                    'Div': f"{div_yield:.2f}%",
                    'Sector': info.get('sector', 'N/A')[:20]
                })
                
            except:
                continue
        
        progress.empty()
        status.empty()
        
        if results:
            st.success(f"✅ Found {len(results)} stocks")
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export",
                csv,
                f"screener_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.warning("⚠️ No matches. Adjust filters.")

def render_compare():
    """Enhanced comparison"""
    st.title("⚖️ Stock Comparison")
    
    compare_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL"
    ).upper()
    
    tickers = [t.strip() for t in compare_input.split(",") if t.strip()]
    
    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers")
        return
    
    if len(tickers) > 5:
        st.warning("Max 5 tickers")
        tickers = tickers[:5]
    
    if st.button("📊 Compare", type="primary"):
        comparison = []
        
        for ticker in tickers:
            try:
                _, info, hist, error = DataFetcher.fetch_stock_data(ticker, period='1y')
                
                if error or not info or hist is None or hist.empty:
                    continue
                
                year_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / 
                              hist['Close'].iloc[0]) * 100 if len(hist) > 0 else 0
                
                comparison.append({
                    'Ticker': ticker,
                    'Price': info.get('currentPrice', 0),
                    'P/E': info.get('trailingPE', 0),
                    'MCap': info.get('marketCap', 0) / 1e9,
                    'Div': (info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                    '1Y': year_return,
                    'Beta': info.get('beta', 0),
                    'Sector': info.get('sector', 'N/A')
                })
                
            except:
                continue
        
        if not comparison:
            st.error("❌ No data")
            return
        
        df = pd.DataFrame(comparison)
        
        # Display
        st.subheader("📊 Fundamentals")
        
        display = df.copy()
        display['Price'] = display['Price'].apply(lambda x: f"${x:.2f}")
        display['P/E'] = display['P/E'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
        display['MCap'] = display['MCap'].apply(lambda x: f"${x:.1f}B")
        display['Div'] = display['Div'].apply(lambda x: f"{x:.2f}%")
        display['1Y'] = display['1Y'].apply(lambda x: f"{x:+.2f}%")
        display['Beta'] = display['Beta'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
        
        st.dataframe(display, use_container_width=True, hide_index=True)
        
        # Performance chart
        st.divider()
        st.subheader("📈 Performance (1Y)")
        
        fig = go.Figure()
        
        for ticker in tickers:
            try:
                _, _, hist, _ = DataFetcher.fetch_stock_data(ticker, period='1y')
                if hist is not None and not hist.empty:
                    normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode='lines',
                        name=ticker,
                        line=dict(width=2)
                    ))
            except:
                continue
        
        fig.update_layout(
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.divider()
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        best = df.loc[df['1Y'].idxmax()]
        lowest_pe = df[df['P/E'] > 0].loc[df[df['P/E'] > 0]['P/E'].idxmin()] if len(df[df['P/E'] > 0]) > 0 else None
        highest_div = df.loc[df['Div'].idxmax()]
        
        with col1:
            st.success(f"""
            **🏆 Best Performer**
            
            {best['Ticker']}
            
            Return: **{best['1Y']:+.2f}%**
            """)
        
        with col2:
            if lowest_pe is not None:
                st.info(f"""
                **💎 Lowest P/E**
                
                {lowest_pe['Ticker']}
                
                P/E: **{lowest_pe['P/E']:.2f}**
                """)
        
        with col3:
            st.warning(f"""
            **💰 Highest Div**
            
            {highest_div['Ticker']}
            
            Yield: **{highest_div['Div']:.2f}%**
            """)

def render_position_sizer():
    """Position sizing calculator"""
    st.title("🧮 Position Sizer")
    st.caption("Calculate optimal position size")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Account")
        account = st.number_input("Size ($)", value=10000.0, min_value=100.0, step=1000.0)
        risk_pct = st.slider("Risk (%)", 0.5, 5.0, 1.0, 0.1)
    
    with col2:
        st.subheader("Trade")
        entry = st.number_input("Entry ($)", value=100.0, min_value=0.01, step=1.0)
        stop = st.number_input("Stop Loss ($)", value=95.0, min_value=0.01, step=1.0)
        target = st.number_input("Target ($)", value=110.0, min_value=0.01, step=1.0)
    
    # Validate
    if stop >= entry:
        st.error("❌ Stop loss must be below entry")
        return
    
    # Calculate
    risk_amt = account * (risk_pct / 100)
    risk_per_share = entry - stop
    shares = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
    position_val = shares * entry
    position_pct = (position_val / account) * 100
    
    profit_per_share = target - entry
    potential_profit = profit_per_share * shares
    potential_return = (profit_per_share / entry) * 100
    
    potential_loss = risk_per_share * shares
    risk_reward = profit_per_share / risk_per_share if risk_per_share > 0 else 0
    
    st.divider()
    st.subheader("📊 Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Shares", f"{shares:,}")
        st.metric("Position", f"${position_val:,.2f}")
        st.metric("% of Account", f"{position_pct:.1f}%")
    
    with col2:
        st.metric("Risk", f"${risk_amt:,.2f}")
        st.metric("Profit Target", f"${potential_profit:,.2f}")
        st.metric("Max Loss", f"${potential_loss:,.2f}")
    
    with col3:
        st.metric("Risk/Reward", f"{risk_reward:.2f}:1")
        st.metric("Return", f"{potential_return:+.2f}%")
        st.metric("Stop %", f"{abs((stop - entry)/entry*100):.2f}%")
    
    # Assessment
    st.divider()
    st.subheader("🎯 Assessment")
    
    if risk_reward > 2:
        st.success("✅ Excellent R/R (>2:1)")
    elif risk_reward > 1:
        st.info("⚠️ Acceptable R/R (1-2:1)")
    else:
        st.error("❌ Poor R/R (<1:1)")
    
    if position_pct > 20:
        st.error(f"⚠️ Position too large ({position_pct:.1f}% > 20%)")
    elif position_pct > 10:
        st.warning(f"⚠️ Moderate position ({position_pct:.1f}%)")
    else:
        st.success(f"✅ Conservative position ({position_pct:.1f}%)")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    if not AuthManager.check_password():
        return
    
    # Initialize
    PortfolioManager.initialize()
    AlertsManager.load_alerts()
    
    # Check alerts
    triggered = AlertsManager.check_alerts()
    if triggered:
        logger.info(f"Alerts triggered: {triggered}")
    
    # Sidebar
    st.sidebar.title("📈 Trading Platform v4.0")
    st.sidebar.caption(f"Welcome, {st.session_state.username}")
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.divider()
    
    # Navigation
    app_mode = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "📊 Deep Analysis",
            "💼 Portfolio",
            "🔍 Screener",
            "⚖️ Compare",
            "📉 Correlation",
            "⏮️ Backtest",
            "🧮 Position Sizer"
        ],
        label_visibility="collapsed"
    )
    
    # Utilities
    quick_tips()
    
    st.sidebar.divider()
    
    # Settings
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        watchlist_input = st.text_area(
            "Watchlist",
            value=", ".join(Config().DEFAULT_WATCHLIST),
            height=100
        )
        custom_watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    
    # Logout
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        AuthManager.logout()
    
    # Route
    watchlist = custom_watchlist or Config().DEFAULT_WATCHLIST
    
    if app_mode == "🏠 Dashboard":
        render_dashboard(watchlist)
    elif app_mode == "📊 Deep Analysis":
        render_deep_analysis()
    elif app_mode == "💼 Portfolio":
        render_portfolio()
    elif app_mode == "🔍 Screener":
        render_screener()
    elif app_mode == "⚖️ Compare":
        render_compare()
    elif app_mode == "📉 Correlation":
        render_correlation()
    elif app_mode == "⏮️ Backtest":
        render_backtest()
    elif app_mode == "🧮 Position Sizer":
        render_position_sizer()

if __name__ == "__main__":
    main()