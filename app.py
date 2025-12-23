import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# 1. CONFIGURATION
class Config:
    DEFAULT_TICKER = "AAPL"
    DEFAULT_WATCHLIST = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"]
    CACHE_TTL = 60

# 2. DATA FETCHER
class DataFetcher:
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_stock_data(ticker, period='1y'):
        try:
            stock = yf.Ticker(ticker)
            # We fetch the data here, but we do NOT return the 'stock' object
            hist = stock.history(period=period)
            info = stock.info
            # Return ONLY info, hist, and None (for error)
            return info, hist, None 
        except Exception as e:
            return None, None, str(e)

# 3. PORTFOLIO MANAGER (This fixes your NameError)
class PortfolioManager:
    @staticmethod
    def initialize():
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []

# 4. ALERTS MANAGER
class AlertsManager:
    @staticmethod
    def load_alerts():
        if 'price_alerts' not in st.session_state:
            st.session_state.price_alerts = {}

# 5. PAGE RENDERERS (Simplifying for testing)
def render_dashboard(watchlist):
    st.title("🏠 Trading Dashboard")
    st.write(f"Monitoring: {', '.join(watchlist)}")
    for ticker in watchlist:
        # We removed the first "_" because we are no longer returning the ticker object
        info, hist, error = DataFetcher.fetch_stock_data(ticker)
        
        if error:
            st.error(f"Error loading {ticker}: {error}")
            continue
            
        if info and 'currentPrice' in info:
            st.metric(ticker, f"${info['currentPrice']:.2f}")

# 6. MAIN APPLICATION
def main():
    # Force logged in state
    st.session_state.authenticated = True
    st.session_state.username = "Admin"
    
    # Initialize components
    PortfolioManager.initialize()
    AlertsManager.load_alerts()
    
    st.sidebar.title("📈 Trading Platform v4.0")
    app_mode = st.sidebar.radio("Navigate", ["🏠 Dashboard", "💼 Portfolio"])
    
    watchlist = Config.DEFAULT_WATCHLIST
    
    if app_mode == "🏠 Dashboard":
        render_dashboard(watchlist)
    else:
        st.write("Portfolio Page Coming Soon")

if __name__ == "__main__":
    main()