import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="ASX Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        color: #333;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #F0F2F6;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type'])
if 'portfolio_metrics' not in st.session_state:
    st.session_state.portfolio_metrics = None
if 'last_calculation' not in st.session_state:
    st.session_state.last_calculation = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

# Helper functions
@st.cache_data
def fetch_stock_data(ticker, start_date=None):
    """Fetch stock data from Yahoo Finance"""
    try:
        # Clean up ticker and add .AX suffix for ASX stocks
        original_ticker = ticker
        ticker = ticker.replace('.ASX', '').replace('.AX', '').upper()
        
        # Add .AX suffix if not present
        if not ticker.endswith('.AX'):
            ticker = ticker + '.AX'
        
        stock = yf.Ticker(ticker)
        if start_date:
            data = stock.history(start=start_date)
        else:
            data = stock.history(period="1d")
        
        return data if not data.empty else None
    except:
        return None

def calculate_portfolio_metrics(portfolio_df):
    """Calculate portfolio metrics for each holding"""
    if portfolio_df.empty:
        return pd.DataFrame()
    
    # Calculate net holdings per stock
    holdings = {}
    for idx, row in portfolio_df.iterrows():
        ticker = row['ASX_Code']
        shares = row['Shares']
        
        if ticker not in holdings:
            holdings[ticker] = {'total_shares': 0, 'total_cost': 0, 'purchase_dates': []}
        
        if row['Transaction_Type'] == 'Buy':
            holdings[ticker]['total_shares'] += shares
            holdings[ticker]['total_cost'] += shares * row['Purchase_Price']
            holdings[ticker]['purchase_dates'].append(row['Purchase_Date'])
        else:  # Sell
            holdings[ticker]['total_shares'] -= shares
    
    results = []
    failed_tickers = []
    
    for ticker, data in holdings.items():
        if data['total_shares'] <= 0:  # Skip if no shares remaining
            continue
            
        avg_price = data['total_cost'] / data['total_shares'] if data['total_shares'] > 0 else 0
        earliest_date = min(data['purchase_dates'])
        
        # Fetch historical data
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is not None and not stock_data.empty:
            current_price = stock_data['Close'].iloc[-1]
            shares = data['total_shares']
            
            # Calculate metrics
            current_value = current_price * shares
            purchase_value = avg_price * shares
            total_return = current_value - purchase_value
            pct_return = (total_return / purchase_value) * 100 if purchase_value > 0 else 0
            
            # Simplified dividend calculation
            dividends_received = 0
            
            results.append({
                'ASX_Code': ticker,
                'Purchase_Date': earliest_date,
                'Shares': shares,
                'Purchase_Price': avg_price,
                'Current_Price': current_price,
                'Purchase_Value': purchase_value,
                'Current_Value': current_value,
                'Total_Return': total_return,
                'Pct_Return': pct_return,
                'Dividends_Received': dividends_received
            })
        else:
            failed_tickers.append(ticker)
    
    if failed_tickers:
        st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")
    
    return pd.DataFrame(results)

# Main app
st.title("🇦🇺 ASX Portfolio Tracker")
st.markdown("Track your Australian stock investments with real-time data and analytics")

# Sidebar for portfolio management
with st.sidebar:
    st.header("Portfolio Management")
    
    # Manual transaction entry
    st.subheader("Add New Transaction")
    
    ticker = st.text_input("ASX Code", placeholder="e.g., BHP")
    col1, col2 = st.columns(2)
    with col1:
        shares = st.number_input("Shares", min_value=1, value=100)
        purchase_date = st.date_input("Purchase Date")
    with col2:
        purchase_price = st.number_input("Price (AUD)", min_value=0.01, value=50.00)
        transaction_type = st.selectbox("Type", ["Buy", "Sell"])
    
    if st.button("Add Transaction"):
        if ticker:
            new_transaction = pd.DataFrame([{
                'ASX_Code': ticker,
                'Purchase_Date': purchase_date,
                'Shares': shares,
                'Purchase_Price': purchase_price,
                'Transaction_Type': transaction_type
            }])
            
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_transaction], ignore_index=True)
            st.session_state.portfolio_metrics = None
            st.success(f"Added {transaction_type} transaction for {ticker}!")
    
    st.divider()
    
    # Import/Export
    st.subheader("Import/Export")
    
    # Export portfolio
    if not st.session_state.portfolio.empty:
        csv = st.session_state.portfolio.to_csv(index=False)
        st.download_button(
            label="Export Portfolio (CSV)",
            data=csv,
            file_name="asx_portfolio.csv",
            mime="text/csv"
        )
    
    # Import portfolio
    st.info("**Supported formats:**\n- Standard: ASX_Code, Purchase_Date, Shares, Purchase_Price\n- CommSec: Date, Code, Quantity, Avg. price, Action")
    uploaded_file = st.file_uploader("Import Portfolio (CSV)", type="csv", key="file_uploader")
    
    if uploaded_file is not None and not st.session_state.file_processed:
        if st.button("Process CSV File"):
            try:
                imported_df = pd.read_csv(uploaded_file)
                
                # Check for CommSec format
                column_mapping = {
                    'Date': 'Purchase_Date',
                    'Code': 'ASX_Code',
                    'Quantity': 'Shares',
                    'Avg. price': 'Purchase_Price'
                }
                
                commsec_columns = ['Date', 'Code', 'Quantity', 'Avg. price', 'Action']
                if all(col in imported_df.columns for col in commsec_columns):
                    # Rename columns
                    imported_df = imported_df.rename(columns=column_mapping)
                    imported_df['Transaction_Type'] = imported_df['Action']
                    
                    # Clean ASX codes
                    imported_df['ASX_Code'] = imported_df['ASX_Code'].str.replace('.ASX', '', regex=False)
                    
                    # Convert date
                    try:
                        imported_df['Purchase_Date'] = pd.to_datetime(imported_df['Purchase_Date'], format='%d/%m/%Y')
                    except:
                        imported_df['Purchase_Date'] = pd.to_datetime(imported_df['Purchase_Date'], format='mixed', dayfirst=True)
                    
                    # Keep only required columns
                    imported_df = imported_df[['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type']]
                    
                    st.session_state.portfolio = imported_df
                    st.session_state.portfolio_metrics = None
                    st.session_state.file_processed = True
                    st.success("Portfolio imported successfully!")
                else:
                    st.error("CSV format not recognized. Please check the required columns.")
            except Exception as e:
                st.error(f"Error importing file: {str(e)}")
    
    # Reset button
    if st.button("Reset All Data"):
        st.session_state.portfolio = pd.DataFrame(columns=['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type'])
        st.session_state.portfolio_metrics = None
        st.session_state.file_processed = False
        st.success("All data reset!")

# Main content
if st.session_state.portfolio.empty:
    st.info("👋 Welcome! Add your first holding using the sidebar to get started.")
else:
    # Calculate button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Calculate Portfolio Metrics", type="primary"):
            with st.spinner('Calculating portfolio metrics...'):
                st.session_state.portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio)
    
    # Display metrics if calculated
    if st.session_state.portfolio_metrics is not None:
        portfolio_metrics = st.session_state.portfolio_metrics
        
        if not portfolio_metrics.empty:
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = portfolio_metrics['Current_Value'].sum()
                st.metric("Portfolio Value", f"${total_value:,.2f}")
            
            with col2:
                total_return = portfolio_metrics['Total_Return'].sum()
                st.metric("Total Return", f"${total_return:,.2f}")
            
            with col3:
                total_dividends = portfolio_metrics['Dividends_Received'].sum()
                st.metric("Dividends Received", f"${total_dividends:,.2f}")
            
            with col4:
                total_cost = portfolio_metrics['Purchase_Value'].sum()
                total_return_pct = (total_return / total_cost) * 100 if total_cost > 0 else 0
                st.metric("Total Return %", f"{total_return_pct:.2f}%")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Holdings", "📈 Performance", "🥧 Allocation", "📋 Transactions"])
            
            with tab1:
                st.subheader("Current Holdings")
                
                display_df = portfolio_metrics.copy()
                for col in ['Purchase_Price', 'Current_Price', 'Current_Value', 'Total_Return']:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
                display_df['Pct_Return'] = display_df['Pct_Return'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with tab2:
                st.subheader("Stock Returns")
                
                fig_returns = px.bar(
                    portfolio_metrics,
                    x='ASX_Code',
                    y='Pct_Return',
                    color='Pct_Return',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title='Stock Returns (%)'
                )
                
                fig_returns.update_layout(height=400)
                st.plotly_chart(fig_returns, use_container_width=True)
            
            with tab3:
                st.subheader("Portfolio Allocation")
                
                fig_allocation = px.pie(
                    portfolio_metrics,
                    values='Current_Value',
                    names='ASX_Code',
                    title='Portfolio Allocation by Value',
                    hole=0.4
                )
                
                fig_allocation.update_traces(textposition='inside', textinfo='percent+label')
                fig_allocation.update_layout(height=600)
                
                st.plotly_chart(fig_allocation, use_container_width=True)
            
            with tab4:
                st.subheader("Transaction History")
                
                transactions_df = st.session_state.portfolio.copy()
                transactions_df['Purchase_Date'] = transactions_df['Purchase_Date'].dt.strftime('%Y-%m-%d')
                transactions_df['Total_Cost'] = transactions_df['Shares'] * transactions_df['Purchase_Price']
                
                # Format currency
                transactions_df['Purchase_Price'] = transactions_df['Purchase_Price'].apply(lambda x: f"${x:.2f}")
                transactions_df['Total_Cost'] = transactions_df['Total_Cost'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(transactions_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No valid holdings found in your portfolio.")
    else:
        st.info("Click 'Calculate Portfolio Metrics' to analyze your portfolio.")

# Footer
st.markdown("---")
st.markdown("Created in May 2025 by Thomas Delafosse. v1.01 draft version only.")
