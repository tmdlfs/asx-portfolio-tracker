import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import logging

# --- Page Config and Basic Setup ---
st.set_page_config(
    page_title="ASX Portfolio Tracker Pro",
    page_icon="💰",
    layout="wide"
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_UNIQUE_STOCKS = 100

# Custom CSS
st.markdown("""
<style>
    /* ... (your existing CSS can go here if needed, or be simplified) ... */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px; font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background-color: #F0F2F6; border-radius: 8px; padding: 15px; margin: 5px;
    }
    .centered-button { text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def initialize_session_state():
    defaults = {
        'portfolio': pd.DataFrame(columns=['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type', 'Fees', 'Notes']),
        'portfolio_metrics': None,
        'file_processed': False,
        'calculation_messages': [],
        'historical_portfolio_performance': None,
        'selected_stock_for_chart': None,
        'edited_transactions_df': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Helper & Caching Functions ---
@st.cache_data(ttl=3600)
def get_stock_info(ticker_ax):
    try:
        stock = yf.Ticker(ticker_ax)
        info = stock.info
        return {
            'longName': info.get('longName', ticker_ax),
            'currency': info.get('currency', 'AUD')
        }
    except Exception as e:
        logger.error(f"Failed to get stock info for {ticker_ax}: {e}")
        # Add message to session state to be displayed in UI
        if 'calculation_messages' in st.session_state:
            st.session_state.calculation_messages.append(f"Error: Could not get company info for {ticker_ax}.")
        return {'longName': ticker_ax, 'currency': 'AUD'}


@st.cache_data(ttl=3600)
def fetch_stock_price_history(ticker_ax, period="max", start_date=None, end_date=None):
    logger.info(f"Fetching price history for {ticker_ax} ({period if period else str(start_date)+' to '+str(end_date)})")
    try:
        stock = yf.Ticker(ticker_ax)
        if start_date and end_date:
            history = stock.history(start=start_date, end=end_date)
        else:
            history = stock.history(period=period)
        if history.empty:
            if 'calculation_messages' in st.session_state:
                 st.session_state.calculation_messages.append(f"Warning: No price history found for {ticker_ax} in the requested period.")
        return history if not history.empty else None
    except Exception as e:
        if 'calculation_messages' in st.session_state:
            st.session_state.calculation_messages.append(f"Error: Could not fetch price history for {ticker_ax}: {e}")
        logger.error(f"fetch_stock_price_history error for {ticker_ax}: {e}")
        return None

@st.cache_data(ttl=86400)
def fetch_dividend_history(ticker_ax):
    logger.info(f"Fetching dividend history for {ticker_ax}")
    try:
        stock = yf.Ticker(ticker_ax)
        dividends = stock.dividends
        if dividends.empty:
            logger.info(f"No dividend history found for {ticker_ax}.")
        return dividends if not dividends.empty else None
    except Exception as e:
        if 'calculation_messages' in st.session_state:
            st.session_state.calculation_messages.append(f"Error: Could not fetch dividend history for {ticker_ax}: {e}")
        logger.error(f"fetch_dividend_history error for {ticker_ax}: {e}")
        return None

def get_price_on_date(price_history_df, target_date):
    if price_history_df is None or price_history_df.empty:
        return 0
    # Ensure target_date is datetime and tz-naive for comparison
    target_date_dt = pd.to_datetime(target_date)
    if target_date_dt.tzinfo is not None:
        target_date_dt = target_date_dt.tz_localize(None)
    
    # Ensure index is also timezone naive
    temp_price_history_df = price_history_df.copy() # Avoid modifying cached data
    if temp_price_history_df.index.tzinfo is not None:
        temp_price_history_df.index = temp_price_history_df.index.tz_localize(None)

    closest_prices = temp_price_history_df[temp_price_history_df.index <= target_date_dt]
    return closest_prices['Close'].iloc[-1] if not closest_prices.empty else 0


# --- Core Logic: Portfolio Metrics Calculation ---
def calculate_portfolio_metrics(portfolio_df):
    # Reset messages if not already done or if called directly
    if 'calculation_messages' not in st.session_state:
        st.session_state.calculation_messages = []
    elif not st.session_state.calculation_messages or (st.session_state.calculation_messages and st.session_state.calculation_messages[-1] != "Info: Starting portfolio metrics calculation..."):
        st.session_state.calculation_messages = ["Info: Starting portfolio metrics calculation..."]


    if portfolio_df.empty:
        st.session_state.calculation_messages.append("Info: Portfolio is empty. No metrics to calculate.")
        return pd.DataFrame()

    portfolio_df['Purchase_Date'] = pd.to_datetime(portfolio_df['Purchase_Date'])
    portfolio_df['Shares'] = pd.to_numeric(portfolio_df['Shares'], errors='coerce')
    portfolio_df['Purchase_Price'] = pd.to_numeric(portfolio_df['Purchase_Price'], errors='coerce')
    portfolio_df['Fees'] = pd.to_numeric(portfolio_df['Fees'], errors='coerce').fillna(0)
    portfolio_df.dropna(subset=['ASX_Code', 'Purchase_Date','Shares', 'Purchase_Price'], inplace=True)
    portfolio_df['ASX_Code'] = portfolio_df['ASX_Code'].astype(str).str.upper()

    portfolio_df = portfolio_df.sort_values(by=['ASX_Code', 'Purchase_Date', 'Transaction_Type'])

    current_holdings_details = {}
    realized_gains_losses = {}
    total_dividends_received_by_stock = {}

    unique_tickers = portfolio_df['ASX_Code'].unique()
    all_stock_price_histories = {
        ticker: fetch_stock_price_history(f"{ticker.replace('.AX', '')}.AX", period="5y")
        for ticker in unique_tickers
    }
    all_stock_dividend_histories = {
        ticker: fetch_dividend_history(f"{ticker.replace('.AX', '')}.AX")
        for ticker in unique_tickers
    }

    for ticker_orig, group in portfolio_df.groupby('ASX_Code'):
        ticker_ax = f"{ticker_orig.replace('.AX', '')}.AX"
        current_holdings_details[ticker_orig] = {'total_shares': 0.0, 'weighted_cost_sum': 0.0, 'purchase_dates': []}
        realized_gains_losses[ticker_orig] = 0.0
        total_dividends_received_by_stock[ticker_orig] = 0.0
        
        stock_dividends_df = all_stock_dividend_histories.get(ticker_orig)
        processed_transactions_for_stock = [] # Track transactions for this stock only

        for idx, row in group.iterrows():
            shares = row['Shares']
            price = row['Purchase_Price']
            fees = row['Fees']
            date = row['Purchase_Date']

            if row['Transaction_Type'] == 'Buy':
                current_holdings_details[ticker_orig]['total_shares'] += shares
                cost_of_this_transaction = (shares * price) + fees
                current_holdings_details[ticker_orig]['weighted_cost_sum'] += cost_of_this_transaction
                current_holdings_details[ticker_orig]['purchase_dates'].append(date)
                processed_transactions_for_stock.append({'date': date, 'shares_change': shares})
            
            elif row['Transaction_Type'] == 'Sell':
                if current_holdings_details[ticker_orig]['total_shares'] < 1e-6:
                    st.session_state.calculation_messages.append(f"Warning: Tried to sell {shares} of {ticker_orig} on {date.strftime('%Y-%m-%d')} with no holdings. Ignored.")
                    continue
                
                shares_to_sell = min(shares, current_holdings_details[ticker_orig]['total_shares'])
                if abs(shares_to_sell - shares) > 1e-6 :
                    st.session_state.calculation_messages.append(f"Warning: Adjusted sell of {ticker_orig} from {shares} to {shares_to_sell:.4f} due to insufficient holdings.")

                # Ensure total_shares is not zero before division
                if abs(current_holdings_details[ticker_orig]['total_shares']) < 1e-6:
                    avg_cost_per_share = 0 # Or handle as error, but this prevents division by zero
                else:
                    avg_cost_per_share = current_holdings_details[ticker_orig]['weighted_cost_sum'] / current_holdings_details[ticker_orig]['total_shares']
                
                cost_of_shares_sold = shares_to_sell * avg_cost_per_share
                proceeds_from_sale = (shares_to_sell * price) - fees
                
                realized_gains_losses[ticker_orig] += (proceeds_from_sale - cost_of_shares_sold)
                current_holdings_details[ticker_orig]['weighted_cost_sum'] -= cost_of_shares_sold
                current_holdings_details[ticker_orig]['total_shares'] -= shares_to_sell
                processed_transactions_for_stock.append({'date': date, 'shares_change': -shares_to_sell})

                if abs(current_holdings_details[ticker_orig]['total_shares']) < 1e-6 :
                    current_holdings_details[ticker_orig]['total_shares'] = 0.0
                    current_holdings_details[ticker_orig]['weighted_cost_sum'] = 0.0
                    current_holdings_details[ticker_orig]['purchase_dates'] = [] # Clear dates if all sold
        
        if stock_dividends_df is not None and not stock_dividends_df.empty:
            # Sort transactions by date to correctly calculate shares held on dividend dates
            processed_transactions_for_stock.sort(key=lambda x: x['date'])
            
            temp_holdings_df_for_div = pd.DataFrame(processed_transactions_for_stock)
            if not temp_holdings_df_for_div.empty:
                temp_holdings_df_for_div['date'] = pd.to_datetime(temp_holdings_df_for_div['date'])
                if temp_holdings_df_for_div['date'].dt.tz is not None:
                    temp_holdings_df_for_div['date'] = temp_holdings_df_for_div['date'].dt.tz_localize(None)


            for div_idx_date, div_amount in stock_dividends_df.items():
                # Ensure div_idx_date is datetime and tz-naive
                div_date_dt = pd.to_datetime(div_idx_date)
                if div_date_dt.tzinfo is not None:
                    div_date_dt = div_date_dt.tz_localize(None)
                
                shares_held_on_dividend_date_calc = 0.0
                if not temp_holdings_df_for_div.empty:
                    relevant_trans_for_div = temp_holdings_df_for_div[temp_holdings_df_for_div['date'] <= div_date_dt]
                    shares_held_on_dividend_date_calc = relevant_trans_for_div['shares_change'].sum()
                                
                if shares_held_on_dividend_date_calc > 1e-6:
                    total_dividends_received_by_stock[ticker_orig] += shares_held_on_dividend_date_calc * div_amount

    results = []
    for ticker_orig, data in current_holdings_details.items():
        current_price = 0
        stock_price_hist = all_stock_price_histories.get(ticker_orig)
        if stock_price_hist is not None and not stock_price_hist.empty:
            current_price = stock_price_hist['Close'].iloc[-1]
        elif data['total_shares'] > 1e-6 :
             st.session_state.calculation_messages.append(f"Warning: Could not fetch current price for {ticker_orig}. Using $0 for value.")

        shares_held = data['total_shares']
        cost_of_current_holdings = data['weighted_cost_sum']
        
        if shares_held > 1e-6:
            avg_purchase_price = cost_of_current_holdings / shares_held
            current_value = current_price * shares_held
            unrealized_return = current_value - cost_of_current_holdings
            pct_return = (unrealized_return / cost_of_current_holdings) * 100 if abs(cost_of_current_holdings) > 1e-6 else 0
        else:
            avg_purchase_price = 0; current_value = 0; unrealized_return = 0; pct_return = 0; shares_held=0.0 # Ensure shares_held is float zero
        
        earliest_date = min(data['purchase_dates']) if data['purchase_dates'] else pd.NaT
            
        results.append({
            'ASX_Code': ticker_orig, 'Earliest_Purchase_Date': earliest_date, 'Shares': shares_held,
            'Avg_Purchase_Price_Net': avg_purchase_price, 'Current_Price': current_price,
            'Cost_of_Current_Holdings': cost_of_current_holdings, 'Current_Value': current_value,
            'Unrealized_Return': unrealized_return, 'Pct_Return': pct_return,
            'Dividends_Received': total_dividends_received_by_stock.get(ticker_orig, 0),
            'Realized_PnL': realized_gains_losses.get(ticker_orig, 0)
        })
    st.session_state.calculation_messages.append("Info: Portfolio metrics calculation finished.")
    return pd.DataFrame(results)

# --- Historical Portfolio Value Calculation ---
def calculate_historical_portfolio_value(portfolio_df): # Removed metrics_df as it wasn't used
    if portfolio_df.empty:
        return None
    st.session_state.calculation_messages.append("Info: Starting historical portfolio value calculation...")

    min_date = portfolio_df['Purchase_Date'].min().date()
    max_date_val = date.today()
    # Ensure min_date is not after max_date_val
    if min_date > max_date_val:
        st.session_state.calculation_messages.append(f"Warning: Earliest transaction date ({min_date}) is after today. No historical data to plot.")
        return pd.DataFrame(columns=['Date', 'Portfolio_Value'])
        
    date_range = pd.date_range(start=min_date, end=max_date_val, freq='B')
    if date_range.empty:
        st.session_state.calculation_messages.append("Warning: Date range for historical calculation is empty.")
        return pd.DataFrame(columns=['Date', 'Portfolio_Value'])


    historical_values = []
    unique_tickers_in_portfolio = portfolio_df['ASX_Code'].unique()

    all_time_histories = {}
    for ticker_orig in unique_tickers_in_portfolio:
        ticker_ax = f"{ticker_orig.replace('.AX', '')}.AX"
        history = fetch_stock_price_history(ticker_ax, start_date=min_date - timedelta(days=7), end_date=max_date_val)
        if history is not None:
            all_time_histories[ticker_orig] = history
        else:
            st.session_state.calculation_messages.append(f"Warning: No historical prices for {ticker_orig} for history calc.")

    for day_dt_in_range in date_range:
        day_as_date_obj = day_dt_in_range.date() # Convert timestamp to date for comparison
        daily_total_value = 0
        for ticker_orig in unique_tickers_in_portfolio:
            relevant_transactions = portfolio_df[
                (portfolio_df['ASX_Code'] == ticker_orig) &
                (portfolio_df['Purchase_Date'].dt.date <= day_as_date_obj)
            ]
            shares_bought = relevant_transactions[relevant_transactions['Transaction_Type'] == 'Buy']['Shares'].sum()
            shares_sold = relevant_transactions[relevant_transactions['Transaction_Type'] == 'Sell']['Shares'].sum()
            shares_held_on_day = shares_bought - shares_sold

            if shares_held_on_day > 1e-6:
                price_hist_df = all_time_histories.get(ticker_orig)
                price_on_day = get_price_on_date(price_hist_df, day_dt_in_range) if price_hist_df is not None else 0
                daily_total_value += shares_held_on_day * price_on_day
        
        # Add value even if zero for continuity, or if it's the first day to establish baseline
        if daily_total_value > 0 or day_as_date_obj == date_range[0].date():
             historical_values.append({'Date': day_dt_in_range, 'Portfolio_Value': daily_total_value})

    st.session_state.calculation_messages.append("Info: Historical portfolio value calculation finished.")
    return pd.DataFrame(historical_values) if historical_values else pd.DataFrame(columns=['Date', 'Portfolio_Value'])


# --- UI Rendering: Sidebar ---
def render_sidebar():
    with st.sidebar:
        st.header("📋 Portfolio Management")
        
        st.subheader("Add New Transaction")
        # Using unique keys for all widgets inside the form
        with st.form("new_transaction_form_unique", clear_on_submit=True):
            ticker_input_sb = st.text_input("ASX Code", placeholder="e.g., BHP", key="ticker_input_sb_key").upper()
            p_date_sb = st.date_input("Transaction Date", value=datetime.today(), key="p_date_sb_key")
            
            col1_sb, col2_sb = st.columns(2)
            shares_input_sb = col1_sb.number_input("Shares", min_value=0.0001, value=100.0, step=0.01, format="%.4f", key="shares_input_sb_key")
            price_input_sb = col2_sb.number_input("Price (AUD)", min_value=0.0000, value=10.00, format="%.4f", key="price_input_sb_key")
            
            type_input_sb = col1_sb.selectbox("Type", ["Buy", "Sell"], key="type_input_sb_key")
            fees_input_sb = col2_sb.number_input("Fees (AUD)", min_value=0.0, value=10.0, format="%.2f", key="fees_input_sb_key")
            
            notes_input_sb = st.text_area("Notes (Optional)", height=50, key="notes_input_sb_key") # Explicit key added
            
            submitted_sb = st.form_submit_button("Add Transaction")
            if submitted_sb:
                if ticker_input_sb and shares_input_sb > 0:
                    current_unique_stocks = st.session_state.portfolio['ASX_Code'].nunique()
                    is_new_stock = ticker_input_sb not in st.session_state.portfolio['ASX_Code'].unique()
                    if current_unique_stocks >= MAX_UNIQUE_STOCKS and is_new_stock:
                        st.error(f"Cannot add. Portfolio limit of {MAX_UNIQUE_STOCKS} unique stocks reached.")
                    else:
                        new_trans = pd.DataFrame([{'ASX_Code': ticker_input_sb, 'Purchase_Date': pd.to_datetime(p_date_sb),
                                                   'Shares': shares_input_sb, 'Purchase_Price': price_input_sb,
                                                   'Transaction_Type': type_input_sb, 'Fees': fees_input_sb, 'Notes': notes_input_sb}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_trans], ignore_index=True)
                        st.session_state.portfolio_metrics = None
                        st.session_state.historical_portfolio_performance = None
                        st.toast(f"Added {type_input_sb} for {ticker_input_sb}!", icon="✅")
                        st.rerun()
                else:
                    st.error("Please fill in Ticker and Shares correctly.")
        
        st.divider()
        st.subheader("Import/Export (Max " + str(MAX_UNIQUE_STOCKS) + " unique stocks)")
        def reset_file_processed_flag_sb(): st.session_state.file_processed = False
        uploaded_file_sb = st.file_uploader("Import Transactions (CSV)", type="csv", key="csv_uploader_sidebar_key", on_change=reset_file_processed_flag_sb)

        if uploaded_file_sb is not None and not st.session_state.get('file_processed', False):
            if st.button("Process Imported CSV File", key="process_csv_sb_btn"):
                try:
                    imported_df = pd.read_csv(uploaded_file_sb)
                    expected_cols = ['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type'] # Fees, Notes are optional
                    if not all(col in imported_df.columns for col in expected_cols):
                        st.error("CSV missing essential columns. Required: " + ", ".join(expected_cols))
                    else:
                        imported_df['ASX_Code'] = imported_df['ASX_Code'].astype(str).str.upper()
                        if imported_df['ASX_Code'].nunique() > MAX_UNIQUE_STOCKS:
                            st.error(f"CSV has {imported_df['ASX_Code'].nunique()} unique stocks, exceeding limit of {MAX_UNIQUE_STOCKS}. Import aborted.")
                        else:
                            imported_df['Purchase_Date'] = pd.to_datetime(imported_df['Purchase_Date'], errors='coerce')
                            for col in ['Shares', 'Purchase_Price', 'Fees']:
                                if col in imported_df.columns:
                                    imported_df[col] = pd.to_numeric(imported_df[col], errors='coerce')
                            imported_df['Fees'] = imported_df.get('Fees', pd.Series(0.0, index=imported_df.index)).fillna(0) # Handle missing Fees column
                            imported_df['Notes'] = imported_df.get('Notes', pd.Series("", index=imported_df.index)).fillna("") # Handle missing Notes

                            # Ensure all required columns exist before dropping NA
                            cols_to_check_na = [col for col in expected_cols if col in imported_df.columns]
                            imported_df.dropna(subset=cols_to_check_na, inplace=True)


                            st.session_state.portfolio = imported_df[['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type', 'Fees', 'Notes']]
                            st.session_state.portfolio_metrics = None
                            st.session_state.historical_portfolio_performance = None
                            st.session_state.file_processed = True
                            st.toast("CSV imported! Replaced current portfolio.", icon="📄")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
                    logger.error(f"CSV Processing error: {e}")

        if not st.session_state.portfolio.empty:
            csv_export = st.session_state.portfolio.copy()
            if 'Purchase_Date' in csv_export.columns:
                 csv_export['Purchase_Date'] = pd.to_datetime(csv_export['Purchase_Date']).dt.strftime('%Y-%m-%d')
            csv_data = csv_export.to_csv(index=False).encode('utf-8')
            st.download_button("Export Transactions (CSV)", csv_data, "asx_portfolio_transactions.csv", "text/csv", key="export_csv_sb_btn")

        st.divider()
        if st.button("⚠️ Reset All Data", type="secondary", key="reset_all_sb_btn"):
            initialize_session_state()
            if 'csv_uploader_sidebar_key' in st.session_state: st.session_state.csv_uploader_sidebar_key = None
            st.toast("All data has been reset.", icon="🗑️")
            st.rerun()

        st.divider()
        st.subheader("📝 Calculation Notes & Logs")
        if st.session_state.calculation_messages:
            with st.expander("View Details", expanded=False):
                for msg in st.session_state.calculation_messages:
                    if msg.lower().startswith("error:"): st.error(msg)
                    elif msg.lower().startswith("warning:"): st.warning(msg)
                    else: st.info(msg)
        else:
            st.caption("No calculation notes yet.")


# --- UI Rendering: Main Page ---
def render_main_content():
    st.title("🇦🇺 ASX Portfolio Tracker Pro")
    st.markdown("Track your Australian stock investments with dividends, historical performance, and more.")

    if st.session_state.portfolio.empty:
        st.info("👋 Welcome! Add your first transaction or import a CSV using the sidebar to get started.")
        return

    st.markdown("<div class='centered-button'>", unsafe_allow_html=True)
    if st.button("🔄 Calculate/Refresh All Metrics", type="primary", use_container_width=False, key="refresh_all_main_btn"):
        # Explicitly clear messages before new calculation run
        st.session_state.calculation_messages = ["Info: Starting full metrics refresh..."]
        with st.spinner("Calculating portfolio metrics..."):
            st.session_state.portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio.copy())
        if st.session_state.portfolio_metrics is not None and not st.session_state.portfolio_metrics.empty:
            with st.spinner("Calculating historical portfolio performance..."):
                 st.session_state.historical_portfolio_performance = calculate_historical_portfolio_value(
                     st.session_state.portfolio.copy()
                 )
        else: # If metrics are empty or None, historical probably shouldn't run or will be empty
            st.session_state.historical_portfolio_performance = pd.DataFrame(columns=['Date', 'Portfolio_Value'])
            if st.session_state.portfolio_metrics is None:
                st.session_state.calculation_messages.append("Warning: Portfolio metrics calculation failed or returned None.")
            elif st.session_state.portfolio_metrics.empty:
                st.session_state.calculation_messages.append("Warning: Portfolio metrics are empty. Historical performance cannot be fully calculated.")


        st.toast("Metrics refreshed!", icon="💡")
        st.rerun() # Rerun to update display with new messages and metrics
    st.markdown("</div>", unsafe_allow_html=True)

    metrics_df = st.session_state.portfolio_metrics
    if metrics_df is None:
        st.info("Click 'Calculate/Refresh All Metrics' to analyze your portfolio.")
        return
    if metrics_df.empty and not st.session_state.portfolio.empty:
        st.warning("Portfolio contains transactions, but no valid metrics could be calculated. Check data or logs in sidebar.")
        return
    if metrics_df.empty: return

    st.subheader("📈 Portfolio Snapshot")
    current_holdings_metrics = metrics_df[metrics_df['Shares'] > 1e-6]
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    total_current_value = current_holdings_metrics['Current_Value'].sum()
    col_sum1.metric("Portfolio Value", f"${total_current_value:,.2f}")
    
    total_unrealized = current_holdings_metrics['Unrealized_Return'].sum()
    total_realized = metrics_df['Realized_PnL'].sum()
    total_dividends = metrics_df['Dividends_Received'].sum()
    overall_gain_loss = total_unrealized + total_realized + total_dividends
    col_sum2.metric("Overall Gain/Loss", f"${overall_gain_loss:,.2f}", help="Unrealized P/L + Realized P/L + Dividends")

    total_cost_current = current_holdings_metrics['Cost_of_Current_Holdings'].sum()
    return_pct_current = (total_unrealized / total_cost_current) * 100 if abs(total_cost_current) > 1e-6 else 0
    col_sum3.metric("Current Holdings Return %", f"{return_pct_current:.2f}%", help="Unrealized P/L / Cost of current holdings")
    col_sum4.metric("Total Dividends Received", f"${total_dividends:,.2f}")
    st.markdown("---")

    tab_titles = ["📊 Holdings", "📜 Transactions", "💹 Stock Charts", "⏳ Portfolio History", "🥧 Allocation"]
    tab_holdings, tab_transactions, tab_stock_charts, tab_portfolio_history, tab_allocation = st.tabs(tab_titles)

    with tab_holdings:
        st.subheader("Current Holdings & Performance")
        if not metrics_df.empty:
            display_df_h = metrics_df.copy()
            currency_cols = ['Avg_Purchase_Price_Net', 'Current_Price', 'Cost_of_Current_Holdings', 
                             'Current_Value', 'Unrealized_Return', 'Realized_PnL', 'Dividends_Received']
            for col in currency_cols:
                if col in display_df_h.columns:
                    display_df_h[col] = display_df_h[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
            if 'Pct_Return' in display_df_h.columns:
                display_df_h['Pct_Return'] = display_df_h['Pct_Return'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
            if 'Shares' in display_df_h.columns: display_df_h['Shares'] = display_df_h['Shares'].round(4)
            st.dataframe(display_df_h, use_container_width=True, hide_index=True)
        else: st.info("No holdings metrics to display.")

    with tab_transactions:
        st.subheader("Transaction Record (Max " + str(MAX_UNIQUE_STOCKS) + " unique stocks)")
        st.caption("Edit transactions directly. Click 'Save Transaction Changes' below the table to apply.")
        
        # More robust check for edited_transactions_df initialization or reset
        if st.session_state.edited_transactions_df is None or \
           st.session_state.portfolio.shape != st.session_state.edited_transactions_df.shape or \
           not st.session_state.portfolio.reset_index(drop=True).equals(st.session_state.edited_transactions_df.reset_index(drop=True)):
             st.session_state.edited_transactions_df = st.session_state.portfolio.copy()
        
        temp_editable_df = st.session_state.edited_transactions_df.copy()
        if 'Purchase_Date' in temp_editable_df.columns: # Ensure it's datetime.date for editor
            temp_editable_df['Purchase_Date'] = pd.to_datetime(temp_editable_df['Purchase_Date']).dt.date

        edited_df_main = st.data_editor(
            temp_editable_df, num_rows="dynamic", use_container_width=True, key="transaction_editor_main",
            column_config={
                "Purchase_Date": st.column_config.DateColumn("Trans. Date", format="YYYY-MM-DD", width="small"),
                "ASX_Code": st.column_config.TextColumn("ASX Code (UPPER)", validate="^[A-Z0-9.]{3,10}$", width="small"),
                "Shares": st.column_config.NumberColumn("Shares", format="%.4f", min_value=0.0001, width="small"),
                "Purchase_Price": st.column_config.NumberColumn("Price", format="%.4f", min_value=0, width="small"),
                "Fees": st.column_config.NumberColumn("Fees", format="%.2f", min_value=0, width="small"),
                "Transaction_Type": st.column_config.SelectboxColumn("Type", options=["Buy", "Sell"], width="small"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            }
        )

        if st.button("💾 Save Transaction Changes", key="save_trans_main_btn"):
            saved_df = edited_df_main.copy()
            saved_df['ASX_Code'] = saved_df['ASX_Code'].astype(str).str.upper()
            if saved_df['ASX_Code'].nunique() > MAX_UNIQUE_STOCKS:
                st.error(f"Edited transactions result in {saved_df['ASX_Code'].nunique()} unique stocks, exceeding limit of {MAX_UNIQUE_STOCKS}. Not saved.")
            else:
                if 'Purchase_Date' in saved_df.columns:
                     saved_df['Purchase_Date'] = pd.to_datetime(saved_df['Purchase_Date']) # Convert back to datetime64
                st.session_state.portfolio = saved_df.reset_index(drop=True) # Reset index after edits
                st.session_state.edited_transactions_df = saved_df.reset_index(drop=True).copy()
                st.session_state.portfolio_metrics = None
                st.session_state.historical_portfolio_performance = None
                st.toast("Transaction changes saved! Recalculate metrics.", icon="💾")
                st.rerun()

    with tab_stock_charts:
        st.subheader("Individual Stock Performance Chart")
        # Use metrics_df for ticker selection as it represents calculated holdings
        if metrics_df is not None and not metrics_df.empty:
            valid_tickers_for_chart = sorted(list(metrics_df['ASX_Code'].unique()))
            
            current_selection_sc = st.session_state.get('selected_stock_for_chart', None)
            if current_selection_sc not in valid_tickers_for_chart and valid_tickers_for_chart:
                current_selection_sc = valid_tickers_for_chart[0]
            elif not valid_tickers_for_chart:
                current_selection_sc = None
            
            st.session_state.selected_stock_for_chart = st.selectbox(
                "Select Stock:", options=valid_tickers_for_chart, 
                index=valid_tickers_for_chart.index(current_selection_sc) if current_selection_sc else 0,
                key="stock_chart_select_key"
            )

            if st.session_state.selected_stock_for_chart:
                ticker_sc = st.session_state.selected_stock_for_chart
                ticker_ax_sc = f"{ticker_sc.replace('.AX', '')}.AX"
                
                stock_info_sc = get_stock_info(ticker_ax_sc)
                st.markdown(f"#### {stock_info_sc['longName']} ({ticker_sc})")

                price_hist_sc = fetch_stock_price_history(ticker_ax_sc, period="5y") # Consider making period selectable
                if price_hist_sc is not None:
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(x=price_hist_sc.index, y=price_hist_sc['Close'], mode='lines', name='Close Price'))
                    stock_transactions_sc = st.session_state.portfolio[st.session_state.portfolio['ASX_Code'] == ticker_sc]
                    buy_dates_sc = stock_transactions_sc[stock_transactions_sc['Transaction_Type'] == 'Buy']['Purchase_Date']
                    buy_prices_sc = stock_transactions_sc[stock_transactions_sc['Transaction_Type'] == 'Buy']['Purchase_Price']
                    sell_dates_sc = stock_transactions_sc[stock_transactions_sc['Transaction_Type'] == 'Sell']['Purchase_Date']
                    sell_prices_sc = stock_transactions_sc[stock_transactions_sc['Transaction_Type'] == 'Sell']['Purchase_Price']
                    fig_sc.add_trace(go.Scatter(x=buy_dates_sc, y=buy_prices_sc, mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')))
                    fig_sc.add_trace(go.Scatter(x=sell_dates_sc, y=sell_prices_sc, mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
                    fig_sc.update_layout(title=f"{stock_info_sc['longName']} Price & Transactions", xaxis_title="Date", yaxis_title=f"Price ({stock_info_sc['currency']})", height=500)
                    st.plotly_chart(fig_sc, use_container_width=True)
                else: st.warning(f"Could not load price history for {ticker_sc}.")
        else: st.info("No stocks with current holdings to chart.")

    with tab_portfolio_history:
        st.subheader("Portfolio Value Over Time")
        hist_perf_df = st.session_state.historical_portfolio_performance
        if hist_perf_df is not None and not hist_perf_df.empty:
            fig_hist = px.line(hist_perf_df, x='Date', y='Portfolio_Value', title="Historical Portfolio Value")
            fig_hist.update_layout(height=500, yaxis_title="Portfolio Value (AUD)")
            st.plotly_chart(fig_hist, use_container_width=True)
        elif st.session_state.portfolio_metrics is not None: # Metrics exist, but historical might not have run or was empty
            st.info("Historical performance data not yet calculated or no data available for the period. Click 'Calculate/Refresh All Metrics'.")
        else: # No metrics at all
            st.info("Calculate metrics to see historical performance.")

    with tab_allocation:
        st.subheader("Portfolio Allocation (by Current Value)")
        alloc_df = current_holdings_metrics[current_holdings_metrics['Current_Value'] > 0] # Use filtered metrics
        if not alloc_df.empty :
            fig_alloc = px.pie(
                alloc_df, values='Current_Value', names='ASX_Code',
                title='Portfolio Allocation by Current Market Value', hole=0.4
            )
            fig_alloc.update_traces(
                textposition='inside', textinfo='percent+label', 
                hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent:.2%}<extra></extra>"
            )
            fig_alloc.update_layout(height=600, legend_title_text='ASX Code')
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("No current holdings with positive value to display allocation for.")

# --- Main Execution ---
if __name__ == "__main__":
    render_sidebar()
    render_main_content()
    logger.info("Streamlit app execution cycle finished.")
