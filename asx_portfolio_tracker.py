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
             st.session_state.calculation_messages.append(f"Warning: No price history found for {ticker_ax} in the requested period.")
        return history if not history.empty else None
    except Exception as e:
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
            # Not an error, many stocks don't pay dividends
            logger.info(f"No dividend history found for {ticker_ax}.")
        return dividends if not dividends.empty else None
    except Exception as e:
        st.session_state.calculation_messages.append(f"Error: Could not fetch dividend history for {ticker_ax}: {e}")
        logger.error(f"fetch_dividend_history error for {ticker_ax}: {e}")
        return None

def get_price_on_date(price_history_df, target_date):
    if price_history_df is None or price_history_df.empty:
        return 0
    target_date_dt = pd.to_datetime(target_date).tz_localize(None) # Ensure timezone naive for comparison
    # Ensure index is also timezone naive
    price_history_df.index = price_history_df.index.tz_localize(None)

    closest_prices = price_history_df[price_history_df.index <= target_date_dt]
    return closest_prices['Close'].iloc[-1] if not closest_prices.empty else 0


# --- Core Logic: Portfolio Metrics Calculation ---
def calculate_portfolio_metrics(portfolio_df):
    st.session_state.calculation_messages = []
    if portfolio_df.empty:
        return pd.DataFrame()

    portfolio_df['Purchase_Date'] = pd.to_datetime(portfolio_df['Purchase_Date'])
    portfolio_df['Shares'] = pd.to_numeric(portfolio_df['Shares'], errors='coerce')
    portfolio_df['Purchase_Price'] = pd.to_numeric(portfolio_df['Purchase_Price'], errors='coerce')
    portfolio_df['Fees'] = pd.to_numeric(portfolio_df['Fees'], errors='coerce').fillna(0)
    portfolio_df.dropna(subset=['ASX_Code', 'Purchase_Date','Shares', 'Purchase_Price'], inplace=True) # ASX_Code is critical
    portfolio_df['ASX_Code'] = portfolio_df['ASX_Code'].astype(str).str.upper()


    portfolio_df = portfolio_df.sort_values(by=['ASX_Code', 'Purchase_Date', 'Transaction_Type'])

    current_holdings_details = {}
    realized_gains_losses = {}
    total_dividends_received_by_stock = {}

    unique_tickers = portfolio_df['ASX_Code'].unique()
    # Pre-fetch data to minimize API calls within loops
    all_stock_price_histories = {
        ticker: fetch_stock_price_history(f"{ticker.replace('.AX', '')}.AX", period="5y")
        for ticker in unique_tickers
    }
    all_stock_dividend_histories = {
        ticker: fetch_dividend_history(f"{ticker.replace('.AX', '')}.AX")
        for ticker in unique_tickers
    }

    for ticker_orig, group in portfolio_df.groupby('ASX_Code'):
        ticker_ax = f"{ticker_orig.replace('.AX', '')}.AX" # Ensure .AX for yfinance
        current_holdings_details[ticker_orig] = {'total_shares': 0.0, 'weighted_cost_sum': 0.0, 'purchase_dates': []}
        realized_gains_losses[ticker_orig] = 0.0
        total_dividends_received_by_stock[ticker_orig] = 0.0
        
        stock_dividends_df = all_stock_dividend_histories.get(ticker_orig)
        processed_transactions = []

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
                processed_transactions.append({'date': date, 'shares_change': shares, 'type': 'Buy'})
            
            elif row['Transaction_Type'] == 'Sell':
                if current_holdings_details[ticker_orig]['total_shares'] < 1e-6: # Effectively zero
                    st.session_state.calculation_messages.append(f"Warning: Tried to sell {shares} of {ticker_orig} on {date.strftime('%Y-%m-%d')} with no holdings. Ignored.")
                    continue
                
                shares_to_sell = min(shares, current_holdings_details[ticker_orig]['total_shares'])
                if abs(shares_to_sell - shares) > 1e-6 : # If adjusted
                    st.session_state.calculation_messages.append(f"Warning: Adjusted sell of {ticker_orig} from {shares} to {shares_to_sell:.4f} due to insufficient holdings.")

                avg_cost_per_share = current_holdings_details[ticker_orig]['weighted_cost_sum'] / current_holdings_details[ticker_orig]['total_shares']
                cost_of_shares_sold = shares_to_sell * avg_cost_per_share
                proceeds_from_sale = (shares_to_sell * price) - fees
                
                realized_gains_losses[ticker_orig] += (proceeds_from_sale - cost_of_shares_sold)
                current_holdings_details[ticker_orig]['weighted_cost_sum'] -= cost_of_shares_sold
                current_holdings_details[ticker_orig]['total_shares'] -= shares_to_sell
                processed_transactions.append({'date': date, 'shares_change': -shares_to_sell, 'type': 'Sell'})

                if abs(current_holdings_details[ticker_orig]['total_shares']) < 1e-6 :
                    current_holdings_details[ticker_orig]['total_shares'] = 0.0
                    current_holdings_details[ticker_orig]['weighted_cost_sum'] = 0.0
                    current_holdings_details[ticker_orig]['purchase_dates'] = []
        
        if stock_dividends_df is not None and not stock_dividends_df.empty:
            processed_transactions.sort(key=lambda x: x['date'])
            shares_held_on_dividend_date_calc = 0.0 # Use a separate var for this calculation pass
            current_transaction_idx = 0
            temp_holdings_for_div_calc = pd.DataFrame(processed_transactions)

            for div_date, div_amount in stock_dividends_df.items():
                div_date_dt = pd.to_datetime(div_date).tz_localize(None) # Ensure timezone naive
                
                # Sum shares_change for transactions UP TO AND INCLUDING the div_date
                relevant_trans_for_div = temp_holdings_for_div_calc[pd.to_datetime(temp_holdings_for_div_calc['date']).dt.tz_localize(None) <= div_date_dt]
                shares_held_on_dividend_date_calc = relevant_trans_for_div['shares_change'].sum()
                                
                if shares_held_on_dividend_date_calc > 1e-6: # If shares held
                    total_dividends_received_by_stock[ticker_orig] += shares_held_on_dividend_date_calc * div_amount

    results = []
    for ticker_orig, data in current_holdings_details.items():
        current_price = 0
        stock_price_hist = all_stock_price_histories.get(ticker_orig) # ticker_orig is already upper
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
            avg_purchase_price = 0; current_value = 0; unrealized_return = 0; pct_return = 0
        
        earliest_date = min(data['purchase_dates']) if data['purchase_dates'] else pd.NaT
            
        results.append({
            'ASX_Code': ticker_orig, 'Earliest_Purchase_Date': earliest_date, 'Shares': shares_held,
            'Avg_Purchase_Price_Net': avg_purchase_price, 'Current_Price': current_price,
            'Cost_of_Current_Holdings': cost_of_current_holdings, 'Current_Value': current_value,
            'Unrealized_Return': unrealized_return, 'Pct_Return': pct_return,
            'Dividends_Received': total_dividends_received_by_stock.get(ticker_orig, 0),
            'Realized_PnL': realized_gains_losses.get(ticker_orig, 0)
        })
    return pd.DataFrame(results)

# --- Historical Portfolio Value Calculation ---
def calculate_historical_portfolio_value(portfolio_df, metrics_df):
    if portfolio_df.empty:
        return None
    st.session_state.calculation_messages.append("Info: Starting historical portfolio value calculation...")

    min_date = portfolio_df['Purchase_Date'].min().date() # Use .date()
    max_date_val = date.today()
    date_range = pd.date_range(start=min_date, end=max_date_val, freq='B')

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

    for day_dt in date_range:
        day = day_dt.date() # Convert timestamp to date for comparison with Purchase_Date.dt.date
        daily_total_value = 0
        for ticker_orig in unique_tickers_in_portfolio:
            relevant_transactions = portfolio_df[
                (portfolio_df['ASX_Code'] == ticker_orig) &
                (portfolio_df['Purchase_Date'].dt.date <= day) # Compare date parts
            ]
            shares_bought = relevant_transactions[relevant_transactions['Transaction_Type'] == 'Buy']['Shares'].sum()
            shares_sold = relevant_transactions[relevant_transactions['Transaction_Type'] == 'Sell']['Shares'].sum()
            shares_held_on_day = shares_bought - shares_sold

            if shares_held_on_day > 1e-6:
                price_hist_df = all_time_histories.get(ticker_orig)
                price_on_day = get_price_on_date(price_hist_df, day_dt) if price_hist_df is not None else 0 # Pass datetime object to get_price_on_date
                daily_total_value += shares_held_on_day * price_on_day
        
        if daily_total_value > 0 or day == date_range[0].date():
             historical_values.append({'Date': day_dt, 'Portfolio_Value': daily_total_value})

    st.session_state.calculation_messages.append("Info: Historical portfolio value calculation finished.")
    return pd.DataFrame(historical_values) if historical_values else None


# --- UI Rendering: Sidebar ---
def render_sidebar():
    with st.sidebar:
        st.header("📋 Portfolio Management")
        
        st.subheader("Add New Transaction")
        with st.form("new_transaction_form", clear_on_submit=True):
            ticker_input = st.text_input("ASX Code", placeholder="e.g., BHP").upper()
            p_date = st.date_input("Transaction Date", value=datetime.today())
            
            col1, col2 = st.columns(2)
            shares_input = col1.number_input("Shares", min_value=0.0001, value=100.0, step=0.01, format="%.4f")
            price_input = col2.number_input("Price (AUD)", min_value=0.0000, value=10.00, format="%.4f")
            
            type_input = col1.selectbox("Type", ["Buy", "Sell"])
            fees_input = col2.number_input("Fees (AUD)", min_value=0.0, value=10.0, format="%.2f")
            
            notes_input = st.text_area("Notes (Optional)", height=50)
            
            submitted = st.form_submit_button("Add Transaction")
            if submitted:
                if ticker_input and shares_input > 0: # Price can be 0
                    # Stock limit check for manual add
                    current_unique_stocks = st.session_state.portfolio['ASX_Code'].nunique()
                    is_new_stock = ticker_input not in st.session_state.portfolio['ASX_Code'].unique()
                    if current_unique_stocks >= MAX_UNIQUE_STOCKS and is_new_stock:
                        st.error(f"Cannot add new stock. Portfolio limit of {MAX_UNIQUE_STOCKS} unique stocks reached.")
                    else:
                        new_trans = pd.DataFrame([{'ASX_Code': ticker_input, 'Purchase_Date': pd.to_datetime(p_date),
                                                   'Shares': shares_input, 'Purchase_Price': price_input,
                                                   'Transaction_Type': type_input, 'Fees': fees_input, 'Notes': notes_input}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_trans], ignore_index=True)
                        st.session_state.portfolio_metrics = None
                        st.session_state.historical_portfolio_performance = None
                        st.toast(f"Added {type_input} for {ticker_input}!", icon="✅")
                        st.rerun()
                else:
                    st.error("Please fill in Ticker and Shares correctly.")
        
        st.divider()
        st.subheader("Import/Export (Max " + str(MAX_UNIQUE_STOCKS) + " unique stocks)")
        # --- CSV Import/Export (Simplified) ---
        def reset_file_processed_flag(): st.session_state.file_processed = False
        uploaded_file = st.file_uploader("Import Transactions (CSV)", type="csv", key="csv_uploader_key", on_change=reset_file_processed_flag)

        if uploaded_file is not None and not st.session_state.get('file_processed', False):
            if st.button("Process Imported CSV File"):
                try:
                    imported_df = pd.read_csv(uploaded_file)
                    # Basic validation: check for essential columns (example)
                    # Assuming columns: ASX_Code,Purchase_Date,Shares,Purchase_Price,Transaction_Type,Fees,Notes
                    expected_cols = ['ASX_Code', 'Purchase_Date', 'Shares', 'Purchase_Price', 'Transaction_Type']
                    if not all(col in imported_df.columns for col in expected_cols):
                        st.error("CSV missing essential columns. Required: " + ", ".join(expected_cols))
                    else:
                        imported_df['ASX_Code'] = imported_df['ASX_Code'].astype(str).str.upper()
                        # Stock limit check for CSV
                        if imported_df['ASX_Code'].nunique() > MAX_UNIQUE_STOCKS:
                            st.error(f"CSV contains {imported_df['ASX_Code'].nunique()} unique stocks, exceeding the limit of {MAX_UNIQUE_STOCKS}. Import aborted.")
                        else:
                            # Ensure correct dtypes before assigning (add more robust parsing if needed)
                            imported_df['Purchase_Date'] = pd.to_datetime(imported_df['Purchase_Date'], errors='coerce')
                            for col in ['Shares', 'Purchase_Price', 'Fees']:
                                if col in imported_df.columns:
                                    imported_df[col] = pd.to_numeric(imported_df[col], errors='coerce')
                            imported_df['Fees'] = imported_df['Fees'].fillna(0)
                            imported_df.dropna(subset=expected_cols, inplace=True)

                            st.session_state.portfolio = imported_df # Replace current portfolio
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
            st.download_button("Export Transactions (CSV)", csv_data, "asx_portfolio_transactions.csv", "text/csv")

        st.divider()
        if st.button("⚠️ Reset All Data", type="secondary"):
            initialize_session_state()
            if 'csv_uploader_key' in st.session_state: st.session_state.csv_uploader_key = None # Reset file uploader
            st.toast("All data has been reset.", icon="🗑️")
            st.rerun()

        # --- Calculation Messages in Sidebar ---
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
    if st.button("🔄 Calculate/Refresh All Metrics", type="primary", use_container_width=False):
        with st.spinner("Calculating portfolio metrics..."):
            st.session_state.portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio.copy())
        if st.session_state.portfolio_metrics is not None and not st.session_state.portfolio_metrics.empty:
            with st.spinner("Calculating historical portfolio performance..."):
                 st.session_state.historical_portfolio_performance = calculate_historical_portfolio_value(
                     st.session_state.portfolio.copy(), 
                     st.session_state.portfolio_metrics.copy() # metrics_df not directly used in current version of hist calc
                 )
        st.toast("Metrics refreshed!", icon="💡")
        st.rerun()
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
    current_holdings_metrics = metrics_df[metrics_df['Shares'] > 1e-6] # Consider float precision for shares
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
    col_sum3.metric("Current Holdings Return %", f"{return_pct_current:.2f}%", help="Unrealized P/L on current holdings / Cost of current holdings")
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
                display_df_h[col] = display_df_h[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
            display_df_h['Pct_Return'] = display_df_h['Pct_Return'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
            if 'Shares' in display_df_h.columns: display_df_h['Shares'] = display_df_h['Shares'].round(4)
            st.dataframe(display_df_h, use_container_width=True, hide_index=True)
        else: st.info("No holdings metrics to display.")

    with tab_transactions:
        st.subheader("Transaction Record (Max " + str(MAX_UNIQUE_STOCKS) + " unique stocks)")
        st.caption("Edit transactions directly. Click 'Save Transaction Changes' below the table to apply.")
        
        if st.session_state.edited_transactions_df is None or \
           st.session_state.portfolio.shape[0] != st.session_state.edited_transactions_df.shape[0] or \
           not st.session_state.portfolio.equals(st.session_state.edited_transactions_df.set_index(st.session_state.portfolio.index).assign(Purchase_Date=lambda x: pd.to_datetime(x['Purchase_Date']))): # Complex check if DF differs
             st.session_state.edited_transactions_df = st.session_state.portfolio.copy()
        
        temp_editable_df = st.session_state.edited_transactions_df.copy()
        if 'Purchase_Date' in temp_editable_df.columns:
            temp_editable_df['Purchase_Date'] = pd.to_datetime(temp_editable_df['Purchase_Date']).dt.date

        edited_df = st.data_editor(
            temp_editable_df, num_rows="dynamic", use_container_width=True, key="transaction_editor",
            column_config={
                "Purchase_Date": st.column_config.DateColumn("Transaction Date", format="YYYY-MM-DD"),
                "ASX_Code": st.column_config.TextColumn("ASX Code (UPPER)", validate="^[A-Z0-9.]{3,10}$"), # Allow .AX
                "Shares": st.column_config.NumberColumn("Shares", format="%.4f", min_value=0.0001),
                "Purchase_Price": st.column_config.NumberColumn("Price", format="%.4f", min_value=0),
                "Fees": st.column_config.NumberColumn("Fees", format="%.2f", min_value=0),
                "Transaction_Type": st.column_config.SelectboxColumn("Type", options=["Buy", "Sell"]),
                "Notes": st.column_config.TextColumn("Notes"),
            }
        )

        if st.button("💾 Save Transaction Changes"):
            saved_df = edited_df.copy()
            saved_df['ASX_Code'] = saved_df['ASX_Code'].astype(str).str.upper() # Ensure uppercase
            if saved_df['ASX_Code'].nunique() > MAX_UNIQUE_STOCKS:
                st.error(f"Edited transactions result in {saved_df['ASX_Code'].nunique()} unique stocks, exceeding the limit of {MAX_UNIQUE_STOCKS}. Changes not saved.")
            else:
                if 'Purchase_Date' in saved_df.columns:
                     saved_df['Purchase_Date'] = pd.to_datetime(saved_df['Purchase_Date'])
                st.session_state.portfolio = saved_df
                st.session_state.edited_transactions_df = saved_df.copy()
                st.session_state.portfolio_metrics = None
                st.session_state.historical_portfolio_performance = None
                st.toast("Transaction changes saved! Recalculate metrics.", icon="💾")
                st.rerun()

    with tab_stock_charts:
        st.subheader("Individual Stock Performance Chart")
        if not current_holdings_metrics.empty: # Use current holdings for selection
            all_tickers = sorted(list(current_holdings_metrics['ASX_Code'].unique()))
            # Persist selected stock or default to first
            current_selection = st.session_state.selected_stock_for_chart
            if current_selection not in all_tickers and all_tickers:
                current_selection = all_tickers[0]
            elif not all_tickers:
                current_selection = None
            
            st.session_state.selected_stock_for_chart = st.selectbox(
                "Select Stock:", options=all_tickers, 
                index=all_tickers.index(current_selection) if current_selection else 0
            )

            if st.session_state.selected_stock_for_chart:
                ticker = st.session_state.selected_stock_for_chart
                ticker_ax = f"{ticker.replace('.AX', '')}.AX"
                
                stock_info = get_stock_info(ticker_ax)
                st.markdown(f"#### {stock_info['longName']} ({ticker})")

                price_hist = fetch_stock_price_history(ticker_ax, period="5y")
                if price_hist is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=price_hist.index, y=price_hist['Close'], mode='lines', name='Close Price'))
                    stock_transactions = st.session_state.portfolio[st.session_state.portfolio['ASX_Code'] == ticker]
                    buy_dates = stock_transactions[stock_transactions['Transaction_Type'] == 'Buy']['Purchase_Date']
                    buy_prices = stock_transactions[stock_transactions['Transaction_Type'] == 'Buy']['Purchase_Price']
                    sell_dates = stock_transactions[stock_transactions['Transaction_Type'] == 'Sell']['Purchase_Date']
                    sell_prices = stock_transactions[stock_transactions['Transaction_Type'] == 'Sell']['Purchase_Price']
                    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='triangle-up')))
                    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
                    fig.update_layout(title=f"{stock_info['longName']} Price & Transactions", xaxis_title="Date", yaxis_title=f"Price ({stock_info['currency']})", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning(f"Could not load price history for {ticker}.")
        else: st.info("No stocks with current holdings to chart.")

    with tab_portfolio_history:
        st.subheader("Portfolio Value Over Time")
        hist_perf_df = st.session_state.historical_portfolio_performance
        if hist_perf_df is not None and not hist_perf_df.empty:
            fig_hist = px.line(hist_perf_df, x='Date', y='Portfolio_Value', title="Historical Portfolio Value")
            fig_hist.update_layout(height=500, yaxis_title="Portfolio Value (AUD)")
            st.plotly_chart(fig_hist, use_container_width=True)
        elif st.session_state.portfolio_metrics is not None:
            st.info("Historical performance data not yet calculated. Click 'Calculate/Refresh All Metrics'.")
        else: st.info("Calculate metrics to see historical performance.")

    with tab_allocation:
        st.subheader("Portfolio Allocation (by Current Value)")
        alloc_df = current_holdings_metrics[current_holdings_metrics['Current_Value'] > 0]
        if not alloc_df.empty :
            fig_alloc = px.pie(
                alloc_df, values='Current_Value', names='ASX_Code',
                title='Portfolio Allocation by Current Market Value', hole=0.4
            )
            # Round hover data for percentage
            fig_alloc.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent:.2%}<extra></extra>" # .2% formats as percentage with 2 decimals
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
