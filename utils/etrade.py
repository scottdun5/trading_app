#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from requests_oauthlib import OAuth1Session, OAuth1
import webbrowser
import pandas as pd
#from alpaca_trade_api.rest import REST
from datetime import datetime
#import talib
import utils.config as c

account_id_key = c.ACCOUNT_ID_KEY
BASE_URL = c.BASE_URL
end_date = datetime.now().strftime("%m%d%Y")


# ===================================================
# 1) ETRADE AUTHORIZATION FUNCTION
# ===================================================
def etrade_authorize(consumer_key, consumer_secret, callback_uri="oob"):
    """
    Runs E*TRADE OAuth1 authorization and returns an authenticated session + access tokens.
    """
    REQUEST_TOKEN_URL = 'https://api.etrade.com/oauth/request_token'
    AUTHORIZATION_URL = 'https://us.etrade.com/e/t/etws/authorize'
    ACCESS_TOKEN_URL = 'https://api.etrade.com/oauth/access_token'
    BASE_URL = 'https://api.etrade.com'
    
    # Step 1: Get request token
    oauth = OAuth1Session(consumer_key, client_secret=consumer_secret, callback_uri=callback_uri)
    tokens = oauth.fetch_request_token(REQUEST_TOKEN_URL)
    resource_owner_key = tokens.get('oauth_token')
    resource_owner_secret = tokens.get('oauth_token_secret')

    # Step 2: Ask user to authorize
    auth_url = f"{AUTHORIZATION_URL}?key={consumer_key}&token={resource_owner_key}"
    print("Go to this URL and authorize the app:")
    print(auth_url)
    webbrowser.open(auth_url)

    # Step 3: Get verifier PIN from user
    verifier = input("Enter the verifier code (PIN) from browser: ")

    # Step 4: Fetch access token
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier
    )
    final_tokens = oauth.fetch_access_token(ACCESS_TOKEN_URL)
    access_token = final_tokens.get('oauth_token')
    access_token_secret = final_tokens.get('oauth_token_secret')
    
    # Step 5: Create a reusable authenticated session
    session = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret
    )
    
    return session, BASE_URL, access_token, access_token_secret

def etrade_authorize_start(consumer_key, consumer_secret, callback_uri="oob"):
    """
    Step 1: Start OAuth authorization. Returns:
    - auth_url: URL user visits to authorize app
    - oauth: OAuth1Session object (needed for step 2)
    - resource_owner_key / resource_owner_secret: temporary request tokens
    """
    REQUEST_TOKEN_URL = 'https://api.etrade.com/oauth/request_token'
    AUTHORIZATION_URL = 'https://us.etrade.com/e/t/etws/authorize'

    oauth = OAuth1Session(consumer_key, client_secret=consumer_secret, callback_uri=callback_uri)
    tokens = oauth.fetch_request_token(REQUEST_TOKEN_URL)
    resource_owner_key = tokens.get('oauth_token')
    resource_owner_secret = tokens.get('oauth_token_secret')

    auth_url = f"{AUTHORIZATION_URL}?key={consumer_key}&token={resource_owner_key}"

    return auth_url, oauth, resource_owner_key, resource_owner_secret

def etrade_authorize_finish(consumer_key, consumer_secret, oauth, resource_owner_key,
                            resource_owner_secret, verifier):
    """
    Step 2: Complete OAuth authorization using the verifier code.
    Returns:
    - session: authenticated OAuth1Session
    - access_token, access_token_secret
    - BASE_URL
    """
    ACCESS_TOKEN_URL = 'https://api.etrade.com/oauth/access_token'

    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier
    )
    final_tokens = oauth.fetch_access_token(ACCESS_TOKEN_URL)
    access_token = final_tokens.get('oauth_token')
    access_token_secret = final_tokens.get('oauth_token_secret')

    session = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret
    )

    return session, access_token, access_token_secret


# ===================================================
# 2) PORTFOLIO FUNCTIONS
# ===================================================
def get_all_open_orders(session, base_url, account_id_key):
    all_orders = []
    page_number = 1
    count_per_page = 50  # Max is 50 per docs

    while True:
        url = (
            f"{base_url}/v1/accounts/{account_id_key}/orders.json?status=OPEN&"
            f"count={count_per_page}&pageNumber={page_number}"
        )
        response = session.get(url)
        if response.status_code != 200:
            print(f"Error fetching page {page_number}: {response.text}")
            break

        data = response.json()
        orders = data.get("OrdersResponse", {}).get("Order", [])

        if not orders:
            break

        all_orders.extend(orders)

        # If fewer than requested, assume it’s the last page
        if len(orders) < count_per_page:
            break

        page_number += 1

    # Extract only open SELL orders
    open_sell_orders = []

    for o in all_orders:
        details = o.get("OrderDetail", [])
        # Filter only the open sell order details inside each order
        filtered_details = [
            d for d in details
            if d.get("status") == "OPEN" and
               any(i.get("orderAction") == "SELL" for i in d.get("Instrument", []))
        ]
        open_sell_orders.extend(filtered_details)  # add matching details to result list

    rows = []
    for order_detail in open_sell_orders:
        placed_time_ms = order_detail.get("placedTime")
        placed_dt = datetime.fromtimestamp(placed_time_ms / 1000) if placed_time_ms else None

        stop_price = order_detail.get("stopPrice")
        order_value = order_detail.get("orderValue")

        for instrument in order_detail.get("Instrument", []):
            product = instrument.get("Product", {})
            rows.append({
                "symbol": product.get("symbol"),
                "stopPrice": stop_price,
                "order_value": order_value,
                "orderedQuantity": instrument.get("orderedQuantity"),
                "placedTime": placed_dt.strftime("%m/%d/%y %H:%M:%S") if placed_dt else None,
                "placedDate": placed_dt.strftime("%m/%d/%y") if placed_dt else None,
            })

    df_open_sell_orders = pd.DataFrame(rows)
    return df_open_sell_orders

def get_account_balance(session, base_url, account_id_key):
    url = f'https://api.etrade.com/v1/accounts/7ajZ7yHABKa8f6aI9XhD-w/balance.json?instType=BROKERAGE&realTimeNAV=true'
    response = session.get(url)
    response.raise_for_status()
    data = response.json()
    
    total_value = data.get('BalanceResponse', {}).get('Computed', {}).get('RealTimeValues',{}).get('totalAccountValue',{})
    net_cash = data.get('BalanceResponse', {}).get('Computed', {}).get('netCash',{})
    return total_value, net_cash

def get_portfolio(session, base_url, account_id_key):
    url = f"{base_url}/v1/accounts/{account_id_key}/portfolio.json"
    resp = session.get(url)
    resp.raise_for_status()
    portfolio_data = resp.json()
    
    positions = []
    for account in portfolio_data['PortfolioResponse']['AccountPortfolio']:
        for pos in account['Position']:
            symbol = pos['symbolDescription']
            placed_time_ms = pos["dateAcquired"]
            placed_dt = datetime.fromtimestamp(placed_time_ms / 1000) if placed_time_ms else None
            qty = float(pos['quantity'])
            marketValue = float(pos['marketValue'])
            last_price = float(pos['marketValue']) / qty if qty != 0 else 0
            avg_price = float(pos.get('pricePaid', 0))
            positions.append({
                'symbol': symbol,
                'qty': qty,
                'marketValue': marketValue,
                'last_price': last_price,
                'avg_price': avg_price,
                "buyTime": placed_dt.strftime("%m/%d/%y %H:%M:%S") if placed_dt else None,
                "buyDate": placed_dt.strftime("%m/%d/%y") if placed_dt else None,
            })
    pf = pd.DataFrame(positions)
    return pf

def current_portfolio_metrics(account_id_key, session, base_url):
    open_sell_orders = get_all_open_orders(session, base_url, account_id_key)
    pf = get_portfolio(session, base_url, account_id_key)
    total_value, net_cash = get_account_balance(session, base_url, account_id_key)

    merged = pf.merge(open_sell_orders, on = 'symbol')
    merged['position_size'] = round(merged['last_price']*merged['orderedQuantity']/total_value*100,1)
    merged['open_heat_value'] = merged['last_price']*merged['orderedQuantity']-merged['order_value']
    merged['open_heat_ec'] = round((merged['last_price']*merged['orderedQuantity']-merged['order_value'])/total_value*100,2)
    merged['open_heat_stock'] = round((merged['stopPrice']-merged['last_price'])/merged['last_price']*100,2)
    merged['ec_percent'] = round((merged['last_price']-merged['avg_price'])*merged['orderedQuantity']/total_value*100,2)
    merged['percent_from_entry'] = round((merged['last_price']-merged['avg_price'])/merged['avg_price']*100,2)
    merged['ner_value'] = (merged['avg_price']*merged['orderedQuantity']-merged['order_value']).clip(lower=0)
    total_open_heat = round(merged.open_heat_value.sum()/total_value*100,2)
    total_ner = round(merged.ner_value.sum()/total_value*100,2)
    all_risk_total_value = round(total_value-merged.open_heat_value.sum(),2)
    
    return total_open_heat, total_ner, total_value, all_risk_total_value, merged

# ===================================================
# 3) HISTORICAL TRADES FUNCTIONS
# ===================================================

def get_all_trades(account_id_key, session, start_date="01012025", end_date = end_date):
    url = f"{BASE_URL}/v1/accounts/{account_id_key}/transactions.json"
    all_trades = []
    marker = None

    while True:
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "count": 50,
            "sortOrder": "ASC"
        }
        if marker:
            params["marker"] = marker

        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        trades = data.get("TransactionListResponse", {}).get("Transaction", [])
        all_trades.extend(trades)

        marker = data.get("TransactionListResponse", {}).get("marker")
        if not marker:
            break  # No more pages

    return all_trades


def parse_trades_to_df(trades):
    parsed = []

    for tx in trades:
        if 'brokerage' in tx:
            # Convert timestamp to formatted date
            timestamp_ms = tx.get('transactionDate')
            date_str = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%m/%d/%y') if timestamp_ms else None

            row = {
                'date': date_str,
                'type': tx.get('transactionType'),
                'symbol': tx['brokerage'].get('displaySymbol'),
                'quantity': tx['brokerage'].get('quantity'),
                'price': tx['brokerage'].get('price')
            }
            parsed.append(row)

    df = pd.DataFrame(parsed)
    return df

from collections import defaultdict

def match_lifo_trades(df):
    matched_trades = []
    lifo_stack = defaultdict(list)

    for _, row in df.iterrows():
        trade_type = row['type'].lower().strip()
        symbol = row['symbol'].strip()
        qty = abs(row['quantity'])
        price = row['price']
        date = row['date']

        if trade_type == 'bought':
            lifo_stack[symbol].append({'date': date, 'qty': qty, 'price': price})
        elif trade_type == 'sold':
            remaining = qty
            while remaining > 0 and lifo_stack[symbol]:
                last_buy = lifo_stack[symbol][-1]

                matched_qty = min(remaining, last_buy['qty'])
                gain = (price - last_buy['price']) * matched_qty

                matched_trades.append({
                    'symbol': symbol,
                    'buy_date': last_buy['date'],
                    'sell_date': date,
                    'buy_price': last_buy['price'],
                    'sell_price': price,
                    'quantity': matched_qty,
                    'gain': gain
                })

                last_buy['qty'] -= matched_qty
                remaining -= matched_qty

                if last_buy['qty'] == 0:
                    lifo_stack[symbol].pop()  # remove depleted position

    df_out = pd.DataFrame(matched_trades)

    if df_out.empty:
        return df_out

    # Weighted aggregation to combine partial fills
    def weighted_avg(x, val_col, wt_col):
        return (x[val_col] * x[wt_col]).sum() / x[wt_col].sum()

    grouped = (
        df_out.groupby(['symbol', 'buy_date', 'sell_date'], as_index=False)
        .apply(lambda g: pd.Series({
            'quantity': g['quantity'].sum(),
            'gain': g['gain'].sum(),
            'buy_price': weighted_avg(g, 'buy_price', 'quantity'),
            'sell_price': weighted_avg(g, 'sell_price', 'quantity'),
        }))
    )

    # Add percentage return
    grouped['pct_return'] = (grouped['gain'] / (grouped['buy_price'] * grouped['quantity'])) * 100

    return grouped

def aggregate_trades_by_buy(trades_df):
    """
    Aggregate LIFO-matched trades into a single trade per symbol + buy_date.
    Returns total qty, gain, pct_return, avg days in trade, and weighted buy_price.
    """
    if trades_df.empty:
        return trades_df

    # Ensure datetime types
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])

    # Add holding days
    trades_df['days_in_trade'] = (trades_df['sell_date'] - trades_df['buy_date']).dt.days

    def aggregate_group(g):
        total_qty = g['quantity'].sum()
        total_gain = g['gain'].sum()
        weighted_buy_price = (g['buy_price'] * g['quantity']).sum() / total_qty
        total_cost = weighted_buy_price * total_qty

        return pd.Series({
            'buy_price': weighted_buy_price,
            'quantity': total_qty,
            'gain': total_gain,
            'pct_return': (total_gain / total_cost) * 100 if total_cost != 0 else 0,
            'avg_days_in_trade': (g['days_in_trade'] * g['quantity']).sum() / total_qty
        })

    aggregated = trades_df.groupby(['symbol', 'buy_date'], as_index=False).apply(aggregate_group)

    return aggregated.reset_index(drop=True)

def calculate_overall_metrics(df, threshold=0.005):
    """
    Calculate trading performance metrics across the full dataset.
    threshold: fraction (e.g., 0.005 = 0.5%)
    """
    if df.empty:
        return {}

    # Define wins/losses
    df['is_win_simple'] = df['pct_return'] > 0
    df['is_win_thresh'] = df['pct_return'] > (threshold * 100)

    total_trades = len(df)

    # Simple >0
    wins_simple = df[df['is_win_simple']]
    losses_simple = df[~df['is_win_simple']]

    # Threshold-based > threshold
    wins_thresh = df[df['is_win_thresh']]
    losses_thresh = df[~df['is_win_thresh']]

    metrics = {
        "total_trades": total_trades,

        # Simple >0
        "win_rate_simple": len(wins_simple) / total_trades * 100,
        "avg_gain_simple": wins_simple['pct_return'].mean() if not wins_simple.empty else 0,
        "avg_loss_simple": losses_simple['pct_return'].mean() if not losses_simple.empty else 0,
        "avg_days_win_simple": wins_simple['avg_days_in_trade'].mean() if not wins_simple.empty else 0,
        "avg_days_loss_simple": losses_simple['avg_days_in_trade'].mean() if not losses_simple.empty else 0,

        # Threshold-based
        "win_rate_thresh": len(wins_thresh) / total_trades * 100,
        "avg_gain_thresh": wins_thresh['pct_return'].mean() if not wins_thresh.empty else 0,
        "avg_loss_thresh": losses_thresh['pct_return'].mean() if not losses_thresh.empty else 0,
        "avg_days_win_thresh": wins_thresh['avg_days_in_trade'].mean() if not wins_thresh.empty else 0,
        "avg_days_loss_thresh": losses_thresh['avg_days_in_trade'].mean() if not losses_thresh.empty else 0,

        # Extremes
        "biggest_gain": df['pct_return'].max(),
        "biggest_loss": df['pct_return'].min(),
    }

    return metrics


def calculate_monthly_metrics(df, threshold=0.005):
    """
    Calculate trading performance metrics per month.
    Returns a DataFrame with one row per month.
    """
    if df.empty:
        return pd.DataFrame()

    df['month'] = df['buy_date'].dt.to_period('M')

    results = []
    for month, g in df.groupby('month'):
        m = calculate_overall_metrics(g, threshold=threshold)
        m['month'] = str(month)
        results.append(m)
    
    metrics_df = pd.DataFrame(results)
    cols = ['month'] + [c for c in metrics_df.columns if c != 'month']
    return metrics_df[cols]
                      
def get_trade_list(account_id_key, session):
    all_trades = get_all_trades(account_id_key, session)
    trades = parse_trades_to_df(all_trades)
    df = match_lifo_trades(trades)
    result = aggregate_trades_by_buy(df)
    monthly = calculate_monthly_metrics(result)
    
    return result, monthly


    

