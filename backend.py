from dhanhq import dhanhq
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import logging
import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
user_data = {
    'name': '',
    'clientId': '',
    'authToken': ''
}
dhan = None
technical_indicators = {
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std_dev': 2.0
}
def calculate_rsi(prices):
    """Calculate RSI using values from technical_indicators"""
    period = technical_indicators['rsi_period']
    
    # Handle case where we don't have enough data
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)
    
    deltas = np.diff(prices)
    seed = deltas[:period]
    
    # Calculate initial average gains and losses
    gains = np.where(seed > 0, seed, 0)
    losses = np.where(seed < 0, -seed, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    # Handle case where avg_loss is 0 (to avoid division by zero)
    if avg_loss == 0:
        return np.full(len(prices), 100)
    
    rs = avg_gain / avg_loss
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate remaining RSI values
    for i in range(period, len(prices)):
        delta = deltas[i-1] if i > 0 else 0
        
        if delta > 0:
            gain = delta
            loss = 0.0
        else:
            gain = 0.0
            loss = -delta
            
        # Smooth the averages
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_bollinger_bands(prices):
    """Calculate Bollinger Bands using values from technical_indicators"""
    window = technical_indicators['bb_period']
    std_dev = technical_indicators['bb_std_dev']
    
    # Handle case where we don't have enough data
    if len(prices) < window:
        return None, None, None
    
    # Calculate rolling mean and standard deviation
    rolling_mean = pd.Series(prices).rolling(window=window).mean()
    rolling_std = pd.Series(prices).rolling(window=window).std()
    
    # Calculate bands
    upper_band = rolling_mean + (rolling_std * std_dev)
    middle_band = rolling_mean
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    return upper_band.values, middle_band.values, lower_band.values



@app.route('/userinfo', methods=['POST'])
def get_user():
    global dhan, user_data
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in userinfo request")
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['name', 'clientId', 'authToken']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing fields in userinfo: {data}")
            return jsonify({'error': 'Missing required fields'}), 400

        user_data.update({
            'name': data['name'],
            'clientId': data['clientId'],
            'authToken': data['authToken']
        })

        # Initialize DhanHQ client
        dhan = dhanhq(data['clientId'], data['authToken'])
        logger.info(f"User authenticated: {data['name']}")

        return jsonify({'message': 'User info updated successfully'}), 200

    except Exception as e:
        logger.error(f"Error in userinfo: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_auth', methods=['GET'])
def check_auth():
    global dhan
    if dhan:
        return jsonify({'authenticated': True}), 200
    return jsonify({'authenticated': False}), 401

@app.route('/portfolio', methods=['POST'])
def get_portfolio():
    global dhan
    if not dhan:
        return jsonify({"error": "User not authenticated"}), 401

    try:
        holdings = dhan.get_holdings()
        logger.info("Portfolio data fetched successfully")
        return jsonify(holdings)
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/balance', methods=['GET'])
def get_balance():
    global dhan
    if not dhan:
        return jsonify({"error": "User not authenticated"}), 401

    try:
        balance = dhan.get_fund_limits()
        logger.info("Balance data fetched successfully")
        return jsonify(balance)
    except Exception as e:
        logger.error(f"Error fetching balance: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_indicators', methods=['POST'])
def update_indicators():
    global technical_indicators
    try:
        data = request.get_json()
        technical_indicators.update({
            'rsi_period': int(data.get('rsi_period', technical_indicators['rsi_period'])),
            'bb_period': int(data.get('bb_period', technical_indicators['bb_period'])),
            'bb_std_dev': float(data.get('bb_std_dev', technical_indicators['bb_std_dev']))
        })
        logger.info(f"Updated indicators: {technical_indicators}")
        return jsonify({'status': 'success', 'data': technical_indicators}), 200
    except Exception as e:
        logger.error(f"Error updating indicators: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_indicators', methods=['GET'])
def get_indicators():
    return jsonify({'status': 'success', 'data': technical_indicators})






def get_trading_recommendation(rsi, current_price, upper_band, lower_band):
    """Generate trading recommendation based on technical indicators"""
    # Define thresholds
    overbought = 70
    oversold = 30
    band_proximity = 0.02  # 2% proximity to bands
    
    # Check RSI conditions
    is_overbought = rsi > overbought
    is_oversold = rsi < oversold
    
    # Check Bollinger Bands conditions
    near_upper = current_price >= upper_band * (1 - band_proximity)
    near_lower = current_price <= lower_band * (1 + band_proximity)
    
    # Generate recommendation
    if is_overbought and near_upper:
        return "STRONG SELL"
    elif is_oversold and near_lower:
        return "STRONG BUY"
    elif is_overbought:
        return "SELL"
    elif is_oversold:
        return "BUY"
    elif near_upper:
        return "CONSIDER SELLING"
    elif near_lower:
        return "CONSIDER BUYING"
    else:
        return "HOLD"


@app.route('/analyze', methods=['POST'])
def analyze_stocks():
    try:
        print("Received data:", request.json)  # Print received data
        data = request.json
    
    # Verify required fields exist
        if not data or 'stocks' not in data:
            return jsonify({'error': 'Missing stocks data'}), 400
        
        stocks = data['stocks']
        rsi_period = data.get('rsi', 14)
        bb_upper = data.get('bollinger_upper', 20)
        bb_lower = data.get('bollinger_lower', 10)
    
        print(f"Analyzing {len(stocks)} stocks with RSI:{rsi_period}, BB-U:{bb_upper}, BB-L:{bb_lower}")
        data = request.get_json()
        stocks = data.get('stocks', [])
        
        results = []
        for symbol in stocks:
            # Get stock data - replace this with your actual data fetching logic
            time.sleep(1)
            stock_data = yf.download(f"{symbol}.NS", start=datetime.now()-timedelta(180), end=datetime.now())
            stock_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
            if stock_data.empty:
                results.append({
                    'stock': symbol,
                    'error': 'No data available'
                })
                continue
                
            close_prices = stock_data['Close'].values
            current_price = close_prices[-1]
            
            # Calculate indicators
            rsi = calculate_rsi(close_prices)[-1]
            upper_band, _, lower_band = calculate_bollinger_bands(close_prices)
            
            results.append({
                'stock': symbol,
                'current_price': float(current_price),
                'rsi': float(rsi),
                'upper_band': float(upper_band[-1]),
                'lower_band': float(lower_band[-1]),
                'recommendation': get_trading_recommendation(
                    rsi, current_price, upper_band[-1], lower_band[-1]
                )
            })
            
        return jsonify({
            'status': 'success',
            'data': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)