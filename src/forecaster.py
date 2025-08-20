import sys
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prophet import Prophet
def run_revenue_forecast(state):
    if not state.get("ticker"):
        print("\nPlease view fundamentals first (option 1) before forecasting revenue.")
        return

    ticker_symbol = state["ticker"]
    ticker = yf.Ticker(ticker_symbol)

    try:
        fin = ticker.financials
        if fin.empty:
            print("No financial data available for forecasting.")
            return

        revenue_history = fin.loc["Total Revenue"].dropna()
        if len(revenue_history) < 2:
            print("Not enough revenue history to forecast.")
            return

        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': revenue_history.index,
            'y': revenue_history.values
        }).reset_index(drop=True)

        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)

        future = m.make_future_dataframe(periods=1, freq='Y')
        forecast = m.predict(future)

        next_revenue = forecast['yhat'].iloc[-1]
        print(f"\nForecasted next period revenue for {ticker_symbol}: ${next_revenue:,.0f}")

    except Exception as e:
        print(f"Error forecasting revenue: {e}")