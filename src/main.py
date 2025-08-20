import sys
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prophet import Prophet



def get_fundamentals(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    fundamentals = {}
    bookValue = info.get("bookValue", None)

    fundamentals["Company Name"] = info.get("longName")
    fundamentals["Industry"] = info.get("industry")
    fundamentals["Market Cap"] = info.get("marketCap")
    fundamentals["Current Price"] = info.get("currentPrice")
    fundamentals["Daily Change %"] = info.get("regularMarketChangePercent")
    fundamentals["EPS"] = info.get("trailingEps")

    fundamentals["P/E Ratio"] = info.get("trailingPE")
    if info.get("currentPrice") and bookValue and bookValue != 0:
        fundamentals["P/B Ratio"] = info.get("currentPrice") / bookValue
    else:
        fundamentals["P/B Ratio"] = None

    fundamentals["Dividend Yield %"] = (
        info.get("dividendYield") * 100 if info.get("dividendYield") else None
    )

    fundamentals["Revenue (TTM)"] = None
    fundamentals["Net Income"] = None
    try:
        fin = ticker.financials
        if not fin.empty:
            fundamentals["Revenue (TTM)"] = fin.loc["Total Revenue"].iloc[0]
            fundamentals["Net Income"] = fin.loc["Net Income"].iloc[0]
    except Exception:
        pass

    fundamentals["Debt-to-Equity"] = info.get("debtToEquity")
    fundamentals["Revenue Growth %"] = (
        info.get("revenueGrowth") * 100 if info.get("revenueGrowth") else None
    )
    fundamentals["Return on Equity %"] = (
        info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None
    )

    return fundamentals


def show_fundamentals(state):
    ticker = input("\nEnter stock ticker symbol: ").strip().upper()
    fundamentals = get_fundamentals(ticker)
    state["ticker"] = ticker
    state["fundamentals"] = fundamentals

    print(f"\n=== {ticker} Fundamentals ===")
    for k, v in fundamentals.items():
        print(f"{k}: {v}")



def run_classifier(state):
    if not state.get("ticker") or not state.get("fundamentals"):
        print("\nPlease view fundamentals first (option 1) before running the classifier.")
        return

    

    df = pd.read_csv("stocks_enhanced.csv")

    features = df.iloc[:, 1:4].values
    target = df.iloc[:, 6].values

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.25
    )

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(features_train, target_train)

    accuracy_train = accuracy_score(target_train, model.predict(features_train))
    accuracy_test = accuracy_score(target_test, model.predict(features_test))

    print("Accuracy on training data:", accuracy_train)
    print("Accuracy on test data:", accuracy_test)

    
    ticker = state["ticker"]
    fundamentals = state["fundamentals"]

    pe = fundamentals.get("P/E Ratio")
    eps = fundamentals.get("EPS")
    price = fundamentals.get("Current Price")

    if None in (pe, eps, price):
        print(f"Not enough data to classify {ticker}")
        return

    example = [[pe, eps, price]]
    result = model.predict(example)
    print(f"Prediction for {ticker}: {result}")



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



def main():
    state = {"ticker": None, "fundamentals": None}  # shared state

    while True:
        print("\n=== Stock Analysis Menu ===")
        print("1. Show Stock Fundamentals")
        print("2. Run Classifier")
        print("3. Forecast Revenue")
        print("4. Exit")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            show_fundamentals(state)
        elif choice == "2":
            run_classifier(state)
        elif choice == "3":
            run_revenue_forecast(state)
        elif choice == "4":
            print("Exiting program...")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
