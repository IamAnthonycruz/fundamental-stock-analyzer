import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prophet import Prophet
import streamlit as st


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
    fundamentals["P/B Ratio"] = info.get("currentPrice") / bookValue if info.get("currentPrice") and bookValue else None
    fundamentals["Dividend Yield %"] = info.get("dividendYield") * 100 if info.get("dividendYield") else None

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
    fundamentals["Revenue Growth %"] = info.get("revenueGrowth") * 100 if info.get("revenueGrowth") else None
    fundamentals["Return on Equity %"] = info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None

    return fundamentals


st.title("Fundamental Stock Analyzer ")

if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "fundamentals" not in st.session_state:
    st.session_state.fundamentals = None

# Input ticker
ticker_input = st.text_input("Enter Stock Ticker Symbol", value="AAPL").upper()

if st.button("Show Fundamentals"):
    st.session_state.ticker = ticker_input
    st.session_state.fundamentals = get_fundamentals(ticker_input)
    st.subheader(f"Fundamentals for {ticker_input}")
    st.table(pd.DataFrame(list(st.session_state.fundamentals.items()), columns=["Metric", "Value"]))

if st.button("Run Classifier"):
    if not st.session_state.ticker or not st.session_state.fundamentals:
        st.warning("Please show fundamentals first before running the classifier.")
    else:
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

        st.write(f"Accuracy on training data: {accuracy_train:.2f}")
        st.write(f"Accuracy on test data: {accuracy_test:.2f}")

        pe = st.session_state.fundamentals.get("P/E Ratio")
        eps = st.session_state.fundamentals.get("EPS")
        price = st.session_state.fundamentals.get("Current Price")

        if None in (pe, eps, price):
            st.warning(f"Not enough data to classify {st.session_state.ticker}")
        else:
            example = [[pe, eps, price]]
            result = model.predict(example)
            st.success(f"Prediction for {st.session_state.ticker}: {result[0]}")


if st.button("Forecast Revenue"):
    if not st.session_state.ticker:
        st.warning("Please show fundamentals first before forecasting revenue.")
    else:
        ticker_symbol = st.session_state.ticker
        ticker = yf.Ticker(ticker_symbol)

        try:
            fin = ticker.financials
            if fin.empty:
                st.warning("No financial data available for forecasting.")
            else:
                revenue_history = fin.loc["Total Revenue"].dropna()
                if len(revenue_history) < 2:
                    st.warning("Not enough revenue history to forecast.")
                else:
                    df_prophet = pd.DataFrame({
                        'ds': revenue_history.index,
                        'y': revenue_history.values
                    }).reset_index(drop=True)

                    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(df_prophet)

                    future = m.make_future_dataframe(periods=1, freq='Y')
                    forecast = m.predict(future)

                    next_revenue = forecast['yhat'].iloc[-1]
                    st.success(f"Forecasted next period revenue for {ticker_symbol}: ${next_revenue:,.0f}")
        except Exception as e:
            st.error(f"Error forecasting revenue: {e}")
