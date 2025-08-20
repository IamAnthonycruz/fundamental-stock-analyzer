import yfinance as yf
import pandas as pd

# Define the list of tickers to fetch
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "ORCL", "INTC", "CSCO", "IBM",
    "ADBE", "CRM", "TXN", "QCOM", "AVGO", "SAP", "NOW", "TEAM", "SNPS", "AMD",
    "INTU", "AMAT", "MU", "KLAC", "LRCX", "ADI", "ANSS", "PANW", "FTNT", "CDNS",
    "SNOW", "MDB", "DOCU", "DDOG", "OKTA", "CRWD", "NET", "SPLK", "TWLO", "WDAY",
    "PLTR", "FSLY", "ZS", "COUP", "COHR", "CDW", "AKAM", "NFLX", "ZBRA", "FFIV"
]

data = []

# Step 1: Collect base data
for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        pe = info.get("trailingPE", None)
        price = info.get("currentPrice", None)
        eps = info.get("trailingEps", None)

        # Build row (only essential columns for fair value calculation)
        row = {
            "Ticker": ticker,
            "TrailingPE": pe,
            "CurrentPrice": price,
            "TrailingEPS": eps,
        }

        if all(value is not None for value in row.values()):
            data.append(row)
        else:
            print(f"Skipping {ticker} due to missing values: {row}")

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        continue

# Step 2: Create DataFrame
df = pd.DataFrame(data)
df.set_index("Ticker", inplace=True)

# Step 3: Calculate mean PE
mean_pe = df["TrailingPE"].mean()
print(f"Mean PE across dataset: {mean_pe:.2f}")

# Step 4: Calculate Fair Market Value = EPS * mean(PE)
df["FairMarketValue"] = df["TrailingEPS"] * mean_pe

# Step 5: Calculate Over/Under Ratio and Valuation
df["OverUnderRatio"] = df["CurrentPrice"] / df["FairMarketValue"]
df["Valuation"] = df["OverUnderRatio"].apply(
    lambda x: "Over Valued" if x > 1 else ("Fair Valued" if x == 1 else "Under Valued")
)
df["ValuePercentage"] = abs((df["OverUnderRatio"] - 1) * 100)

print(df)

# Save to CSV
df.to_csv("stocks_enhanced.csv")
print("CSV file 'stocks_enhanced.csv' created successfully with calculated fair market values.")
