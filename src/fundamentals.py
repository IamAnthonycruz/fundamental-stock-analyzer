import yfinance as yf

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



if __name__ == "__main__":
    data = get_fundamentals("AAPL")
    for k, v in data.items():
        print(f"{k}: {v}")
