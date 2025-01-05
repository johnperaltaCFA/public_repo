from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd

class InvestmentUniverse:
    def __init__(self, tickers=None):
        if tickers is None:
            tickers = []
        self.tickers = tickers

        self.date_of_data = datetime.now()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)
        self.close_prices = pd.DataFrame()
        for ticker in self.tickers:
            data = yf.download(ticker, start=start_date, end=end_date)
            self.close_prices[ticker] = data["Close"]
        
        years = (pd.to_datetime(self.close_prices.index[-1]) - pd.to_datetime(self.close_prices.index[0])).days / 365

        geometric_returns = (self.close_prices.iloc[-1] / self.close_prices.iloc[0]) ** (1 / years) - 1

        self.geometric_returns_df = pd.DataFrame(
            {
            "Ticker": self.tickers,
            "Annualized Geometric Return": geometric_returns.values,
            }
        )

        self.daily_returns = self.close_prices.pct_change().dropna()

        self.daily_std = self.daily_returns.std()
        annualized_std = self.daily_std * (252**0.5)

        self.std_df = pd.DataFrame(
            {
                "Ticker": self.tickers,
                "Annualized Standard Deviation": annualized_std.values,
            }
        )

        self.combined_df = pd.merge(self.geometric_returns_df, self.std_df, on="Ticker")

        self.cova_matrix = self.daily_returns.cov()
        self.correlation_matrix = self.daily_returns.corr()


    def show_date_of_data(self):
        print(self.date_of_data)


    def show_tickers(self):
        print(self.tickers)    


    def show_close_prices(self):
        print(self.close_prices)


    def show_daily_returns(self):
        print(self.daily_returns)


    def show_geometric_returns(self):
        print(self.geometric_returns_df)


    def show_annualized_std(self):
        print(self.std_df)

    
    def show_annualized_returns_and_std(self):
        print(self.combined_df)


    def show_covariance_matrix(self):
        print(self.cova_matrix)


    def show_correlation_matrix(self):
        print(self.correlation_matrix)


    def plot_assets(self):
        df = self.combined_df
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df["Annualized Standard Deviation"],
            df["Annualized Geometric Return"],
            color="blue",
            alpha=0.7,
        )

        for i, ticker in enumerate(df["Ticker"]):
            plt.text(
                df["Annualized Standard Deviation"][i],
                df["Annualized Geometric Return"][i],
                ticker,
                fontsize=9,
                ha="right",
                va="bottom",
            )

        plt.xlabel("Standard Deviation (Risk)")
        plt.ylabel("Geometric Return")
        plt.title("Risk vs Return for Assets")
        plt.grid(alpha=0.3)
        plt.show()


    def plot_efficient_frontier(self):
        df = self.combined_df
        cov_matrix = self.cova_matrix
        rf_rate = 0.0463
        num_assets = len(df)

        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            weights = np.clip(weights, 0, 1)

            weights_record.append(weights)

            portfolio_return = np.dot(weights, df["Annualized Geometric Return"])

            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            sharpe_ratio = (portfolio_return - rf_rate) / portfolio_risk

            results[0, i] = portfolio_risk
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio    

        max_sharpe_idx = np.argmax(results[2])
        min_risk_idx = np.argmin(results[0])

        max_sharpe_portfolio = results[:, max_sharpe_idx]
        min_risk_portfolio = results[:, min_risk_idx]

        portfolios = np.array([results[0, :], results[1, :]]).T
        portfolios = portfolios[portfolios[:, 0].argsort()]

        efficient_frontier = [portfolios[0]]
        for portfolio in portfolios[1:]:
            if portfolio[1] > efficient_frontier[-1][1]:
                efficient_frontier.append(portfolio)

        efficient_frontier = np.array(efficient_frontier)

        plt.figure(figsize=(12, 8))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap="viridis", alpha=0.5)
        plt.colorbar(label="Sharpe Ratio")
        plt.scatter(
            0,
            rf_rate,
            color="red",
            label="Risk Free Rate",
        )
        plt.scatter(
            max_sharpe_portfolio[0],
            max_sharpe_portfolio[1],
            color="red",
            label="Max Sharpe Ratio",
        )
        plt.scatter(
            min_risk_portfolio[0],
            min_risk_portfolio[1],
            color="red",
            label="Min Risk Portfolio",
        )

        cml_x = [0, max_sharpe_portfolio[0]]
        cml_y = [rf_rate, max_sharpe_portfolio[1]]
        plt.plot(cml_x, cml_y, color="blue", label="Capital Market Line", linewidth=2, linestyle="--")

        plt.plot(efficient_frontier[:, 0], efficient_frontier[:, 1], color="blue", label="Efficient Frontier", linewidth=2, linestyle="--")

        plt.title("Efficient Frontier")
        plt.xlabel("Portfolio Risk (Standard Deviation)")
        plt.ylabel("Portfolio Return")
        plt.legend()
        plt.grid()
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        plt.show()

        max_sharpe_weights = weights_record[max_sharpe_idx]
        min_risk_weights = weights_record[min_risk_idx]

        combined_weights_df = pd.DataFrame(
            {
            "Ticker": df["Ticker"],
            "Max Sharpe Ratio Weight": max_sharpe_weights,
            "Min Risk Weight": min_risk_weights,
            }
        )
        print(combined_weights_df)


sp99_tickers = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
]


tsx60_tickers = [
    "AEM.TO", "AQN.TO", "ATD.TO", "BMO.TO", "BNS.TO", "ABX.TO", "BCE.TO", "BAM.TO", 
    "BN.TO", "BIP.UN.TO", "CAE.TO", "CCO.TO", "CAR.UN.TO", "CM.TO", "CNR.TO", 
    "CNQ.TO", "CP.TO", "CTC.A.TO", "CCL.B.TO", "CVE.TO", "GIB.A.TO", "CSU.TO", 
    "DOL.TO", "EMA.TO", "ENB.TO", "FM.TO", "FSV.TO", "FTS.TO", "FNV.TO", 
    "WN.TO", "GIL.TO", "H.TO", "IMO.TO", "IFC.TO", "K.TO", "L.TO", "MG.TO", 
    "MFC.TO", "MRU.TO", "NA.TO", "NTR.TO", "OTEX.TO", "PPL.TO", "POW.TO", 
    "QSR.TO", "RCI.B.TO", "RY.TO", "SAP.TO", "SHOP.TO", "SLF.TO", "SU.TO", 
    "TRP.TO", "TECK.B.TO", "T.TO", "TRI.TO", "TD.TO", "TOU.TO", "WCN.TO", 
    "WPM.TO", "WSP.TO"
]

tsx5_tickers = [
    "BMO.TO", "ENB.TO", "RY.TO", "TD.TO", "TRP.TO"
]

sp99 = InvestmentUniverse(sp99_tickers)

sp99.show_close_prices()
sp99.show_daily_returns()
sp99.show_annualized_returns_and_std()

tsx5_tickers = InvestmentUniverse(tsx5_tickers)

tsx5_tickers.show_correlation_matrix()
tsx5_tickers.plot_efficient_frontier()

sp99_tickers = InvestmentUniverse(sp99_tickers)

sp99_tickers.plot_efficient_frontier()