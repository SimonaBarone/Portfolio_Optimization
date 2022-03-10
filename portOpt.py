import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as pdr
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


class AMPT:
    def __init__(self, tickers, dates_backtesting, weights='equal'):
        start_date, end_date = dates_backtesting.split(":")  # insert dates like "20180101:20220303"
        start_date = pd.Timestamp(start_date).date()

        if end_date:
            end_date = pd.Timestamp(end_date).date()
        else:
            end_date = pd.Timestamp("today").date()

        print(start_date, end_date)

        print("Loading data...")
        self.df = pdr.get_data_yahoo(tickers, start_date, end_date)
        print("Done!")
        self.df = self.df['Adj Close']
        self.clean_data()
        if weights == 'equal':
            self.weights = np.ones(len(tickers)) / len(tickers)
        else:
            self.weights = weights

    def clean_data(self):
        self.df = self.df.dropna()


    def equally_weights_statistics(self):
        log_returns = np.log(self.df / self.df.shift())
        cov_matrix_annual = log_returns.cov() * 252
        portfolio_var = np.dot(self.weights.T, np.dot(cov_matrix_annual, self.weights))
        portfolio_std = np.sqrt(portfolio_var)
        portfolio_returns = np.sum(log_returns.mean() * self.weights) * 252
        percent_var = str(round(portfolio_var, 3) * 100) + '%'
        percent_std = str(round(portfolio_std, 3) * 100) + '%'
        percent_returns = str(round(portfolio_returns, 3) * 100) + '%'
        print('Expected annual return: ' + percent_returns)
        print('Annual std/risk: ' + percent_std)
        print('Annual variance: ' + percent_var)



    def monte_carlo(self):
        n = 5000
        self.weights = np.zeros((n, 10))
        exp_returns = np.zeros(n)
        exp_std = np.zeros(n)
        exp_sharpe_ratios = np.zeros(n)
        log_returns = np.log(self.df / self.df.shift())

        for i in range(n):
            weight = np.random.random(10)
            weight /= weight.sum()
            self.weights[i] = weight
            exp_returns[i] = np.sum(log_returns.mean() * weight) * 252
            exp_std[i] = np.sqrt(np.dot(weight.T, np.dot(log_returns.cov() * 252, weight)))
            exp_sharpe_ratios[i] = exp_returns[i] / exp_std[i]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(exp_std, exp_returns, c=exp_sharpe_ratios)
        ax.set_xlabel('Expected Volatility')
        ax.set_ylabel('Expected Return')



    def opt_cov(self):
        S = risk_models.sample_cov(self.df)
        S = plotting.plot_covariance(S, plot_correlation=True)
        return S


    def opt_return(self):
        mu = expected_returns.mean_historical_return(self.df)
        print(mu)


    def opt_weights(self):
        mu = expected_returns.mean_historical_return(self.df)
        S = risk_models.sample_cov(self.df)
        efficient_front = EfficientFrontier(mu, S)
        comfl_index = efficient_front.tickers.index("COMF.L")
        efficient_front.add_constraint(lambda w: w[comfl_index] <= 0.5)
        efficient_front.max_sharpe()
        opt_weights = efficient_front.clean_weights()
        print(opt_weights)


    def opt_port_statistics(self):
        mu = expected_returns.mean_historical_return(self.df)
        S = risk_models.sample_cov(self.df)
        efficient_front = EfficientFrontier(mu, S)
        efficient_front.max_sharpe()
        print(efficient_front.portfolio_performance(verbose=True))


    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        mu = expected_returns.mean_historical_return(self.df)
        S = risk_models.sample_cov(self.df)

        efficient_front = EfficientFrontier(mu, S)
        plotting.plot_efficient_frontier(efficient_front, ax=ax, show_assets=False)

        efficient_front = EfficientFrontier(mu, S)
        comfl_index = efficient_front.tickers.index("COMF.L")
        efficient_front.add_constraint(lambda w: w[comfl_index] <= 0.5)

        efficient_front.max_sharpe()
        weights = efficient_front.clean_weights()
        ret_tangent, std_tangent, _ = efficient_front.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label='Max Sharpe')

        ax.set_title("Efficient Frontier")
        ax.legend()
        plt.tight_layout()
        plt.show()


    def discrete_allocation(self):
        latest_prices = get_latest_prices(self.df)

        money= int(input("Insert your funds: €"))

        mu = expected_returns.mean_historical_return(self.df)
        S = risk_models.sample_cov(self.df)
        efficient_front = EfficientFrontier(mu, S)
        comfl_index = efficient_front.tickers.index("COMF.L")
        efficient_front.add_constraint(lambda w: w[comfl_index] <= 0.5)
        efficient_front.max_sharpe()
        opt_weights = efficient_front.clean_weights()

        discrete_all = DiscreteAllocation(opt_weights, latest_prices, total_portfolio_value=money)

        allocation, money_left = discrete_all.lp_portfolio()
        print('Optimal allocation:' + str(allocation))
        print('Funds left: ' + '€' + str(round(money_left, 3)))
        
       
