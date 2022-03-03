
# Portfolio-Optimization
This is a project for my University course of **Computer Science with Python**, provided by University of Turin, MSC in Quantitative Finance and Insurance.

In this project I want to show how to **optimize a portfolio of ETFs**, using my own ETFs portfolio.

First I create an equally weighted portfolio and compute some statistics to show this portfolio is actually not efficient.

Then I simulate N portfolios to plot the opportunity set and compute the efficient frontier according to the **Markowitz mean-variance criterion**. 

Then I compute the tangency portfolio by maximizing the Sharpe Ratio with respect to the portfolio weights. What I obtained is the optimal allocation of the ETF that maximize the Sharpe Ratio i.e. I'm maximizing the performance of the portfolio, choosing the more profitable per unit of risk.

At the end I computed the portfolio discrete allocation, i.e. the number of quotes of each ETFs should I earn at the current price.
