"""
Created on Mon Nov 13 2017

@author: Osama Iqbal

Code uses Python 2.7, packaged with Anaconda 4.4.0
Code developed on Windows 10 OS.

Project 1:
* Download JP Morgan stock historical prices from an appropriate financial website such as Google Finance, Yahoo Finance, Quandl, CityFALCON, or another similar source
Period: April 1, 2015 - June 25 2015
Price considered in the analysis: Close price adjusted for dividends and splits

1. Calculate Average stock value
2. Calculate Stock volatility
3. Calculate Daily stock return

4. Implement a two-variable regression
* Explained variable: JP Morgan stock (close price)
* Explanatory variable: S&P500
* Period: April 1, 2015 - June 25 2015
"""
# Some Metadata about the script
__author__ = 'Osama Iqbal (iqbal.osama@icloud.com)'
__license__ = 'MIT'
__vcs_id__ = '$Id$'
__version__ = '1.0.0'  # Versioning: http://www.python.org/dev/peps/pep-0386/

import logging  # Logging class for logging in the case of an error, makes debugging easier
import sys  # For gracefully notifying whether the script has ended or not
from pandas_datareader import data as pdr  # The pandas Data Module used for fetching data from a Data Source
import warnings  # For removing Deprication Warning w.r.t. Yahoo Finance Fix
import math
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fix_yahoo_finance import pdr_override  # For overriding Pandas DataFrame Reader not connecting to YF


def yahoo_finance_bridge():
    """
    This function fixes problems w.r.t. fetching data from Yahoo Finance
    :return: None
    """
    logging.info('Correcting Yahoo Finance')
    pdr_override()


def main():
    """
    This function is called from the main block. The purpose of this function is to contain all the calls to
    business logic functions
    :return: int - Return 0 or 1, which is used as the exist code, depending on successful or erroneous flow
    """
    # Wrap in a try block so that we catch any exceptions thrown by other functions and return a 1 for graceful exit
    try:
        # ===== Step 0: Sanitation =====
        # Fix Pandas Datareader's Issues with Yahoo Finance (Since yahoo abandoned it's API)
        yahoo_finance_bridge()

        # ===== Step 1: Get JPM Data for April 1st 2015 to June 25th 2015 =====
        data = pdr.get_data_yahoo('JPM', start='2015-04-01', end='2015-06-25', auto_adjust=True)

        # ===== Step 2: Calculate Average Stock Value =====
        print('Average Stock Value of JPM: %s' % str(data['Close'].mean()))

        # ===== Step 3: Calculate Stock Daily Stock Return =====
        stock_return = data['Close'].pct_change()
        print('Daily Returns of JPM: %s' % str(stock_return))

        # ===== Step 4: Calculate Stock Volatility =====
        stock_vol = stock_return.std()
        print('Stock Volatility of JPM: %s' % str(stock_vol))
        print('Annualized Stock Volatility of JPM: %s' % str(stock_vol * math.sqrt(252)))

        # ===== Step 5: Perform Linear Regression =====
        # Get SnP500 data for the period
        data_snp = pdr.get_data_yahoo('^GSPC', start='2015-04-01', end='2015-06-25', auto_adjust=True)
        y = np.reshape(data['Close'], (len(data['Close']), 1))
        x = np.reshape(data_snp['Close'], (len(data_snp['Close']), 1))

        regr = linear_model.LinearRegression()
        regr.fit(x, y)

        y_predict = regr.predict(x)

        print('Coefficients: %s' % str(regr.coef_))
        print('Mean Squared Error: %s' % str(mean_squared_error(y, y_predict)))
        print('Variance Score: %s' % str(r2_score(y, y_predict)))

        # Plot outputs
        plt.scatter(x, y, color='black')
        plt.plot(x, y_predict, color='blue')
        plt.xlabel('SnP 500 Close')
        plt.ylabel('JPM Close')
        plt.title('Linear Regression: SnP500 and JPM CLose Prices')
        plt.xticks(())
        plt.yticks(())

        plt.show()
        pass
        print()

    except BaseException as e:
        # Casting a wide net to catch all exceptions
        print('\n%s' % str(e))
        return 1


# Main block of the program. The program begins execution from this block when called from a cmd
if __name__ == '__main__':
    # Initialize Logger
    logging.basicConfig(format='%(asctime)s %(message)s: ')
    logging.info('Application Started')
    exit_code = main()
    logging.info('Application Ended')
    sys.exit(exit_code)
