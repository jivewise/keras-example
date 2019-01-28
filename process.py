"""Process module that processes data for neural network consumption

This script takes a DataFrame and updates it by scaling the data and spliting
it into training and test sets.

    * pick - picks a specific stock to work on
    * prep - preps the data by dropping extraneous columns and parsing out the expected output
    * split - splits up the data into training and test sets
    * run - uses above functions to prep, split, and scale data
"""

import history
import numpy as np

from dataset import DataSet
from sklearn.model_selection import train_test_split

def pick(stocks, tickers):
    """Picks a specific stock to work on
    """
    query_arr = list(map(lambda tick: "ticker == '%s'" % (tick), tickers))
    query_str = " or ".join(query_arr)
    picked = stocks.query(query_str)
    return picked

def prep(stocks, predict_value):
    """Extracts features from out dataframe, and puts them into X and y
    """
    stocks = stocks.reset_index(level='date', drop=True)
    stocks = stocks.reset_index(level='ticker')

    stocks.dropna(subset=['close', 'open', 'high', 'prev_close',
                          'adj_open', 'past_day_gain', 'past_week_gain'])

    drop_columns = [
        'close', 'adj_close', 'low', 'adj_high', 'adj_low', 'should_invest',
        'volume', 'split_ratio', 'ex-dividend', 'adj_open', 'high_to_open',
        'high_is_enough', 'close_is_enough', 'past_month_gain', 'low_to_open',
        'past_day_gain',
    ] if predict_value else [
        'close', 'adj_close', 'high', 'low', 'adj_high', 'adj_low', 'high_is_enough',
        'close_is_enough', 'volume', 'adj_volume', 'split_ratio', 'ex-dividend', 'open', 'adj_open',
        'prev_high', 'prev_close', 'prev_low', 'past_month_gain', 'low_to_open', 'should_invest',
    ]

    print("Dropping columns:*******************************")
    print(",".join(drop_columns))
    stocks = stocks.loc[stocks.past_year_gain != 0]

    high = stocks.pop('high') if predict_value else False
    y = high if predict_value else stocks.should_invest

    stocks = stocks.drop(labels=drop_columns, axis=1)
    print("Remaining columns:*******************************")
    print(stocks.columns.values)

    X = stocks.iloc[:, 1:]
    print("Using columns:*******************************")
    print(",".join(X.columns))

    return X, y


def split(X, y):
    """Splits data into training and test sets
    """
    test_size = 0.25
    print("Test data split: %s ***********************" % test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    data_obj = DataSet(X_train, X_test, y_train, y_test)
    return data_obj

def run(stocks=None, ticker='MSFT', predict_value=False):
    """Processes stock data for neural network consumption
    Parameters
    ----------
    stocks : DataFrame
        DataFrame with stock data filled in
    tickers: String
        Ticker symbol we want to predict
    predict_values: Boolean
        Set to true if you want to predict the value of the stock or false if you want to categorize it
    Returns
    -------
    DataSet
        DataSet object filled with scaled and split data ready for consumption
    """

    stocks = history.run() if stocks is None else stocks

    tech = pick(stocks, [ticker])
    X, y = prep(tech, predict_value)
    data = split(X, y)

    data.scale(predict_value)

    return data
