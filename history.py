"""History module that imports data

This script imports data from a csv file, sets indicies, and manipulates it
using pandas to add columns and will return a pandas dataframe with stock data in it

    * read - loads in data from csv file using pandas and returns a dataframe
    * set_index - updates the index for the dataframe
    * adjust - adds columns into data frame that will be used later
    * ticker_past - fills in columns based on past data for one particular ticker
    * past - runs through dataframe ticker by ticker and adds data based on past data
    * run - uses above functions to load data, update dataframe based on past data, and returns a data frame
"""

import numpy as np
import pandas as pd

def read(csv='./tech.csv', metadata='./metadata.json'):
    """Reads stock data from file and returns a dataframe

    Parameters
    ----------
    csv : string
        CSV file path
    metadata : string
        metadata file path

    Returns
    -------
    Pandas DataFrame
        Dataframe willed with stock data from csv
    """

    metadata = pd.read_json(metadata)
    columns = metadata.datatable['columns']
    headings = [heading['name'] for heading in columns]
    stocks = pd.read_csv(
        filepath_or_buffer=csv, header=None, names=headings)

    return stocks


def set_index(stocks):
    """Sets indices for stock dataframe
    Parameters
    ----------
    stocks : DataFrame
        DataFrame to add indices to
    Returns
    -------
    Pandas DataFrame
        Dataframe with indices set for ticker and date
    """
    stocks.set_index(["ticker", "date"], inplace=True)
    return stocks


def adjust(stocks):
    """Adds in columns for past and historical data to be filled in later
    Parameters
    ----------
    stocks : DataFrame
        DataFrame to add columns in
    Returns
    -------
    Pandas DataFrame
        Dataframe with columns added in
    """

    # remove rows where close, open, or high prices are not available
    # add columns and fill with temp data
    stocks['past_day_gain'] = np.nan
    stocks['past_week_gain'] = np.nan
    stocks['past_month_gain'] = np.nan
    stocks['past_quarter_gain'] = np.nan
    stocks['past_year_gain'] = np.nan

    stocks['close_to_open'] = np.nan
    stocks['high_to_open'] = np.nan
    stocks['low_to_open'] = np.nan

    stocks['high_is_enough'] = np.nan
    stocks['close_is_enough'] = np.nan
    stocks['should_invest'] = np.nan
    return stocks


def ticker_past(ticker):
    """Fills in past data for the particular ticker
    Parameters
    ----------
    ticker : DataFrame
        Dataframe for ticker
    Returns
    -------
    Pandas DataFrame
        ticker Dataframe with past data filled in
    """
    prev_day = ticker.shift(1)
    drop_columns = ['adj_close', 'adj_high', 'adj_low', 'volume', 'adj_volume', 'split_ratio', 'ex-dividend', 'open', 'adj_open', 'past_day_gain',
                    'past_week_gain', 'past_month_gain', 'high_is_enough', 'close_is_enough', 'should_invest'
                    ,'past_quarter_gain', 'past_year_gain'
    ]

    prev_day = prev_day.drop(labels=drop_columns, axis=1)
    prev_day = prev_day.iloc[:, 0:3]
    prev_day.columns = ['prev_high', 'prev_low', 'prev_close']
    frames = [ticker, prev_day]
    new_ticker = pd.concat(frames, axis=1, join='inner')

    new_ticker.past_day_gain = new_ticker['open'].pct_change(1)
    new_ticker.past_week_gain = new_ticker['open'].pct_change(5)
    new_ticker.past_month_gain = new_ticker['open'].pct_change(20)
    new_ticker.past_quarter_gain = new_ticker['open'].pct_change(65)
    new_ticker.past_year_gain = new_ticker['open'].pct_change(261)

    new_ticker.close_to_open = (
        new_ticker.open - new_ticker.prev_close) / new_ticker.prev_close
    new_ticker.high_to_open = (
        new_ticker.open - new_ticker.prev_high) / new_ticker.prev_high
    new_ticker.low_to_open = (
        new_ticker.open - new_ticker.prev_low) / new_ticker.prev_low

    new_ticker.high_is_enough = (
        (new_ticker.high - new_ticker.open) / new_ticker.open) >= 0.025
    new_ticker.should_invest = new_ticker.high_is_enough

    return new_ticker


def past(stocks):
    """Fills in past data for all stocks
    Parameters
    ----------
    stocks : DataFrame
        DataFrame with stock data filled in
    Returns
    -------
    Pandas DataFrame
        Dataframe with past data filled in
    """
    grouped = stocks.groupby(level=0)
    s = grouped.apply(ticker_past)

    # replace all infinite values and NaN with 0
    # values from past_* will be infinite if close happens to be 0 or 0.01
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.fillna(0)
    return s


def run(csv='./tech.csv', metadata='./metadata.json'):
    """Imports the tech stock data and returns a dataframe filled with data
    Returns
    -------
    Pandas DataFrame
        Dataframe filled with stock data with indices set and past data filled
    """
    stocks = read(csv, metadata)
    print("STOCKS READ********************************************")
    stocks = adjust(stocks)
    print("DATAFRAME COLUMNS ADDED********************************")
    stocks = set_index(stocks)
    print("DATAFRAME INDEX SET************************************")
    stocks = past(stocks)
    print("DATAFRAME FILLED OUT***********************************\a\a")
    return stocks
