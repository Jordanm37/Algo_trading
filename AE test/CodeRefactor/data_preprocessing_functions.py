import os
import itertools
import warnings
from datetime import datetime as dt

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


warnings.filterwarnings("ignore")

def generate_data_summary(raw_df):
    # Generate a summary for the raw data
    summary = raw_df.describe(include='all')
    summary.loc['type'] = raw_df.dtypes
    summary.loc['median'] = (raw_df.median() * 10000 // 100 / 100)
    summary.loc['mean'] = (raw_df.mean() * 10000 // 100 / 100)
    summary.loc['mode'] = raw_df.mode().iloc[0]
    summary.loc['missing'] = (
        (100.0 * raw_df.isna().sum() / len(raw_df)).round(2).astype(str) + '%')
    summary.loc['unique'] = raw_df.nunique()
    summary.loc['min'] = raw_df.min()
    summary.loc['max'] = raw_df.max()

    columns_order = ['type', 'count', 'missing', 'unique', 'min',
                     '25%', '50%', '75%', 'max', 'mean', 'median', 'std', 'mode']
    summary = summary.loc[columns_order].round(2)
    return summary


def find_missing_ranges(series):
    # Find the ranges of missing values in the series
    is_missing = series.isna()
    changes = is_missing.ne(is_missing.shift()).cumsum()
    missing_ranges = is_missing.groupby(changes).apply(
        lambda x: (x.index[0], x.index[-1]))
    return missing_ranges


def plot_time_series(ts1, ts2, ts1_label='Time Series 1', ts2_label='Time Series 2', figsize=(12, 10)):
    # Plot two time series on separate subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    # Plot the first time series
    axes[0].plot(ts1)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].set_title(ts1_label)

    # Plot the second time series
    axes[1].plot(ts2)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')
    axes[1].set_title(ts2_label)

    # Show the plots
    plt.show()
    save_plot('ts1_ts2', f'Series Plots')


def save_plot(title, plot_name, fig=None):
    if not fig:
        fig = plt.gcf()

    # Create the directory if it doesn't exist
    if not os.path.exists('Pre_processing'):
        os.makedirs('Pre_processing')

    # Create the subdirectory for the series if it doesn't exist
    series_path = f'Pre_processing/{title}'
    if not os.path.exists(series_path):
        os.makedirs(series_path)

    # Save the plot in the corresponding subdirectory with the plot_name in the file name
    fig.savefig(f'{series_path}/{plot_name}.png')
    plt.close()


def count_missing_values(series):
    return series.isna().sum()


def max_missing_range(series):
    mask = series.isna()
    changes = mask.ne(mask.shift()).cumsum()
    counts = mask.groupby(changes).sum()
    counts = counts[counts > 1]
    return 0 if counts.empty else counts.max()


def intervals_to_days(intervals):
    minutes = intervals * 2
    hours = minutes / 60
    days = hours / 24
    return days


def count_days_frequency(series):
    series = series.dropna()
    day_of_week_series = pd.Series(series.index.dayofweek, index=series.index)
    day_counts = day_of_week_series.value_counts().sort_index()
    day_counts.index = pd.to_datetime(
        day_counts.index, unit='D', origin=pd.Timestamp('2000-01-03')).day_name()
    return day_counts


def plot_streak_histogram(ts, title):
    isnull = ts.isnull()
    groups = isnull.ne(isnull.shift()).cumsum()
    streak_lengths = isnull.groupby(groups).sum()

    bins = pd.cut(streak_lengths, bins=[0, 1, 2, 3, 4, 5, 10, 100, 1000, streak_lengths.max()],
                  labels=[str(x*2) + ' mins' for x in range(1, 6)] + ['22 mins', '4 hours', '1.3 days', '4+ days'])
    plt.figure(figsize=(8, 7))
    # plot histogram of binned data
    ax = bins.value_counts().sort_index().plot(kind='bar', edgecolor='black')
    plt.title(title)
    # add values to bars
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() /
                    2, p.get_height()), ha='center', va='bottom')
    # show the plot
    save_plot(title, f'Missing Values Streaks Histogram_{title}')


def calculate_time_gaps(ts):
    isnull = ts.isnull()
    groups = isnull.ne(isnull.shift()).cumsum()
    streak_lengths = isnull.groupby(groups).sum()
    time_gaps = pd.to_timedelta(
        streak_lengths[streak_lengths > 0] * 2, unit='m')
    return time_gaps


def time_gaps_stats(series):
    return (series.dt.total_seconds() / 60).describe()


def rolling_mean(series, window=120):
    # Check if there are any missing values in the input series
    if series.isnull().any():
        # Create a copy of the input series for imputation
        imputed_series = series.copy()

        # Identify non-missing value blocks in the input series
        non_missing_blocks = series.notna().astype(int).groupby(
            series.isna().astype(int).cumsum()).cumsum()

        # Iterate through each non-missing value block
        for _, block in non_missing_blocks.groupby(non_missing_blocks):
            start_idx = block.index[0]
            end_idx = block.index[-1] + pd.Timedelta(minutes=2)

            # Calculate rolling mean for the block
            rolling_mean_block = series[start_idx:end_idx].rolling(
                window=window, min_periods=1).mean()

            # Impute missing values in the block with rolling mean
            imputed_series[start_idx:end_idx] = series[start_idx:end_idx].fillna(
                rolling_mean_block)

        # Fill any remaining missing values using backfill method
        imputed_series.fillna(method='bfill', inplace=True)
    else:
        # If there are no missing values, return the original series
        imputed_series = series
    return imputed_series


def rolling_median(series, window=120):
    # Check if there are any missing values in the input series
    if series.isnull().any():
        # Create a copy of the input series for imputation
        imputed_series = series.copy()

        # Identify non-missing value blocks in the input series
        non_missing_blocks = series.notna().astype(int).groupby(
            series.isna().astype(int).cumsum()).cumsum()

        # Iterate through each non-missing value block
        for _, block in non_missing_blocks.groupby(non_missing_blocks):
            start_idx = block.index[0]
            end_idx = block.index[-1] + pd.Timedelta(minutes=2)

            # Calculate rolling mean for the block
            rolling_mean_block = series[start_idx:end_idx].rolling(
                window=window, min_periods=1).median()

            # Impute missing values in the block with rolling mean
            imputed_series[start_idx:end_idx] = series[start_idx:end_idx].fillna(
                rolling_mean_block)

        # Fill any remaining missing values using backfill method
        imputed_series.fillna(method='bfill', inplace=True)
    else:
        # If there are no missing values, return the original series
        imputed_series = series
    return imputed_series


def impute_missing_values_mean(data, short_gap_threshold=120, long_gap_threshold=2880):
    # Identify the size of missing value gaps in the input data
    missing_gaps = data.isna().astype(int).groupby(
        data.notna().astype(int).cumsum()).cumsum()

    # Impute missing values for short gaps
    data_short_gaps = data.copy()
    data_short_gaps[missing_gaps <=
                    short_gap_threshold] = rolling_mean(data_short_gaps)

    # Impute missing values for long gaps
    data_long_gaps = data.copy()
    data_long_gaps[missing_gaps > short_gap_threshold] = rolling_mean(
        data_long_gaps, long_gap_threshold)

    # Combine the short and long gap imputed data
    data_imputed = data_short_gaps.combine_first(data_long_gaps)

    # Fill any remaining missing values using backfill and forward fill methods
    data_imputed.fillna(method='bfill', inplace=True)
    data_imputed.fillna(method='ffill', inplace=True)

    return data_imputed


def impute_missing_values_median(data, short_gap_threshold=120, long_gap_threshold=2880):
    # Identify the size of missing value gaps in the input data
    missing_gaps = data.isna().astype(int).groupby(
        data.notna().astype(int).cumsum()).cumsum()

    # Impute missing values for short gaps
    data_short_gaps = data.copy()
    data_short_gaps[missing_gaps <=
                    short_gap_threshold] = rolling_median(data_short_gaps)

    # Impute missing values for long gaps
    data_long_gaps = data.copy()
    data_long_gaps[missing_gaps > short_gap_threshold] = rolling_median(
        data_long_gaps, long_gap_threshold)

    # Combine the short and long gap imputed data
    data_imputed = data_short_gaps.combine_first(data_long_gaps)

    # Fill any remaining missing values using backfill and forward fill methods
    data_imputed.fillna(method='bfill', inplace=True)
    data_imputed.fillna(method='ffill', inplace=True)

    return data_imputed


def impute_missing_values_lin(data, short_gap_threshold=120):
    # Identify the size of missing value gaps in the input data
    missing_gaps = data.isna().astype(int).groupby(
        data.notna().astype(int).cumsum()).cumsum()

    # Impute missing values for short gaps using linear interpolation
    data_short_gaps = data.copy()
    data_short_gaps[missing_gaps <= short_gap_threshold] = linear_interpolation(
        data_short_gaps)

    # Impute missing values for long gaps using linear interpolation
    data_long_gaps = data.copy()
    data_long_gaps[missing_gaps >
                   short_gap_threshold] = linear_interpolation(data_long_gaps)

    # Combine the short and long gap imputed data
    data_imputed = data_short_gaps.combine_first(data_long_gaps)

    # Fill any remaining missing values using backfill and forward fill methods
    data_imputed.fillna(method='bfill', inplace=True)
    data_imputed.fillna(method='ffill', inplace=True)

    return data_imputed


def linear_interpolation(series):
    # Check if there are any missing values in the input series
    if series.isnull().any():
        # Create a copy of the input series for imputation
        imputed_series = series.copy()

        # Identify non-missing value blocks in the input series
        non_missing_blocks = series.notna().astype(int).groupby(
            series.isna().astype(int).cumsum()).cumsum()

        # Iterate through each non-missing value block
        for _, block in non_missing_blocks.groupby(non_missing_blocks):
            start_idx = block.index[0]
            end_idx = block.index[-1] + pd.Timedelta(minutes=2)

            # Impute missing values in the block using linear interpolation
            imputed_series[start_idx:end_idx] = series[start_idx:end_idx].interpolate(
                method='linear')

        # Fill any remaining missing values using backfill method
        imputed_series.fillna(method='bfill', inplace=True)
    else:
        # If there are no missing values, return the original series
        imputed_series = series
    return imputed_series


def load_data(filename):
    raw_df = pd.read_csv(filename, header=None)
    raw_df.columns = ['datetime', 'ts1', 'ts2']
    raw_df['datetime'] = pd.to_datetime(
        raw_df['datetime']-719529, unit='d').round('s')
    raw_df.set_index('datetime', inplace=True)
    return raw_df


def generate_data_summary(df):
    print(df.describe())


def analyze_time_gaps(df):
    time_gaps = df.index.to_series().diff().dropna()
    time_gap_stats = time_gaps.describe()
    print('Time gaps:\n')
    print(time_gap_stats)
    outlier_threshold = pd.Timedelta(minutes=10)
    outliers = time_gaps[time_gaps > outlier_threshold]
    plt.hist(time_gaps.dt.total_seconds() / 60, bins=50)
    plt.xlabel('Time Gap (minutes)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Gaps')
    plt.show()


def remove_weekends(df):
    # Remove rows corresponding to weekends (Saturday and Sunday)
    return df[df.index.dayofweek != 5]


def time_gaps_stats(s):
    return s.describe()


def count_days_frequency_resampled(ts):
    day_counts = count_days_frequency(ts.resample('D').mean())
    return day_counts


def analyze_time_gaps(df):
    time_gaps = df.index.to_series().diff().dropna()
    time_gap_stats = time_gaps.describe()
    print('Time gaps:\n')
    print(time_gap_stats)
    # plot_time_gaps_histogram(time_gaps)


def plot_time_gaps_histogram(time_gaps):
    outlier_threshold = pd.Timedelta(minutes=10)
    outliers = time_gaps[time_gaps > outlier_threshold]
    plt.hist(time_gaps.dt.total_seconds() / 60, bins=50)
    plt.xlabel('Time Gap (minutes)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Gaps')
    plt.show()


def count_days_frequency_resampled(ts):
    return count_days_frequency(ts.resample('D').mean())

def analyze_missing_values(ts, ts_name):
    # '''Input: ts - time series pandas series
    #             ts_name - name of the time series
    # '''
    missing_ts = count_missing_values(ts)
    print(f"Missing values in {ts_name}: {missing_ts}")

    max_ts = max_missing_range(ts)
    print(f"Max missing range in {ts_name}: {max_ts}")

    days_ts = intervals_to_days(max_ts)
    print(f"Max missing range in {ts_name}: {days_ts:.2f} days")