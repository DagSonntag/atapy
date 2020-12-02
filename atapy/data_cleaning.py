import pandas as pd
import datetime
import logging
from atapy.interval import Interval
from typing import Dict

logger = logging.getLogger()

"""
Various data cleaning and preprocessing methods used by the data connections to clean data
"""


def get_liquid_hours(liquid_hours: str) -> (datetime.time, datetime.time):
    """ Parses a liquid_hours string (HH:mm-HH:mm) into a start datetime.time and end datetime.time"""
    if liquid_hours is None or len(liquid_hours) != 9:
        raise NotImplementedError(
            "The liquid hours are given in an unexpected format, please investigate: {}".format(liquid_hours))
    start_trade_time, end_trade_time = liquid_hours.split("-")
    start_trade_time = datetime.time(int(start_trade_time[0:2]), int(start_trade_time[2:]))
    if end_trade_time == '2400':
        end_trade_time = datetime.time.max
    else:
        end_trade_time = datetime.time(int(end_trade_time[0:2]), int(end_trade_time[2:]))
    return start_trade_time, end_trade_time


def add_start_and_end_time(data_df: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    """Adds the start and endtime, as well as conversts the date column to datetime.date format"""
    df = data_df.copy()
    # Convert the date column
    df['start_time'] = pd.to_datetime(df.date)
    df['date'] = df.start_time.dt.date
    df['end_time'] = df.start_time + datetime.timedelta(minutes=interval.in_minutes())
    return df


def filter_out_non_traded_hours(data_df: pd.DataFrame, start_trade_time: datetime.time,
                                end_trade_time: datetime.time) -> pd.DataFrame:
    """Filters out any samples that are outside the given trading hours"""
    return data_df[(data_df.start_time.dt.time >= start_trade_time)
                   & (data_df.end_time.dt.time <= end_trade_time)]


def fill_missing_intervals(intraday_df: pd.DataFrame, interval: Interval, traded_dates: pd.Series,
                           start_daily_trade_time: datetime.time, end_daily_trade_time: datetime.time) -> pd.DataFrame:
    """
    Fills in missing intervals (if there are any) for an intraday_df. Will also create 'fake' samples if for the given
    traded days if all samples are missing for these days, and they are in between the first and last sample
    """

    end_of_day_samples = intraday_df.groupby('date').tail(1).copy()
    end_of_day_samples['next_trading_date'] = list(end_of_day_samples['date'].iloc[1:]) + [None]

    def fill_missing_intervals_per_day(current_date: datetime.date, single_day_df: pd.DataFrame,
                                       fill_initial, fill_tail):
        expected_start_times = pd.Series(pd.date_range(
            start=datetime.datetime.combine(current_date, start_daily_trade_time) if fill_initial else
            single_day_df.start_time.iloc[0],
            end=datetime.datetime.combine(current_date, end_daily_trade_time) if fill_tail else
            single_day_df.end_time.iloc[-1],
            freq="{}s".format(interval.in_seconds()),
            closed='left',
            name='start_time'))
        # If some initial slots are missing
        if fill_initial and single_day_df.start_time.min() != expected_start_times[0]:
            # Get the closing price from previously day
            prev_closing_price = end_of_day_samples[end_of_day_samples.next_trading_date == current_date].close
            if prev_closing_price.shape[0] != 1:
                # Can occur if the first sample is missing, in that case, fill with None to simplify usage.
                # Should not happen since initial date should not have _fill_initial=True but still
                prev_closing_price = None
            else:
                prev_closing_price = prev_closing_price.values[0]
            initial_df = pd.DataFrame({
                'open': prev_closing_price,
                'high': prev_closing_price,
                'low': prev_closing_price,
                'close': prev_closing_price,
                'volume': 0,
                'average': prev_closing_price,
                'nr_of_trades': 0,
                'start_time': expected_start_times[expected_start_times < single_day_df.start_time.min()] if
                single_day_df.shape[0] > 0 else expected_start_times,
                'end_time': (expected_start_times[expected_start_times < single_day_df.start_time.min()] if
                             single_day_df.shape[0] > 0 else expected_start_times) + datetime.timedelta(
                    minutes=interval.in_minutes()),
                'real_sample': False})
            a = single_day_df.shape[0]
            single_day_df = pd.concat([initial_df, single_day_df])
        if single_day_df.shape[0] < expected_start_times.shape[0]:
            single_day_df = pd.merge(single_day_df, pd.DataFrame(expected_start_times), on='start_time', how='right')
            single_day_df[['open', 'high', 'low', 'close', 'average']] = single_day_df[
                ['open', 'high', 'low', 'close', 'average']].fillna(
                method='ffill')
            single_day_df[['volume', 'nr_of_trades']] = single_day_df[['volume', 'nr_of_trades']].fillna(0)
            single_day_df[['real_sample']] = single_day_df[['real_sample']].fillna(False)
            single_day_df['end_time'] = single_day_df.start_time + datetime.timedelta(minutes=interval.in_minutes())
        # Set the index right
        single_day_df.index = [current_date] * single_day_df.shape[0]
        return single_day_df

    intraday_df['real_sample'] = True  # Indicate which samples are real and which are added to fill missing slots
    # Only include the dates within the given date-range
    traded_dates = traded_dates[(traded_dates >= intraday_df.date.iloc[0])
                                & (traded_dates <= intraday_df.date.iloc[-1])]
    # Create a temp df to index over the dates, this will significantly speed up the process. Note that the groupby must
    # be done on the daily data since there may be entire dates where no trades have taken place (and hence don't have
    # five_min data)
    temp_df = intraday_df.set_index('date').sort_index()
    filled_temp_df = traded_dates.reset_index(drop=True).groupby(level=0, group_keys=False).apply(
        lambda current_date: fill_missing_intervals_per_day(
            current_date=current_date.iloc[0], single_day_df=temp_df.loc[[current_date.iloc[0]]],
            fill_initial=current_date.iloc[0] != traded_dates.iloc[0],
            fill_tail=current_date.iloc[0] != traded_dates.iloc[-1]))
    filled_temp_df.index.name = 'date'
    filled_temp_df = filled_temp_df.sort_values('start_time')
    return filled_temp_df.reset_index()[intraday_df.columns]


def aggregate_intraday_data(intraday_df: pd.DataFrame, interval: Interval) -> Dict[Interval, pd.DataFrame]:
    """Aggregates up the data from intraday data to higher level intervals. Very time-consuming!"""

    # Aggregate the data to the other time intervals. Done column wise (and not on the entire dataframe at once) since
    # this is significantly faster
    def aggregate_df(interval_keys, df):
        temp_df = pd.DataFrame(
            {'open': df.groupby(interval_keys).open.apply(lambda x: x.iloc[0]).values,
             'high': df.groupby(interval_keys).high.max().values,
             'low': df.groupby(interval_keys).low.min().values,
             'close': df.groupby(interval_keys).close.apply(lambda x: x.iloc[-1]).values,
             'volume': df.groupby(interval_keys).volume.sum().values,
             'nr_of_trades': df.groupby(interval_keys).nr_of_trades.sum().values,
             'average': df.groupby(interval_keys)[['volume', 'average']].apply(
                              lambda x: sum(x.volume * x.average) / sum(x.volume) if sum(x.volume) > 0 else
                              x.average.iloc[0]).values,
             'start_time': df.groupby(interval_keys).start_time.min().values,
             'end_time': df.groupby(interval_keys).end_time.max().values,
             'real_sample': df.groupby(interval_keys).real_sample.any().values})
        temp_df['date'] = temp_df.start_time.dt.date
        return temp_df.reset_index(drop=True)

    aggregated_dfs = {interval: intraday_df}
    for aggregation_interval in Interval:
        if aggregation_interval > interval:
            dat = aggregate_df(
                aggregation_interval.datetime_identifiers(intraday_df.start_time), intraday_df)
            aggregated_dfs[aggregation_interval] = dat.sort_values('start_time')
    return aggregated_dfs


def add_time_based_info(data_df: pd.DataFrame) -> pd.DataFrame:
    market_open_dates = data_df.start_time.dt.date.unique()
    data_df['next_day_open'] = (data_df.date + datetime.timedelta(days=1)).isin(market_open_dates)

    data_df['day_of_week'] = data_df.start_time.dt.dayofweek
    data_df['day_name'] = data_df.start_time.dt.day_name().astype('category')
    data_df['slot_in_day'] = data_df.groupby('date').cumcount()
    data_df['remaining_slots_in_day'] = data_df.groupby('date').cumcount(ascending=False)

    # Add info on which day in the month the slot is in (and how many remaining trade days that exists) as well as which
    # month it is
    month_data = pd.DataFrame(pd.to_datetime(data_df.start_time.dt.date.unique()), columns=['datetime'])
    month_data['date'] = month_data.datetime.dt.date
    month_data['month'] = month_data.datetime.dt.month
    month_data['year'] = month_data.datetime.dt.year
    month_data['trade_day_in_month'] = month_data.groupby(['year', 'month']).cumcount()
    month_data['remaining_trade_days_in_month'] = month_data.groupby(['year', 'month']).cumcount(ascending=False)
    data_df = data_df.iloc[:, ~data_df.columns.isin(['remaining_trade_days_in_month', 'trade_day_in_month', 'month'])].merge(
        month_data[['date', 'month', 'trade_day_in_month', 'remaining_trade_days_in_month']], on='date')
    return data_df
