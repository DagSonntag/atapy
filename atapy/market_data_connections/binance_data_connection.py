from atapy.data_cleaning import get_liquid_hours, add_start_and_end_time, \
    fill_missing_intervals, aggregate_intraday_data
from atapy.data_accessor import RAW_DATA_TYPES, FEATURE_DATA_TYPES, ASSET_INFO_TYPES
from atapy.utils import to_datetime, to_utc_milliseconds, round_up_til_full_minute
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.market_data_connection import MarketDataConnection
from binance.client import Client
from atapy.data_accessor import DataAccessor
import pandas as pd
import datetime
import logging

logger = logging.getLogger()

class BinanceDataConnection(MarketDataConnection):
    """
    The Binance data connection. Collects info and data from the Binance Crypto currency exchange. Note that
    an API_key and api_secret is necessary to access data. More info can be found at
    https://binance-docs.github.io/apidocs/spot/en/#api-key-setup .
    The Class is also built on top of https://python-binance.readthedocs.io/en/latest/overview.html (whose client is
    accessible in the client instance variable)
    """
    connection_name = 'binance'

    def __init__(self, data_accessor: DataAccessor, api_key=None, api_secret=None, api_details_filename=None):
        if (api_key is None or api_secret is None) and api_details_filename is None:
            raise ValueError("Either the api key and api_secret must be provided or a filename containing the details "
                             "must be provided")
        self.client = Client(api_key=api_key, api_secret=api_secret)
        # Ping to make sure the target can be reached
        _ = self.client.ping()
        self.data_accessor = data_accessor

    def collect_asset_information(self, asset_type: str = 'currency'):
        if asset_type.upper() != 'CURRENCY':
            raise ValueError('The Binance exchange only support (crypto) currency assets')
        info = self.client.get_exchange_info()
        symbol_details = pd.DataFrame(info['symbols'])
        symbol_details['security_type'] = 'currency'
        symbol_details['exchange'] = 'Binance'
        symbol_details['currency'] = symbol_details['quoteAsset']
        symbol_details['full_name'] = symbol_details['symbol']
        symbol_details['industry'] = 'crypto-currency'
        symbol_details['category'] = symbol_details['baseAsset']
        symbol_details['subcategory'] = 'NA'
        symbol_details['liquid_hours'] = '0000-2400'
        self.data_accessor.write_asset_information(symbol_details[ASSET_INFO_TYPES.keys()])
        # Possibly save another table for later use if necessary for order details

    def collect_historical_asset_data(self, asset: Asset, redownload_data: bool = False,
                                      update_feature_data: bool = False):
        """ Downloads minute data and aggregates this into larger intervals in feature data """
        def get_historical_data_safe(_asset, _start_time, _end_time):
            try_again = True
            while try_again:
                try:
                    data = self.client.get_historical_klines(_asset.symbol, interval=Client.KLINE_INTERVAL_1MINUTE,
                                                             start_str=to_utc_milliseconds(_start_time),
                                                             end_str=to_utc_milliseconds(_end_time), limit=1000)
                    try_again = False
                except ConnectionError as e:
                    logger.error(e)
                    continue
            return data

        if asset.exchange != 'Binance':
            raise ValueError("BinanceDataConnection only supports the Binance exchange")
        existing_data_df = self.data_accessor.get_raw_asset_data(asset, Interval.minute)
        if redownload_data or existing_data_df.shape[0] == 0:
            start_time = to_datetime('2000-01-01')
        else:
            start_time = existing_data_df.date.iloc[-1]
        end_time = datetime.datetime.now()
        data = get_historical_data_safe(asset, start_time, end_time)
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'total_cost', 'nr_of_trades', 'taker_volume', 'taker_quote_asset_volume',
                                         'ignore'])
        df['date'] = pd.to_datetime(df.open_time, unit='ms')
        df = df.astype({key: val for key, val in RAW_DATA_TYPES.items() if key in df.columns})
        df['total_cost'] = df['total_cost'].astype(float)
        df['average'] = df['total_cost']/df.volume
        self.data_accessor.write_raw_asset_data(asset, Interval.minute, df[RAW_DATA_TYPES.keys()])
        if update_feature_data:
            existing_feature_data = self.data_accessor.get_feature_asset_data(asset, Interval.minute)
            # prepare feature data and save
            new_raw_df = pd.concat([existing_data_df[existing_data_df.date.dt.date >= existing_feature_data.date.max()],
                                    df[RAW_DATA_TYPES.keys()]])
            new_raw_df = new_raw_df[~new_raw_df.date.duplicated(keep='last')]
            feature_df = new_raw_df.copy()
            # Some datetimes are not on even minutes. Solve this by
            # (1) removing milli and nanosecond
            # (2) rounding up seconds to the next minute slot
            feature_df['date'] = feature_df['date'].astype('datetime64[s]')
            feature_df['date2'] = feature_df.date.apply(round_up_til_full_minute)
            # Remove duplicates by keeping those with the lowest date
            feature_df = feature_df.sort_values(['date'])
            feature_df = feature_df[~feature_df.date2.duplicated()]
            feature_df['date'] = feature_df['date2']
            feature_df = feature_df.drop(columns='date2')
            # Then add start and end_time and fill missing slots
            feature_df = add_start_and_end_time(feature_df, Interval.minute)
            a = feature_df.groupby('date').count().sort_index()
            # First check that all extra slots are removed
            assert (not any(a.open > 1440))
            start_daily_trade_time, end_daily_trade_time = get_liquid_hours(
                self.data_accessor.get_asset_information(asset=asset).liquid_hours.iloc[0])
            feature_df = feature_df.sort_values('start_time')
            traded_dates = feature_df.date.drop_duplicates()
            feature_df = fill_missing_intervals(feature_df, interval=Interval.minute, traded_dates=traded_dates,
                                                start_daily_trade_time=start_daily_trade_time,
                                                end_daily_trade_time=end_daily_trade_time)
            a = feature_df.groupby('date').count().sort_index()
            # Assert that all slots are filled except on the first and last days
            assert (not any((a.open != 1440) & (a.index != feature_df.date.min()) & (a.index != feature_df.date.max())))
            logger.debug(
                "Shape before feature transformation {}, shape after {}, missing dates: {}".format(
                    new_raw_df.shape, feature_df.shape,
                    set(new_raw_df.date.dt.date) - set(feature_df.start_time.dt.date)))
            # Aggregate up the data to higher intervals
            dfs = aggregate_intraday_data(feature_df, Interval.minute)
            # Save the data
            for agg_interval, interval_df in dfs.items():
                self.data_accessor.write_feature_asset_data(asset, agg_interval,
                                                            interval_df.astype(FEATURE_DATA_TYPES))
