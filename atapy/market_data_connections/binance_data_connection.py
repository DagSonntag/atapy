from requests import RequestException
import time
from atapy.data_accessor import RAW_DATA_COLUMN_TYPES, ASSET_INFO_COLUMN_TYPES
from atapy.utils import to_datetime, to_utc_milliseconds
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.market_data_connection import MarketDataConnection
from binance.client import Client
from atapy.data_accessor import DataAccessor
import pandas as pd
import datetime
import logging
import pytz

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
        self.interval_to_aggregate = Interval.minute
        self.data_accessor = data_accessor

    def collect_asset_information(self, asset_type: str = 'currency'):
        if asset_type.upper() != 'CURRENCY':
            raise ValueError('The Binance exchange only support (crypto) currency assets')
        info = self.client.get_exchange_info()
        symbol_details = pd.DataFrame(info['symbols'])
        symbol_details['asset_type'] = 'currency'
        symbol_details['exchange'] = 'Binance'
        symbol_details['currency'] = symbol_details['quoteAsset']
        symbol_details['full_name'] = symbol_details['symbol']
        symbol_details['industry'] = 'crypto-currency'
        symbol_details['category'] = symbol_details['baseAsset']
        symbol_details['subcategory'] = 'NA'
        symbol_details['liquid_hours'] = '0000-2400'
        symbol_details['time_zone'] = 'UTC'

        self.data_accessor.write_asset_information(symbol_details[ASSET_INFO_COLUMN_TYPES.keys()])
        # Possibly save another table for later use if necessary for order details

    def collect_historical_data(self, asset: Asset, redownload_data: bool = False,
                                update_feature_data: bool = False):
        """ Downloads minute data and aggregates this into larger intervals in feature data. """
        def get_historical_data_safe(_asset, _start_time, _end_time):
            """ Runs the data collection call in a while loop to allow temporary disconnects and timeouts """
            while True:
                try:
                    dat = self.client.get_historical_klines(_asset.symbol, interval=Client.KLINE_INTERVAL_1MINUTE,
                                                            start_str=to_utc_milliseconds(_start_time),
                                                            end_str=to_utc_milliseconds(_end_time), limit=1000)
                    return dat
                except (ConnectionError, RequestException) as e:
                    logger.error(e)
                    time.sleep(10)
                    continue

        if asset.exchange != 'Binance':
            raise ValueError("BinanceDataConnection only supports the Binance exchange")
        existing_data_df = self.data_accessor.get_raw_asset_data(asset, Interval.minute)
        if redownload_data or existing_data_df.shape[0] == 0:
            start_time = pytz.timezone('UTC').localize(to_datetime('2000-01-01'))
        else:
            start_time = existing_data_df.date.iloc[-1].to_pydatetime()
        end_time = datetime.datetime.now(pytz.timezone('UTC'))
        data = get_historical_data_safe(asset, start_time, end_time)
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'total_cost', 'nr_of_trades', 'taker_volume', 'taker_quote_asset_volume',
                                         'ignore'])
        df['date'] = pd.to_datetime(df.open_time, unit='ms')
        df['date'] = df['date'].dt.tz_localize('UTC')
        df = df.astype({key: val for key, val in RAW_DATA_COLUMN_TYPES.items() if key in df.columns})
        df['total_cost'] = df['total_cost'].astype(float)
        df['average'] = df['total_cost']/df.volume
        df.loc[df.average.isna(), 'average'] = df.close[df.average.isna()]  # The case when volume=0
        self.data_accessor.write_raw_asset_data(asset, Interval.minute, df[RAW_DATA_COLUMN_TYPES.keys()])
        if update_feature_data:
            self.preprocess_raw_data(asset, False)

    def collect_realtime_data(self, asset: Asset) -> None:
        # TODO
        raise NotImplementedError

    def stop_collecting_realtime_data(self, asset: Asset) -> None:
        # TODO
        raise NotImplementedError
