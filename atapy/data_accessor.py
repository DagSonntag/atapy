
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Type, Optional
import datetime
from abc import ABC, abstractmethod
from atapy.utils import load_classes_in_dir, add_time_zone_to_time
from atapy.constants import DATA_ACCESSORS_DIR
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.exceptions import AssetNotFoundException
import copy
import pytz

logger = logging.getLogger()

""" 
The data columns and types stored for the different types of data. 
Some notes:

* Volume can be floats in for example currency (crypto) assets
* The feature data should contain one sample per interval for all liquid hours, even if the exchange was closed during 
this time. This to simplify handling at a later stage. To differentiate real samples vs inferred samples, the 
real_sample flag exists
"""

ASSET_INFO_COLUMN_TYPES = {'asset_type': str, 'exchange': str, 'symbol': str, 'currency': str, 'full_name': str,
                           'industry': str, 'category': str, 'subcategory': str, 'liquid_hours': str, 'time_zone': str}

RAW_DATA_COLUMN_TYPES = {'date': 'datetime64[ns, UTC]', 'open': np.float64, 'high': np.float64, 'low': np.float64,
                         'close': np.float64, 'volume': np.float64, 'average': np.float64, 'nr_of_trades': np.int64}

FEATURE_DATA_COLUMN_TYPES = {'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64,
                             'volume': np.float64, 'average': np.float64, 'nr_of_trades': np.int64,
                             'start_time': 'datetime64[ns, UTC]', 'end_time': 'datetime64[ns, UTC]',
                             'real_sample': bool}


class DataAccessor(ABC):
    """
    The interface class for accessing and storing data in the data storage with the methods and properties required for
    implementation, as well as some general methods and the factory to instantiate connections by name.

    A DataAccessor (sub)class is responsible for storing data of the data types:

    * Asset info: Information about the asset such as exchange, asset_type, liquid_hours etc
    * Raw data: Raw data directly downloaded from the brokers/data_sources
    * Feature data: Cleaned up raw data where all slots (for the interval) exists during the assets liquid hours the
    days it were traded
    * Custom data: Custom data tables to be stored in the accessor by for example data_connections or agents

    Moreover, the accessor is also responsible for providing data for readers for a set time period (by default all).
    The time-limits can be set either in the initializer or with the get_time_restricted_instance method, and will
    restrict any access to raw or feature data for any class (agent) trying to use it, simulating that only that data
    exists.

    Time-zones: All data is stored in UTC timezones. (The accessor will convert any other to this timezone before
    saving the data). When reading the data it is by default converted back to the local (exchange) timezone.
    """

    @property
    @abstractmethod
    def accessor_name(self):
        """ The name of the DataAccessor subclass. Used when instances are created by the factory """
        raise RuntimeError

    start_time_limit: datetime.datetime = None  # The start_time_limit for accessing raw and feature data
    end_time_limit: datetime.datetime = None  # The end_time_limit for accessing raw and feature data

    @classmethod
    def factory(cls, accessor_name, *accessor_args, **accessor_kwargs):
        """
        Instantiates a DataAccessor (subclass) using the connection name. For the list of necessary and
        optional arguments please refer to the specific subclass.
        """
        load_classes_in_dir(Path(DATA_ACCESSORS_DIR))
        accessor: Type[DataAccessor]
        for accessor in cls.__subclasses__():
            if accessor.accessor_name == accessor_name:
                return accessor(*accessor_args, **accessor_kwargs)
        raise NotImplementedError("Subclass {} of DataAccessor is not implemented".format(accessor_name))

    @abstractmethod
    def write_asset_information(self, data_df: pd.DataFrame, on_duplicates: str = 'overwrite') -> None:
        """
        Writes data to the asset information data storage
        :param data_df: the DataFrame containing new information to be stored with columns as described in
        ASSET_INFO_COLUMN_TYPES
        :param on_duplicates: How to handle duplicates (duplicated exchange, symbol, asset_type and currency). Possible
        values are ['overwrite', 'raise_error', 'keep_old', 'only_new']
        :return: None
        """
        pass

    @abstractmethod
    def get_asset_information(self, exchange: str = None, asset_symbol: str = None, asset_type: str = None,
                              asset: Asset = None, sql_selection: str = None) -> pd.DataFrame:
        """
        Returns the matching asset information for the requested asset given the given the filled search parameters
        """
        pass

    @abstractmethod
    def write_raw_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame,
                             on_duplicates: str = 'overwrite'):
        """
        Writes raw asset data to the raw data storage
        :param asset: The asset to be written
        :param interval: The interval to be written
        :param data_df: the DataFrame containing new data to be stored with columns as described in
        RAW_DATA_COLUMN_TYPES
        :param on_duplicates: How to handle duplicates (duplicated date values for the same interval). Possible
        values are ['overwrite', 'raise_error', 'keep_old', 'only_new']
        :return: None
        """
        pass

    @abstractmethod
    def get_raw_asset_data(self, asset: Asset, interval: Interval,
                           time_zone: Optional[str or datetime.tzinfo] = 'local'
                           ) -> pd.DataFrame:
        """ Reads raw asset data from the raw data storage """
        pass

    @abstractmethod
    def get_feature_asset_data(self, asset: Asset, interval: Interval) -> pd.DataFrame:
        pass

    @abstractmethod
    def write_feature_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame,
                                 on_duplicates: str = 'overwrite'):
        """
       Writes raw asset data to the raw data storage
       :param asset: The asset to be written
       :param interval: The interval to be written
       :param data_df: the DataFrame containing new data to be stored with columns as described in
       FEATURE_DATA_COLUMN_TYPES
       :param on_duplicates: How to handle duplicates (duplicated date values for the same interval). Possible
       values are ['overwrite', 'raise_error', 'keep_old', 'only_new']
       :return: None
       """
        pass

    @staticmethod
    def get_empty_asset_info() -> pd.DataFrame:
        """ Returns an empty asset info frame """
        return pd.DataFrame(columns=list(ASSET_INFO_COLUMN_TYPES.keys())).astype(ASSET_INFO_COLUMN_TYPES)

    @staticmethod
    def get_empty_raw_asset_data() -> pd.DataFrame:
        """ Returns an empty raw data frame """
        return pd.DataFrame(columns=list(RAW_DATA_COLUMN_TYPES.keys())).astype(RAW_DATA_COLUMN_TYPES)

    @staticmethod
    def get_empty_feature_asset_data() -> pd.DataFrame:
        """ Returns an empty feature data frame """
        return pd.DataFrame(columns=list(FEATURE_DATA_COLUMN_TYPES.keys())).astype(FEATURE_DATA_COLUMN_TYPES)

    def get_time_restricted_instance(self, start_time: Optional[datetime.datetime] = None,
                                     end_time: Optional[datetime.datetime] = None, inclusive: str = 'end_time'
                                     ) -> 'DataAccessor':
        """
        Return a copy of the accessor with restrictions on what raw and feature data that can be accessed by for example
        agents. This to simulate that only this data is available, and prevent the agents to look on for example testing
        data
        :param start_time: The start time limit. If provided with a timezone that timezone will be used, otherwise the
        local exchange time of the asset (that is collected with get_raw_asset_data or get_feature_asset_data)
        :param end_time: The end time limit. If provided with a timezone that timezone will be used, otherwise the
        local exchange time of the asset (that is collected with get_raw_asset_data or get_feature_asset_data)
        :param inclusive: Whether or not to make the timespan inclusive. Possible values:
        ['end_time' [default], 'start_time' , 'both', 'none']
        :return: A new instance of this accessor, with the available data restricted
        """
        if inclusive != 'end_time':
            raise NotImplementedError
        new_accessor = copy.deepcopy(self)
        if start_time is not None and (self.start_time_limit is None or start_time > self.start_time_limit):
            new_accessor.start_time_limit = start_time
        if end_time is not None and (self.end_time_limit is None or end_time <= self.end_time_limit):
            new_accessor.end_time_limit = end_time
        return new_accessor

    @abstractmethod
    def write_custom_table(self, table_name: str, data_df: pd.DataFrame, on_duplicates: str = 'overwrite'):
        """
        Writes a custom table with some data to the data storage
        :param table_name: The given name of the table the data should be written to
        :param data_df: The DataFrame containing new data to be stored
        :param on_duplicates: How to handle duplicates (duplicated date values for the same interval). Possible
        values are ['overwrite', 'raise_error', 'keep_old', 'only_new']
        :return: None
        """
        pass

    @abstractmethod
    def read_custom_table(self, table_name: str, sql_selection: str = None) -> pd.DataFrame:
        """
        Reads data stored in a custom table
        :param table_name: The given table name containing the data
        :param sql_selection: A subselection of the data, where the table is referred to either by name, or as df
        :return: A pandas DataFrame containing the selected data
        """
        pass

    @abstractmethod
    def clear(self):
        """ Wipe the data storage and reset it to an initial setup"""
        pass

    def get_asset_local_tzinfo(self, asset: Asset) -> pytz.tzinfo.DstTzInfo or pytz.tzinfo.StaticTzInfo:
        """ A helper method to return the time_zone of an asset """
        info = self.get_asset_information(asset=asset)
        if info.shape[0] == 0:
            raise AssetNotFoundException
        else:
            return pytz.timezone(info.time_zone.iloc[0])

    def get_asset_liquid_hours(self, asset) -> (datetime.time, datetime.time):
        """ Returns the start and end liquid time for an asset in naive datetime.time objects """
        info = self.get_asset_information(asset=asset)
        if info.shape[0] == 0:
            raise AssetNotFoundException
        liquid_hours_str = info.liquid_hours.iloc[0]
        if liquid_hours_str is None or len(liquid_hours_str) != 9:
            raise NotImplementedError(
                "The liquid hours are given in an unexpected format, please investigate: {}".format(liquid_hours_str))
        start_trade_time, end_trade_time = liquid_hours_str.split("-")
        time_zone = self.get_asset_local_tzinfo(asset)
        start_trade_time = datetime.time(int(start_trade_time[0:2]), int(start_trade_time[2:]))
        if end_trade_time == '2400':
            end_trade_time = datetime.time.max
        else:
            end_trade_time = datetime.time(int(end_trade_time[0:2]), int(end_trade_time[2:]))
        return start_trade_time, end_trade_time
