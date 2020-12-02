
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Type, Optional
import datetime
from abc import ABC, abstractmethod
from atapy.utils import load_classes_in_dir
from atapy.constants import DATA_ACCESSORS_DIR
from atapy.asset import Asset
from atapy.interval import Interval
import copy

logger = logging.getLogger()

# Note that the volume can be floats in for example Crypto
FEATURE_DATA_TYPES = {'date': object, 'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64,
                      'volume': np.float64, 'average': np.float64, 'nr_of_trades': np.int64,
                      'start_time': np.datetime64, 'end_time': np.datetime64, 'real_sample': bool}

RAW_DATA_TYPES = {'date': np.datetime64, 'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64,
                  'volume': np.float64, 'average': np.float64, 'nr_of_trades': np.int64}

ASSET_INFO_TYPES = {'security_type': str, 'exchange': str, 'symbol': str, 'currency': str, 'full_name':str,
                    'industry': str, 'category': str, 'subcategory': str, 'liquid_hours': str}

class DataAccessor(ABC):

    @property
    @abstractmethod
    def accessor_name(self):
        raise RuntimeError

    start_time_limit = None
    end_time_limit = None

    @classmethod
    def factory(cls, accessor_name, *args, **accessor_args):
        load_classes_in_dir(Path(DATA_ACCESSORS_DIR))
        accessor: Type[DataAccessor]
        for accessor in cls.__subclasses__():
            if accessor.accessor_name == accessor_name:
                return accessor(*args, **accessor_args)
        raise NotImplementedError("Subclass {} of DataAccessor is not implemented".format(accessor_name))

    @abstractmethod
    def write_asset_information(self, data_df: pd.DataFrame):
        """ Writes data to the asset information data storage """
        pass

    @abstractmethod
    def get_asset_information(self, exchange: str = None, asset_symbol: str = None, asset_type: str = None,
                              currency: str = None, asset: Asset = None, sql_selection: str = None) -> pd.DataFrame:
        """ Returns the matching asset information for the requested asset given the given the filled search parameters
        """
        pass

    @abstractmethod
    def write_raw_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame):
        """ Writes raw asset data to the raw data storage """
        pass

    @abstractmethod
    def get_raw_asset_data(self, asset: Asset, interval: Interval) -> pd.DataFrame:
        """ Reads raw asset data from the raw data storage """
        pass

    @abstractmethod
    def get_feature_asset_data(self, asset: Asset, interval: Interval) -> pd.DataFrame:
        pass

    def write_feature_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame):
        pass

    @staticmethod
    def get_empty_raw_asset_data() -> pd.DataFrame:
        return pd.DataFrame(columns=list(RAW_DATA_TYPES.keys())).astype(RAW_DATA_TYPES)

    @staticmethod
    def get_empty_feature_asset_data() -> pd.DataFrame:
        return pd.DataFrame(columns=list(FEATURE_DATA_TYPES.keys())).astype(FEATURE_DATA_TYPES)

    @staticmethod
    def get_empty_asset_info() -> pd.DataFrame:
        return pd.DataFrame(columns=list(ASSET_INFO_TYPES.keys())).astype(ASSET_INFO_TYPES)

    def get_time_restricted_instance(self, start_time: Optional[datetime.datetime] = None,
                                     end_time: Optional[datetime.datetime] = None, inclusive: str = 'end_time'):
        """

        :param start_time: not inclusive
        :param end_time: inclusive
        :param inclusive: start_time, end_time, None
        :return:
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
    def write_custom_table(self, table_name: str, df: pd.DataFrame, method: str = 'overwrite'):
        pass

    @abstractmethod
    def read_custom_table(self, table_name: str, sql_selection: str = None):
        pass

    @abstractmethod
    def clear(self):
        pass
