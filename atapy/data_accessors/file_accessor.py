from pathlib import Path
import pandas as pd
import pandasql as ps
import logging
import datetime
from typing import List, Optional
from atapy.utils import to_datetime, check_types, to_tzinfo
from atapy.asset import Asset
from atapy.data_accessor import DataAccessor, RAW_DATA_COLUMN_TYPES, ASSET_INFO_COLUMN_TYPES, FEATURE_DATA_COLUMN_TYPES
from atapy.interval import Interval
import shutil

logger = logging.getLogger()


class FileAccessor(DataAccessor):

    accessor_name = 'file'

    def __init__(self, data_folder: str = 'data',
                 start_time: Optional[str or datetime.date or datetime.datetime] = None,
                 end_time: Optional[str or datetime.date or datetime.datetime] = None):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.exchanges_folder = self.data_folder / 'exchanges'
        self.exchanges_folder.mkdir(exist_ok=True)
        self.asset_folder_path = str(self.data_folder / 'exchanges/{exchange}/{asset_type}/{asset_symbol}')
        self.raw_data_path = self.asset_folder_path + '/raw/{interval}_df.ftr'
        self.feature_data_path = self.asset_folder_path + '/feature/{interval}.ftr'
        self.custom_table_path = str(self.data_folder / "custom_tables/{}.ftr")
        self.start_time_limit = to_datetime(start_time) if start_time is not None else None
        self.end_time_limit = to_datetime(end_time) if end_time is not None else None

    @staticmethod
    def _update_file(filepath: Path, new_data_df: pd.DataFrame, key_column_name: str or List[str], on_duplicates: str):
        filepath.parent.mkdir(exist_ok=True, parents=True)
        if filepath.exists():
            data_df = pd.concat([pd.read_feather(filepath), new_data_df], axis=0)
            if on_duplicates == 'raise_error':
                if data_df[key_column_name].duplicated().shape[0] > 0:
                    raise ValueError("Duplicate entries found when storing new data. Duplicate keys are: "
                                     "{}".format(data_df[key_column_name].duplicated().to_list()))
            elif on_duplicates == 'overwrite':
                data_df = data_df[~data_df[key_column_name].duplicated(keep='last')]
            elif on_duplicates == 'keep_old':
                data_df = data_df[~data_df[key_column_name].duplicated(keep='first')]
            elif on_duplicates == 'only_new':
                data_df = new_data_df
        else:
            data_df = new_data_df
        data_df = data_df.sort_values(key_column_name)
        data_df.reset_index(drop=True).to_feather(filepath)

    def write_asset_information(self, data_df: pd.DataFrame, on_duplicates: str = 'overwrite'):
        data_df = check_types(data_df, ASSET_INFO_COLUMN_TYPES)
        key_columns = ['exchange', 'symbol', 'asset_type']
        self._update_file(self.data_folder / 'symbol_details.ftr', data_df, key_columns, on_duplicates)
        for exchange, exchange_df in data_df.groupby('exchange'):
            path = self.data_folder / 'exchanges/{}/symbol_details.ftr'.format(exchange)
            self._update_file(path, exchange_df, key_columns, on_duplicates)

    def get_asset_information(self, exchange: str = None, asset_symbol: str = None, asset_type: str = None,
                              asset: Asset = None, sql_selection: str = None) -> pd.DataFrame:
        if asset is not None:
            if exchange is not None or asset_symbol is not None or asset_type is not None:
                raise ValueError("If an asset is given the exchange, symbol or type is provided as well")
            exchange, asset_symbol, asset_type = asset.exchange, asset.symbol, asset.type
        if exchange is None:
            file_path = self.data_folder / 'symbol_details.ftr'
        else:
            file_path = self.data_folder / 'exchanges/{}/symbol_details.ftr'.format(exchange)
        if Path(file_path).exists():
            df = pd.read_feather(file_path)
        else:
            return self.get_empty_asset_info()
        if asset_symbol is not None:
            df = df[df.symbol == asset_symbol]
        if sql_selection is not None:
            df = ps.sqldf(sql_selection)
        return df

    def get_raw_asset_data(self, asset: Asset, interval: Interval,
                           time_zone: Optional[str or datetime.tzinfo] = 'local') -> pd.DataFrame:
        time_zone = self.get_asset_local_tzinfo(asset) if time_zone == 'local' else to_tzinfo(time_zone)
        file_path = self.raw_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)
        if Path(file_path).exists():
            df = pd.read_feather(file_path)
            if self.start_time_limit is not None:
                if self.start_time_limit.tzinfo is None:
                    # If no timezone is given, assume it is given in local time
                    df = df[df.date > self.get_asset_local_tzinfo(asset).localize(self.start_time_limit)]
                else:
                    df = df[df.date >= self.start_time_limit]
            if self.end_time_limit is not None:
                if self.end_time_limit.tzinfo is None:
                    df = df[df.date < self.get_asset_local_tzinfo(asset).localize(self.end_time_limit)]
                else:
                    df = df[df.date < self.end_time_limit]
            df = df.reset_index(drop=True)
        else:
            df = self.get_empty_raw_asset_data()
        if time_zone.zone == 'UTC':
            return df
        else:
            df['date'] = df.date.dt.tz_convert(time_zone)
            return df

    def write_raw_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame,
                             on_duplicates: str = 'overwrite'):
        data_df = check_types(data_df, RAW_DATA_COLUMN_TYPES)
        self._update_file(Path(self.raw_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)), new_data_df=data_df, key_column_name='date', on_duplicates=on_duplicates)

    def get_feature_asset_data(self, asset: Asset, interval: Interval,
                               time_zone: Optional[str or datetime.tzinfo] = 'local') -> pd.DataFrame:
        time_zone = self.get_asset_local_tzinfo(asset) if time_zone == 'local' else to_tzinfo(time_zone)
        file_path = self.feature_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)
        if Path(file_path).exists():
            df = pd.read_feather(file_path)
            if self.start_time_limit is not None:
                if self.start_time_limit.tzinfo is None:
                    # If no timezone is given, assume it is given in local time
                    df = df[df.end_time > self.get_asset_local_tzinfo(asset).localize(self.start_time_limit)]
                else:
                    df = df[df.end_time > self.start_time_limit]
            if self.end_time_limit is not None:
                if self.end_time_limit.tzinfo is None:
                    # If no timezone is given, assume it is given in local time
                    df = df[df.end_time <= self.get_asset_local_tzinfo(asset).localize(self.end_time_limit)]
                else:
                    df = df[df.end_time <= self.end_time_limit]
            df = df.reset_index(drop=True)
        else:
            df = self.get_empty_feature_asset_data()
        if time_zone.zone == 'UTC':
            return df
        else:
            df['start_time'] = df['start_time'].dt.tz_convert(time_zone)
            df['end_time'] = df['end_time'].dt.tz_convert(time_zone)
            return df

    def write_feature_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame,
                                 on_duplicates: str = 'overwrite'):
        data_df = check_types(data_df, FEATURE_DATA_COLUMN_TYPES)
        self._update_file(Path(self.feature_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)), new_data_df=data_df, key_column_name='start_time', on_duplicates=on_duplicates)

    def write_custom_table(self, table_name: str, df: pd.DataFrame, on_duplicates: str = 'overwrite'):
        file_path = Path(self.custom_table_path.format(table_name))
        file_path.parent.mkdir(exist_ok=True, parents=True)
        self._update_file(Path(file_path), df, list(df.columns), on_duplicates)

    def read_custom_table(self, table_name: str, sql_query: str = None):
        file_path = Path(self.custom_table_path.format(table_name))
        if file_path.exists():
            df = pd.read_feather(file_path)
            if sql_query is not None:
                exec(table_name + '= df')
                df = ps.sqldf(sql_query)
            return df
        else:
            return None

    def clear(self):
        shutil.rmtree(self.data_folder)

    def __str__(self):
        return "FileAccessor({}) from {} to {}".format(self.data_folder, self.start_time_limit, self.end_time_limit)

    __repr__ = __str__
