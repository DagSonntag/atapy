
from pathlib import Path
import pandas as pd
import pandasql as ps
import numpy as np
import logging
import datetime
from typing import Callable, List
from atapy.utils import to_datetime, check_types
from atapy.asset import Asset
from atapy.data_accessor import DataAccessor, RAW_DATA_TYPES, ASSET_INFO_TYPES, FEATURE_DATA_TYPES
from atapy.interval import Interval
import shutil

logger = logging.getLogger()


class FileAccessor(DataAccessor):

    accessor_name = 'file'

    def __init__(self, data_folder: str = 'data', start_time: str or datetime.date or datetime.datetime = None,
                 end_time: str or datetime.date or datetime.datetime = None):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.exchanges_folder = self.data_folder / 'exchanges'
        self.exchanges_folder.mkdir(exist_ok=True)
        self.raw_data_path = str(self.data_folder / 'exchanges/{exchange}/{asset_type}/{asset_symbol}/raw/{interval}_df.ftr')
        self.feature_data_path = str(self.data_folder / 'exchanges/{exchange}/{asset_type}/{asset_symbol}/asset_feature_frame/{interval}.ftr')
        self.custom_table_path = str(self.data_folder / "custom_tables/{}.ftr")
        self.start_time_limit = to_datetime(start_time) if start_time is not None else None
        self.end_time_limit = to_datetime(end_time) if end_time is not None else None

    @staticmethod
    def _update_file(filepath: Path, new_data_df: pd.DataFrame, key_column_name: str or List[str]):
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        if filepath.exists():
            new_data_df = pd.concat([pd.read_feather(filepath), new_data_df], axis=0)
            new_data_df = new_data_df[~new_data_df[key_column_name].duplicated(keep='last')]
        new_data_df = new_data_df.sort_values(key_column_name)
        new_data_df.reset_index(drop=True).to_feather(filepath)

    def _update_information_file(self, filepath: Path, new_data_df: pd.DataFrame):
        self._update_file(Path(filepath), new_data_df, ['exchange', 'symbol', 'security_type'])

    def _update_raw_data_file(self, filepath: Path or str, new_data_df: pd.DataFrame):
        self._update_file(Path(filepath), new_data_df, 'date')

    def _update_feature_data_file(self, filepath: Path or str, new_data_df: pd.DataFrame):
        self._update_file(Path(filepath), new_data_df, 'start_time')

    def write_asset_information(self, data_df: pd.DataFrame):
        data_df = check_types(data_df, ASSET_INFO_TYPES)
        self._update_information_file(self.data_folder / 'symbol_details.ftr', data_df)
        for exchange, exchange_df in data_df.groupby('exchange'):
            path = self.data_folder / 'exchanges/{}/symbol_details.ftr'.format(exchange)
            self._update_information_file(path, exchange_df)

    def get_asset_information(self, exchange: str = None, asset_symbol: str = None, asset_type: str = None,
                              asset_currency: str = None, asset: Asset = None,
                              sql_expression: str = None) -> pd.DataFrame:
        if asset is not None:
            if exchange is not None or asset_symbol is not None or asset_type is not None or asset_currency is not None:
                raise ValueError("If an asset is given the exchange, symbol, type or currency cannot be given as well")
            exchange, asset_symbol, asset_type, asset_currency = asset.exchange, asset.symbol, asset.type, asset.currency
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
        if asset_currency is not None:
            df = df[df.currency == asset_currency]
        if sql_expression is not None:
            df = ps.sqldf(sql_expression)
        return df

    def get_raw_asset_data(self, asset: Asset, interval: Interval) -> pd.DataFrame:
        file_path = self.raw_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)
        if Path(file_path).exists():
            df = pd.read_feather(file_path)
            if self.start_time_limit is not None:
                df = df[df.date >= self.start_time_limit]
            if self.end_time_limit is not None:
                df = df[df.date < self.end_time_limit]
            return df.reset_index(drop=True)
        else:
            return self.get_empty_raw_asset_data()

    def write_raw_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame):
        data_df = check_types(data_df, RAW_DATA_TYPES)
        self._update_raw_data_file(self.raw_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name), data_df)

    def _save_feature_asset_data(self, asset: Asset, interval: Interval, df_to_save: pd.DataFrame):
        file_path = self.feature_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        self._update_feature_data_file(file_path, df_to_save)

    def get_feature_asset_data(self, asset: Asset, interval: Interval) -> pd.DataFrame:
        file_path = self.feature_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name)
        if Path(file_path).exists():
            df = pd.read_feather(file_path)
            if self.start_time_limit is not None:
                df = df[df.end_time > self.start_time_limit]
            if self.end_time_limit is not None:
                df = df[df.end_time <= self.end_time_limit]
            return df.reset_index(drop=True)
        else:
            return self.get_empty_feature_asset_data()

    def write_feature_asset_data(self, asset: Asset, interval: Interval, data_df: pd.DataFrame):
        data_df = check_types(data_df, FEATURE_DATA_TYPES)
        self._update_feature_data_file(self.feature_data_path.format(
            exchange=asset.exchange.upper(), asset_type=asset.type.upper(), asset_symbol=asset.symbol.upper(),
            interval=interval.name), data_df)

    def write_custom_table(self, table_name: str, df: pd.DataFrame, method: str = 'overwrite'):
        file_path = Path(self.custom_table_path.format(table_name))
        file_path.parent.mkdir(exist_ok=True, parents=True)
        if method == 'overwrite':
            df.to_feather(file_path)
        else:
            raise NotImplementedError

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
        return "Fileaccessor({}) from {} to {}".format(self.data_folder, self.start_time_limit, self.end_time_limit)

    __repr__ = __str__
