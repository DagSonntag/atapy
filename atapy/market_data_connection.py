import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type
import pandas as pd
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.utils.method_utils import load_classes_in_dir
from atapy.constants import MARKET_DATA_CONNECTIONS_DIR
from atapy.data_accessor import DataAccessor, FEATURE_DATA_COLUMN_TYPES
from atapy.utils.data_cleaning_utils import add_start_and_end_time, filter_out_non_traded_hours, \
    fill_missing_intervals, aggregate_intraday_feature_data

logger = logging.getLogger()


class MarketDataConnection(ABC):
    """
    The interface class for data connection with the methods and properties required for implementation, as well as some
    general methods and the factory to instantiate connections by name.

    An MarketDataConnection (sub)class is responsible for interacting with the broker/data source regarding data
    collection, i.e.:

    * Collect information about available data at the broker (i.e. available assets, exchanges etc)
    * Collect historical data for selected assets
    * Collect realtime data for selected assets
    * Preprocess (clean and unify) the data provided by the broker to conform to the feature data standard
    """
    data_accessor: DataAccessor = None
    interval_to_aggregate: Interval = None

    @abstractmethod
    def connection_name(self) -> str:
        """ The name of the connection subclass. Used when instances are created by the factory """
        raise RuntimeError

    @classmethod
    def factory(cls, connection_name: str, *connection_args, **connection_kwargs) -> 'MarketDataConnection':
        """
        Instantiates a MarketDataConnection (subclass) using the connection name. For the list of necessary and
        optional arguments please refer to the specific subclass.
        """
        load_classes_in_dir(Path(MARKET_DATA_CONNECTIONS_DIR))
        connection: Type[MarketDataConnection]
        for connection in cls.__subclasses__():
            if connection.connection_name == connection_name:
                return connection(*connection_args, **connection_kwargs)
        raise NotImplementedError("Subclass {} of Market data connection is not implemented".format(connection_name))

    @abstractmethod
    def collect_asset_information(self, asset_type: str) -> None:
        """
        Collects information about what assets and exchanges that are provided by the broker/data_source for the given
        asset type and saves this to the data storage using the class data accessor. See the DataAcessor class for more
        details on what this data must contain.
        """
        pass

    @abstractmethod
    def collect_historical_data(self, asset: Asset, redownload_data: bool = False,
                                update_feature_data: bool = False) -> None:
        """
        Collects historical data for the given asset, making sure the (raw) data stored is up to date with respect to
        the broker at the time the method is called. The method only collects new data from the last data already
        downloaded, unless the redownload_data flag is set to True. Moreover, it also updates the stored feature data
        only if the update_feature_data flag is set to True. The data is then stored using the class data_accessor.
        """
        pass

    @abstractmethod
    def collect_realtime_data(self, asset: Asset) -> None:
        """
        Starts a subscription on the data of the specified asset, making sure that the data always is up to date, even
        when the method is not called. This may require some multiprocessing libraries depending on how the broker/data
        source API is built.
        """
        pass

    @abstractmethod
    def stop_collecting_realtime_data(self, asset: Asset) -> None:
        """
        Cancels the subscription for the given asset. Note that this automatically is done when the MarketDataConnection
        instance is garbage collected.
        """
        pass

    def preprocess_raw_data(self, asset: Asset, complete_reprocess: bool = False) -> None:
        """
        Preprocess raw data collected using the MarketDataConnection class and stores the result as feature_data for
        a given asset. If complete_reprocess=True all feature data is reprocessed, otherwise just the part for which new
        raw data has (possibly) been made available since last time this method was executed.

        This method can for example be used in collect_historical_data or collect_realtime_data to preprocess data, but
        also when data has not preprocessed during collection, or when the preprocess algorithm has changed.

        Provided here is a standard implementation. This method may have been overloaded by certain subclasses if
        necessary
         """
        if self.interval_to_aggregate is None:
            raise RuntimeError("Standard preprocessing is only possible for MarketDataConnection subclasses where "
                               "interval_to_aggregate is set")
        raw_df = self.data_accessor.get_raw_asset_data(asset, self.interval_to_aggregate)
        if not complete_reprocess:
            feature_df = self.data_accessor.get_feature_asset_data(asset, self.interval_to_aggregate)
            new_raw_df = raw_df[raw_df.date >= feature_df.start_time.max()] if feature_df.shape[0] > 0 else raw_df
            if new_raw_df.shape[0] == 1 and new_raw_df.nr_of_trades.iloc[0] == feature_df.nr_of_trades.iloc[-1]:
                logger.debug("No new raw data found for {}. No preprocessing necessary".format(asset))
                return
            # Reprocess the last day, since this is the highest aggregation level
            raw_df = raw_df[raw_df.date.dt.date >= new_raw_df.date.iloc[0].date()]
        # Round up all values to the next full interval and remove duplicates
        raw_df['date'] = raw_df['date'].dt.ceil('{}s'.format(self.interval_to_aggregate.in_seconds()))
        raw_df = raw_df[~raw_df.date.duplicated(keep='last')]
        feature_df = add_start_and_end_time(raw_df, self.interval_to_aggregate)
        start_daily_trade_time, end_daily_trade_time = self.data_accessor.get_asset_liquid_hours(asset)
        time_zone = self.data_accessor.get_asset_local_tzinfo(asset)
        feature_df = filter_out_non_traded_hours(data_df=feature_df, start_trade_time=start_daily_trade_time,
                                                 end_trade_time=end_daily_trade_time)
        traded_dates = pd.concat([feature_df.date.drop_duplicates(), self.data_accessor.get_raw_asset_data(
            asset, Interval.daily).date.dt.date]).drop_duplicates()
        traded_dates = traded_dates.sort_values().reset_index(drop=True)
        feature_df = fill_missing_intervals(feature_df, interval=self.interval_to_aggregate, traded_dates=traded_dates,
                                            start_daily_trade_time=start_daily_trade_time,
                                            end_daily_trade_time=end_daily_trade_time, time_zone=time_zone)
        dfs = aggregate_intraday_feature_data(feature_df.drop(columns=['date']), self.interval_to_aggregate)
        for agg_interval, interval_df in dfs.items():
            self.data_accessor.write_feature_asset_data(
                asset, agg_interval, interval_df.astype(FEATURE_DATA_COLUMN_TYPES),
                on_duplicates='only_new' if complete_reprocess else 'overwrite')
