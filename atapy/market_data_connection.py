import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type
from atapy.asset import Asset
from atapy.utils import load_classes_in_dir
from atapy.constants import MARKET_DATA_CONNECTIONS_DIR

logger = logging.getLogger()


class MarketDataConnection(ABC):

    @abstractmethod
    def connection_name(self):
        raise RuntimeError

    @classmethod
    def factory(cls, connection_name: str, *connection_args):
        load_classes_in_dir(Path(MARKET_DATA_CONNECTIONS_DIR))
        connection: Type[MarketDataConnection]
        for connection in cls.__subclasses__():
            if connection.connection_name == connection_name:
                return connection(*connection_args)
        raise NotImplementedError("Subclass {} of Market data connection is not implemented".format(connection_name))

    @abstractmethod
    def collect_asset_information(self, asset_type: str = 'stk'):
        pass

    @abstractmethod
    def collect_historical_asset_data(self, asset: Asset, redownload_data: bool = False,
                                      update_feature_data: bool = False):
        pass
