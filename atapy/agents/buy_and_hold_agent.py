from typing import Optional, Dict, Any
import datetime
import logging
from atapy.agent import Agent
from atapy.data_accessor import DataAccessor
from atapy.order import MarketOrder
from atapy.asset import Asset
from atapy.portfolio_handlers.backtest_portfolio_handler import BacktestPortfolioHandler

logger = logging.getLogger()


class BuyAndHoldAgent(Agent):
    """
    A buy and hold agent used for comparisons (single asset)
    """

    agent_name = 'buy_and_hold'

    def __init__(self, asset: Asset, interval_size: str = 'daily'):
        # Here we will also need more trade info, such as cost per trade etc
        self.asset = asset
        self.interval_size = interval_size

    def step(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
             current_time: Optional[datetime.datetime] = None):
        portfolio_handler.execute_order(
            MarketOrder(asset=self.asset, action='buy', quantity=-1, creation_time=current_time))

    def execute(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
                start_time: Optional[datetime.datetime] = None):
        data_df = accessor.get_feature_asset_data(self.asset, interval=self.interval_size)
        if start_time is not None:
            data_df = data_df[data_df.end_time >= start_time]

        portfolio_handler.execute_order(
            MarketOrder(asset=self.asset, action='buy', quantity=-1, creation_time=data_df.end_time.iloc[0]))

    def fit(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        return {'asset': self.asset, 'interval_size': self.interval_size}

    def __repr__(self):
        return "{} ({}:{})".format(self.agent_name, self.asset, self.interval_size)
