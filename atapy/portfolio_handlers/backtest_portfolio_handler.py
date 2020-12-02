
import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import math
import logging

from atapy.order import Order, MarketOrder
from atapy.data_accessor import DataAccessor
from atapy.portfolio_handler import PortfolioHandler, CURRENCY_ASSET
from atapy.interval import Interval


logger = logging.getLogger()


class BacktestPortfolioHandler(PortfolioHandler):
    # Keeps track of the portfolio and "forwards" the orders to the broker (in case of realtime system)

    handler_name = 'backtest'

    @property
    def portfolio(self):
        return dict(self._portfolio)

    def __init__(self, initial_currency_value: float, accessor: DataAccessor, courtage: Optional[float or Dict] = 0.0005):
        self._portfolio = {CURRENCY_ASSET: initial_currency_value}
        self.accessor = accessor
        self.history_df = pd.DataFrame()
        if courtage is None:
            courtage = 0
        self.courtage = courtage

    def execute_order(self, order: Order or List[Order]):
        # Make sure the orders are sorted
        if isinstance(order, Order):
            orders = [order]
        elif isinstance(order, list):
            orders = sorted(order)
        else:
            raise NotImplementedError
        # Check that the last performed order is before the first new order
        if self.history_df.shape[0] > 0 and self.history_df[CURRENCY_ASSET].dropna().index[-1] > orders[0].creation_time:
            raise NotImplementedError("Unable to handle case when orders are not given sequentially")

        for order in orders:
            logger.debug(order)
            if not isinstance(order, MarketOrder):
                raise NotImplementedError("Only market orders are currently implemented for BacktestPortfolioHandler")
            if order.action == 'buy':
                # First check if we need to collect more data from the accessor for later use
                if (order.asset, 'price') not in self.history_df.columns:
                    new_asset_data = self.accessor.get_feature_asset_data(order.asset, Interval.five_min)
                    new_asset_data = new_asset_data[['end_time', 'close']].set_index('end_time')
                    new_asset_data.columns = pd.MultiIndex.from_tuples([(order.asset, 'price')], names=['asset', 'type'])
                    # If it is the first asset added to the history_df
                    if self.history_df.shape[0] == 0:
                        self.history_df = new_asset_data
                        self.history_df[(CURRENCY_ASSET, 'price')] = 1
                        self.history_df[(CURRENCY_ASSET, 'quantity')] = np.nan
                        self.history_df.loc[self.history_df.index[0], (CURRENCY_ASSET, 'quantity')] = self.portfolio[CURRENCY_ASSET]
                    else:
                        self.history_df = self.history_df.merge(
                            new_asset_data, how='outer', left_index=True, right_index=True)
                    self.history_df[(order.asset, 'quantity')] = np.nan
                    self.history_df.loc[self.history_df.index[0], (order.asset, 'quantity')] = 0
                # Calculate price and quantity
                current_price = self.history_df.loc[order.creation_time, (order.asset, 'price')]
                if order.quantity == -1:
                    quantity = self.portfolio[CURRENCY_ASSET] / (current_price * (1 + self.courtage))
                    if order.asset.type == 'crypto':
                        minimal_size = 1e-8
                        quantity = math.floor(quantity / minimal_size) * minimal_size
                    else:
                        quantity = math.floor(quantity)
                else:
                    quantity = order.quantity
                full_order_price = current_price * quantity
                courtage_cost = self.courtage * full_order_price
                if full_order_price + courtage_cost > self.portfolio[CURRENCY_ASSET]:
                    logger.error("Unable to fulfill order due to insufficient funds. "
                                 + "Order will be scrapped {}".format(order))
                    continue
                else:
                    # change the portfolio
                    self._portfolio[order.asset] = quantity + self._portfolio[
                        order.asset] if order.asset in self._portfolio.keys() else quantity
                    # Round to fix computational errors
                    self._portfolio[CURRENCY_ASSET] = round(self._portfolio[CURRENCY_ASSET] - full_order_price - courtage_cost, 2)
                    # Update the history_df accordingly
                    self.history_df.loc[order.creation_time, (CURRENCY_ASSET, 'quantity')] = self._portfolio[CURRENCY_ASSET]
                    self.history_df.loc[order.creation_time, (order.asset, 'quantity')] = self._portfolio[order.asset]
            if order.action == 'sell':
                # Check that there are the portfolio contains the assets
                if order.asset not in self._portfolio.keys() or order.quantity > self._portfolio[order.asset]:
                    logger.error("Unable to fulfill order due to insufficient owned quantity. "
                                 + " Order will be scrapped {}".format(order))
                    continue
                else:
                    current_price = self.history_df.loc[order.creation_time, (order.asset, 'price')]
                    quantity = self._portfolio[order.asset] if order.quantity == -1 else order.quantity
                    full_order_price = current_price * quantity
                    courtage_cost = full_order_price*self.courtage
                    # Then update the porfolio and history_df accordingly
                    self._portfolio[order.asset] = self._portfolio[order.asset] - quantity
                    self._portfolio[CURRENCY_ASSET] = self._portfolio[CURRENCY_ASSET] + full_order_price - courtage_cost
                    self.history_df.loc[order.creation_time, (CURRENCY_ASSET, 'quantity')] = self._portfolio[CURRENCY_ASSET]
                    self.history_df.loc[order.creation_time, (order.asset, 'quantity')] = \
                        self._portfolio[order.asset]

    def __repr__(self):
        return "BacktestPorfolioHandler with accessor {}".format(self.accessor)

    __str__ = __repr__
