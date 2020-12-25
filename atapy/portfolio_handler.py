
import datetime
from typing import List, Optional, Dict, Type
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from atapy.order import Order
from atapy.utils.method_utils import load_classes_in_dir
from atapy.constants import *
import matplotlib.pyplot as plt
from atapy.asset import Asset
import math


logger = logging.getLogger()

# Todo: add this as init input and instance parameter, and also handle multiple currencies
CURRENCY_ASSET = Asset('na', 'currency', 'currency')


class PortfolioHandler(ABC):
    portfolio: Dict[Asset, float] = {}
    history_df: pd.DataFrame = None  # Dataframe with index end_time and hierarchical columns asset, type

    @property
    @abstractmethod
    def handler_name(self):
        raise RuntimeError

    @classmethod
    def factory(cls, handler_name, *handler_args):
        load_classes_in_dir(Path(PORTFOLIO_HANDLERS_DIR))
        handler: Type[PortfolioHandler]
        for handler in cls.__subclasses__():
            if handler.handler_name == handler_name:
                return handler(*handler_args)
        raise NotImplementedError("Subclass {} of PortfolioHandler is not implemented".format(handler_name))

    @abstractmethod
    def execute_order(self, order: Order or List[Order]) -> None:
        """ Executes the order or list of orders given in the portfolio """
        pass

    def visualize(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None, plotlib='plotly',
                  **xargs):
        """ Visualizes the portfolio value and the changes in it with comparison to traded assets """
        data = self.history_df.fillna(method='ffill')
        if start_time is not None:
            data = data[data.index >= start_time]
        if end_time is not None:
            data = data[data.index < end_time]
        assets = set(data.columns.get_level_values('asset'))
        prices = data.loc[:, [(asset, 'price') for asset in assets if asset != CURRENCY_ASSET]]
        total_value = self.calculate_value(start_time=start_time, end_time=end_time)
        relative_prices = prices / prices.iloc[0, :] * total_value.iloc[0]
        order_values = -(data[(CURRENCY_ASSET, 'price')] * data[
            (CURRENCY_ASSET, 'quantity')]).diff()  # positive means buy, negative sell
        buy_signals = total_value.loc[order_values > 0]
        sell_signals = total_value.loc[order_values < 0]
        if plotlib == 'plotly':
            from plotly.graph_objs import Figure
            from plotly.graph_objects import Scatter
            import plotly.graph_objects as go
            # Trace scatters for the total
            portfolio_value_trace = go.Scatter(name='portfolio_value', x=total_value.index, y=total_value.values,
                                               xaxis='x1', yaxis='y1')
            # Scatters for relative prices
            relative_price_tracers = [
                go.Scatter(name=colname[0].symbol, x=val.index, y=val.values, xaxis='x1', yaxis='y1') for colname, val
                in relative_prices.iteritems()]
            order_value_trace = go.Scatter(name="buy/sell", x=order_values.index, y=order_values.values, xaxis='x1',
                                           yaxis='y2')
            buy_trace = go.Scatter(name='buy', mode='markers', x=buy_signals.index, y=buy_signals.values,
                                   marker={"size": 12, "color": "green"}, marker_symbol='triangle-up', xaxis='x1',
                                   yaxis='y1')
            sell_trace = go.Scatter(name='sell', mode='markers', x=sell_signals.index, y=sell_signals.values,
                                    marker={"size": 12, "color": "red"}, marker_symbol='triangle-down', xaxis='x1',
                                    yaxis='y1')

            layout = {
                'autosize': False,
                'width': 1200,
                'height': 1000,
                "xaxis1": {
                    "anchor": "y3",
                    "domain": [0.0, 1.0]
                },
                "yaxis1": {
                    "anchor": "free",
                    "domain": [0.2, 1],
                    "position": 0.0
                },
                "yaxis2": {
                    "anchor": "free",
                    "domain": [0, 0.15],
                    "position": 0.0
                },
            }
            fig = go.Figure(
                data=[portfolio_value_trace] + relative_price_tracers + [order_value_trace, buy_trace, sell_trace],
                layout=layout, )
            fig.show()
        elif plotlib == 'matplotlib':
            if 'ax' in xargs.keys():
                ax = xargs['ax']
            else:
                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            total_value.plot(ax=ax)
            for colname, val in relative_prices.iteritems():
                val.plot(ax=ax)
        else:
            raise ValueError("Unknown plotlib value {}".format(plotlib))

    def calculate_value(self, start_time: Optional[datetime.datetime] = None, end_time: Optional[datetime.datetime] = None
                        ) -> pd.Series or np.float64:
        """ Returns the value of the portfolio for the timeperiod [start_time, end_time["""
        if self.history_df.shape[0] == 0:
            return pd.Series(self.portfolio[CURRENCY_ASSET])
        df = self.history_df.fillna(method='ffill')
        if start_time is not None:
            df = df[df.index >= start_time]
        if end_time is not None:
            df = df[df.index < end_time]
        for asset in set(df.columns.get_level_values('asset')):
            df[(asset, 'value')] = df[(asset, 'price')] * df[(asset, 'quantity')]
        total_portfolio_value_series = df.loc[:, pd.IndexSlice[:, 'value']].fillna(0).sum(axis=1)
        return total_portfolio_value_series

    def get_total_ratio(self, start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None) -> float:
        """ Calculates the total gain/loss of the portfolio during the given period """
        if self.history_df.shape[0] == 0:
            return 1.0
        calculated_df = self.calculate_value(start_time, end_time)
        calculated_df = calculated_df/calculated_df.iloc[0]
        return calculated_df.iloc[-1]

    def get_average_daily_ratio(self, start_time: Optional[datetime.datetime] = None,
                                end_time: Optional[datetime.datetime] = None) -> float:
        """ Calculates the daily gain/loss ratio of the portfolio during the given period """
        if self.history_df.shape[0] == 0:
            return 1.0
        calculated_df = self.calculate_value(start_time, end_time)
        calculated_df = calculated_df / calculated_df.iloc[0]
        total_nr_of_days = (calculated_df.index[-1].date() - calculated_df.index[0].date()).days
        return math.pow(calculated_df.iloc[-1], 1/total_nr_of_days)

