from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import datetime
import logging
from tqdm import tqdm
from atapy.constants import COLUMN_NAME_USED_BY_EMA
from atapy.interval import Interval
from atapy.agent import Agent
from atapy.data_accessor import DataAccessor
from atapy.order import MarketOrder, Order
from atapy.asset import Asset
from atapy.utils import to_list, plotly_visualization
from atapy.portfolio_handlers.backtest_portfolio_handler import BacktestPortfolioHandler

logger = logging.getLogger()


class EmaCrossoverAgent(Agent):
    """
    An initial agent that only trades on the ema_crossover
    I.e. if the short goes above the long, buy, if it goes below, sell
    """
    agent_name = 'EmaCrossover'

    # ema_periods = (2, 3, 4, 5, 7, 9, 12, 21, 26, 40, 50, 60, 100, 200), interval_sizes = INTERVAL_SIZES
    def __init__(self, asset: Asset or List[Asset], interval: Interval or List[Interval] = Interval.hourly,
                 short_ema_interval: int or List[int] = 9, long_ema_interval: int or List[int] = 21):
        self.fit_configuration = {
            'asset': to_list(asset),
            'interval': to_list(interval),
            'short_ema_interval': to_list(short_ema_interval),
            'long_ema_interval': to_list(long_ema_interval),
            }
        self.asset = self.fit_configuration['asset'][0]
        self.interval = self.fit_configuration['interval'][0]
        self.short_ema_interval = self.fit_configuration['short_ema_interval'][0]
        self.long_ema_interval = self.fit_configuration['long_ema_interval'][0]

    def step(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
             current_time: Optional[datetime.datetime] = None):
        raise NotImplementedError
        # data_df = accessor.get_feature_asset_data(self.asset, self.interval)
        #
        # # Filter away data after the given time
        # if current_time is not None:
        #     data_df = data_df[data_df.end_time <= current_time]
        # else:
        #     current_time = datetime.datetime.now()
        # # Calculate if there has been a crossover
        # data_df, short_ema_name = self._ensure_ema_available(data_df, self.short_ema_interval)
        # data_df, long_ema_name = self._ensure_ema_available(data_df, self.long_ema_interval)
        #
        # diff_series = data_df[short_ema_name].iloc[-2:] - data_df[long_ema_name].iloc[-2:]
        #
        # if diff_series.iloc[-2] > 0 and diff_series.iloc[-1] < 0:
        #     # Short has gone from above long to below, sell
        #     portfolio_handler.execute_order(
        #         MarketOrder(asset=self.asset, action='sell', quantity=-1, creation_time=current_time))
        # elif diff_series.iloc[-2] < 0 and diff_series.iloc[-1] > 0:
        #     # Short has gone from below long to above: Buy order
        #     portfolio_handler.execute_order(
        #         MarketOrder(asset=self.asset, action='buy', quantity=-1, creation_time=current_time))
        # else:
        #     pass

    @staticmethod
    def _ensure_ema_available(data_df: pd.DataFrame, ema_windowsize: int) -> Tuple[pd.DataFrame, str]:
        col_name = 'ema{}'.format(ema_windowsize)
        if col_name not in data_df.columns:
            data_df[col_name] = data_df[COLUMN_NAME_USED_BY_EMA].ewm(span=ema_windowsize, adjust=False).mean()
        return data_df, col_name

    def visualize(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
                  include_buy_sell_signals: bool = True):
        data_df = accessor.get_feature_asset_data(self.asset, self.interval)
        data_df, short_ema_name = self._ensure_ema_available(data_df, self.short_ema_interval)
        data_df, long_ema_name = self._ensure_ema_available(data_df, self.long_ema_interval)
        if start_time is not None:
            data_df = data_df[data_df.end_time >= start_time]
        data_df = data_df.set_index('end_time')
        candlestick_data = data_df[['open', 'high', 'low', 'close']]
        single_value_data_dict = {long_ema_name: data_df[long_ema_name], short_ema_name: data_df[short_ema_name]}
        if include_buy_sell_signals:
            orders = self.execute(accessor, BacktestPortfolioHandler(10000, accessor), start_time)
            buy_indexes = [order.creation_time for order in orders if order.action == 'buy']
            buy_series = data_df.close[buy_indexes]
            sell_indexes = [order.creation_time for order in orders if order.action == 'sell']
            sell_series = data_df.close.loc[sell_indexes]
        else:
            buy_series = None
            sell_series = None
        plotly_visualization(candlestick_data_df=candlestick_data, single_value_data_dict=single_value_data_dict,
                             buy_signal_series=buy_series, sell_signal_series=sell_series)

    def execute(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
                start_time: Optional[datetime.datetime] = None) -> List[Order]:
        data_df = accessor.get_feature_asset_data(self.asset, interval=self.interval)
        # Do the same as for step, but on a full dataframe at once
        data_df, short_ema_name = self._ensure_ema_available(data_df, self.short_ema_interval)
        data_df, long_ema_name = self._ensure_ema_available(data_df, self.long_ema_interval)

        if start_time is not None:
            data_df = data_df[data_df.end_time > start_time]

        data_df = data_df[['start_time', 'end_time', short_ema_name, long_ema_name]].copy()
        data_df['diff'] = data_df[short_ema_name] - data_df[long_ema_name]
        data_df['prev_diff'] = [None] + data_df['diff'].iloc[:-1].tolist()
        data_df['signal'] = None
        data_df.loc[(data_df['prev_diff'] > 0) & (data_df['diff'] < 0), 'signal'] = 'sell'
        data_df.loc[(data_df['prev_diff'] < 0) & (data_df['diff'] > 0), 'signal'] = 'buy'
        orders = []
        if data_df[short_ema_name].iloc[0] > data_df[long_ema_name].iloc[0]:
            orders.append(MarketOrder(self.asset, 'buy', -1, data_df.end_time.iloc[0]))

        data_df = data_df[data_df.signal.notna()]
        # The info is known just before closing on the start_time date

        orders.extend([MarketOrder(self.asset, row.signal, -1, row.end_time) for index, row in data_df.iterrows()])
        if len(orders) > 0 and self.asset not in portfolio_handler.portfolio.keys() and orders[0].action == 'sell':
            orders = orders[1:]
        portfolio_handler.execute_order(orders)
        return orders

    def fit(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
            courtage: Optional[float or Dict] = None, return_all_res: bool = False) -> Dict[str, Any]:
        configs = []
        res = {}
        all_portfolios = {}
        init_value = 10000
        for asset in self.fit_configuration['asset']:
            for interval in self.fit_configuration['interval']:
                for short_ema_interval in self.fit_configuration['short_ema_interval']:
                    for long_ema_interval in self.fit_configuration['long_ema_interval']:
                        if short_ema_interval < long_ema_interval:
                            configs.append((asset, interval, short_ema_interval, long_ema_interval))
        # For each conf, set the parameters accordingly, then run the backtest
        for conf in tqdm(configs):
            portfolio_handler = BacktestPortfolioHandler(init_value, accessor, courtage=courtage)
            self.asset, self.interval, self.short_ema_interval, self.long_ema_interval = conf
            self.execute(accessor, portfolio_handler, start_time=start_time)
            res[conf] = portfolio_handler.get_average_daily_ratio()
            all_portfolios[conf] = portfolio_handler
        best_res = sorted(res.items(), key=lambda x: x[1])[-1]
        logger.debug('Best res found with return {} with conf {}'.format(best_res[1], best_res[0]))
        best_conf = {'asset': best_res[0][0], 'interval': best_res[0][1], 'short_ema_interval': best_res[0][2],
                     'long_ema_interval': best_res[0][3]}
        self.interval = best_conf['interval']
        self.asset = best_conf['asset']
        self.short_ema_interval = best_conf['short_ema_interval']
        self.long_ema_interval = best_conf['long_ema_interval']
        if not return_all_res:
            return best_conf
        else:
            return all_portfolios

    def __repr__(self):
        return "{} ({}:{} short:{}, long:{})".format(self.agent_name, self.asset, self.interval,
                                                     self.short_ema_interval, self.long_ema_interval)

    __str__ = __repr__
