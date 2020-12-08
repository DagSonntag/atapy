from typing import Optional, Dict, Any, Tuple, List, Callable, Collection, Hashable
import pandas as pd
import datetime
import logging
from tqdm import tqdm
import itertools
import copy
import tulipy as ty
import numpy as np
from atapy.interval import Interval
from atapy.agent import Agent
from atapy.data_accessor import DataAccessor
from atapy.order import MarketOrder, Order
from atapy.asset import Asset
from atapy.utils import to_list, feature_visualization
from atapy.portfolio_handlers.backtest_portfolio_handler import BacktestPortfolioHandler

logger = logging.getLogger()


def tulipy_generic_feature_func(data_df, ty_func, column_name_used_for_calc='close', **xargs):
    res = ty_func(data_df[column_name_used_for_calc].values, **xargs)
    if isinstance(res, np.ndarray):
        # Handle strange occurrences when very small periods are used
        if res.shape[0] > data_df.shape[0]:
            res = res[res.shape[0] - data_df.shape[0]:]

        if res.shape[0] == data_df.shape[0]:
            return pd.Series(res, index=data_df.index)
        else:
            return pd.Series(res, index=data_df.index[data_df.shape[0]-res.shape[0]:])
    else:
        raise NotImplementedError


def ema_feature_func(data_df, period, column_name_used_for_calc='close'):
    return pd.Series(ty.ema(data_df[column_name_used_for_calc].values, period), index=data_df.index)


def hull_moving_average_feature_func(data_df, period, column_name_used_for_calc='close'):
    return tulipy_generic_feature_func(data_df, ty.hma, column_name_used_for_calc=column_name_used_for_calc,
                                       period=period)


def kaufmans_moving_average_feature_func(data_df, period, column_name_used_for_calc='close'):
    return tulipy_generic_feature_func(data_df, ty.kama, column_name_used_for_calc=column_name_used_for_calc,
                                       period=period)


def zero_lag_ema_feature_func(data_df, period, column_name_used_for_calc='close'):
    return tulipy_generic_feature_func(data_df, ty.zlema, column_name_used_for_calc=column_name_used_for_calc,
                                       period=period)


def triple_ema_feature_func(data_df, period, column_name_used_for_calc='close'):
    """
    The Triple Exponential Moving Average is similar to the Exponential Moving Average or the
    Double Exponential Moving Average, but provides even less lag. Triple Exponential Moving Average is probably best
    viewed as an extension of Double Exponential Moving Average."""
    return tulipy_generic_feature_func(data_df, ty.tema, column_name_used_for_calc=column_name_used_for_calc,
                                       period=period)


MOVING_AVERAGES_DICT = {'sma': ty.sma, 'ema': ty.ema, 'dema': ty.dema, 'tema': ty.tema, 'kama': ty.kama,
                        'hma': ty.hma, 'zlema': ty.zlema}
def moving_average_feature_func(data_df, period,  column_name_used_for_calc='close', method='ema'):
    """ A handling function for the different ways of calculating moving averages. 
    See tulipy for details on the different methods. Possible values are """
    if method in MOVING_AVERAGES_DICT.keys():
        return tulipy_generic_feature_func(data_df, MOVING_AVERAGES_DICT[method],
                                           column_name_used_for_calc=column_name_used_for_calc, period=period)
    else:
        raise NotImplementedError("Unknown method {} to calculate moving averages".format(method))
moving_average_feature_func.__doc__ += str(list(MOVING_AVERAGES_DICT.keys()))

def moving_average_with_delay_feature_func(data_df, period,  delay=24, column_name_used_for_calc='close', method='ema'):
    res = moving_average_feature_func(data_df, period, column_name_used_for_calc, method)
    return res.shift(delay).dropna()

def zlema_velocity_feature_func(data_df, period, window, column_name_used_for_calc='close'):
    zlema = moving_average_feature_func(data_df, period, column_name_used_for_calc, method='zlema')
    return zlema - zlema.shift(window)

def bbands_feature_func(data_df, period, n_stddev, column_name_used_for_calc='close', tulipy_mean_func=ty.sma):
    mean = tulipy_mean_func(data_df[column_name_used_for_calc].values, period=period)
    std = ty.stddev(data_df[column_name_used_for_calc].values, period)
    df = pd.DataFrame([mean[(mean.shape[0]-std.shape[0]):]-n_stddev*std, mean[(mean.shape[0]-std.shape[0]):],
                       mean[(mean.shape[0]-std.shape[0]):]+n_stddev*std]).T
    df.index = data_df.index[(data_df.shape[0]-df.index.shape[0]):]
    return df


def percentage_bands_feature_func(data_df, period, band_percentage, column_name_used_for_calc='close',
                                  tulipy_mean_func=ty.sma):
    mean = tulipy_mean_func(data_df[column_name_used_for_calc].values, period=period)
    diff = mean*band_percentage/100
    df = pd.DataFrame([mean-diff, mean, mean+diff]).T
    df.index = data_df.index[(data_df.shape[0]-df.index.shape[0]):]
    return df

def macd_feature_func(data_df, short_period, long_period, signal_period, column_name_used_for_calc='close'):
    res = ty.macd(data_df[column_name_used_for_calc].values, short_period, long_period, signal_period)
    df = pd.DataFrame(res)
    df.columns = data_df.index[data_df.shape[0] - df.shape[1]:]
    return df.T


def rsi_feature_func(data_df, period, column_name_used_for_calc='close'):
    return tulipy_generic_feature_func(data_df, ty.rsi, column_name_used_for_calc=column_name_used_for_calc,
                                       period=period)

def channel_decision_func(data_df, col_name, buy_value, sell_value):
    data_df = data_df[[col_name]].copy()
    data_df['signal'] = None
    if buy_value >= sell_value:
        # Buy when the value goes above the buy_value and sell when below the sell_value
        data_df.loc[data_df[col_name] > buy_value, 'signal'] = 'buy'
        data_df.loc[data_df[col_name] < sell_value, 'signal'] = 'sell'
    else:
        # Buy when the value goes below the buy value and sell when above the sell value (normal RSI case)
        data_df.loc[data_df[col_name] < buy_value, 'signal'] = 'buy'
        data_df.loc[data_df[col_name] > sell_value, 'signal'] = 'sell'
    return data_df.signal


def crossover_decision_func(data_df, short_col_name, long_col_name):
    data_df = data_df.copy()
    data_df = data_df[['start_time', 'end_time', short_col_name, long_col_name]].copy()
    data_df['diff'] = data_df[short_col_name] - data_df[long_col_name]
    data_df['signal'] = None
    data_df.loc[data_df['diff'] < 0, 'signal'] = 'sell'
    data_df.loc[data_df['diff'] > 0, 'signal'] = 'buy'
    return data_df.signal


def triple_crossover_decision_func(data_df, short_col_name, medium_col_name, long_col_name,
                                   sub_strategy='short_to_both_crossover'):
    data_df = data_df.copy()
    data_df = data_df[['start_time', 'end_time', short_col_name, medium_col_name, long_col_name]].copy()
    data_df['short_to_medium_diff'] = data_df[short_col_name] - data_df[medium_col_name]
    data_df['short_to_long_diff'] = data_df[short_col_name] - data_df[long_col_name]
    data_df['medium_to_long_diff'] = data_df[medium_col_name] - data_df[long_col_name]

    data_df['major_trend'] = ""
    # When the 55 EMA is below both the 9 and 21, we will consider the trend to be up
    data_df.loc[(data_df['medium_to_long_diff'] > 0) & (data_df['short_to_long_diff'] > 0), 'major_trend'] = 'up'
    # When the indicator is above both of the shorter term moving averages, we will consider the longer term trend to
    # be down
    data_df.loc[(data_df['medium_to_long_diff'] < 0) & (data_df['short_to_long_diff'] < 0), 'major_trend'] = 'down'
    # Medium trends
    data_df['medium_trend'] = ""
    # We want to see the 21 below the 9 and above the 55 for an uptrend
    data_df.loc[(data_df['medium_to_long_diff'] > 0) & (data_df['short_to_medium_diff'] > 0), 'medium_trend'] = 'up'
    # The 21 should be above the 9 and below the 55 for a down trend
    data_df.loc[(data_df['medium_to_long_diff'] < 0) & (data_df['short_to_medium_diff'] < 0), 'medium_trend'] = 'down'
    data_df['short_to_both'] = ""
    # The 9 EMA crossing over the 21 while already above the 55, is an uptrend and looking for a buy trade
    data_df.loc[(data_df['short_to_long_diff'] > 0) & (data_df['short_to_medium_diff'] > 0), 'short_to_both'] = 'up'
    # If it crosses below the 21 while already below the 55, that is a down trend and looking for a sell trade
    data_df.loc[(data_df['short_to_long_diff'] < 0) & (data_df['short_to_medium_diff'] < 0), 'short_to_both'] = 'down'

    if sub_strategy == 'short_to_both_crossover':
        data_df['signal'] = None
        data_df.loc[data_df['short_to_both'] == 'up', 'signal'] = 'buy'
        data_df.loc[data_df['short_to_both'] == 'down', 'signal'] = 'sell'
    elif sub_strategy == 'major_and_minor_crossover':
        # Buys and sells if both trends gives the same indication, as well as if the short trend is changing on the
        # opposite side of the major trend. I.e. if both medium and short is smaller than the long ma, and short is
        # crossing medium downwards, this gives a sell signal
        data_df['signal'] = None
        data_df.loc[data_df['short_to_both'] == 'up', 'signal'] = 'buy'
        data_df.loc[data_df['short_to_both'] == 'down', 'signal'] = 'sell'
        # if the short is above medium but below long buy
        data_df.loc[(data_df['short_to_medium_diff'] > 0)
                    & (data_df['short_to_long_diff'] < 0)
                    & (data_df['medium_to_long_diff'] < 0), 'signal'] = 'buy'
        # If the short is below medium but above long, sell
        data_df.loc[(data_df['short_to_medium_diff'] < 0)
                    & (data_df['short_to_long_diff'] > 0)
                    & (data_df['medium_to_long_diff'] > 0), 'signal'] = 'sell'
    elif sub_strategy == 'major_to_both':
        data_df['signal'] = None
        data_df.loc[data_df['major_trend'] == 'up', 'signal'] = 'buy'
        data_df.loc[data_df['major_trend'] == 'down', 'signal'] = 'sell'
    else:
        raise NotImplementedError
    return data_df.signal


def zero_crossover_decision_func(data_df, col_name, reverse=False):
    """ Returns buy signals when the value in the column is above 0, sell otherwise """
    data_df = data_df.copy()
    data_df['signal'] = None
    if not reverse:
        data_df.loc[data_df[col_name] > 0, 'signal'] = 'buy'
        data_df.loc[data_df[col_name] < 0, 'signal'] = 'sell'
    else:
        data_df.loc[data_df[col_name] < 0, 'signal'] = 'buy'
        data_df.loc[data_df[col_name] > 0, 'signal'] = 'sell'
    return data_df.signal


class SingleAssetFeatureDecisionAgent(Agent):
    """
    A general agent type that can use several different very similar strategies. The requirement is that the strategies
    should work as follows:
    * They generate a set of features given the market data
    * From these features they use a decision engine to make a decision [buy, do_nothing, sell] for each sample
    * These decisions are then transferred into orders by removing do_nothing and any consecutive duplicates

    This means that for each strategy two functions must be implemented, a feature function generating features from the
    feature data (returning the 'enhanced' feature data) and the decision engine returning a decision for each sample
    Implemented strategies are:
    * ema_crossover(short_ema_period, long_ema_period)
    * macd strategy
    * Bollinger bands

    To allow fitting of the methods the function and possibly parameter values must be passed to the init method as
    follows:
    * feature_dict = {feature_name: (function, function_parameter_dict)} where function_parameter_dict is a dict of the
    {argument_name: List[possible_values]}. Unfitted agents will use the first of the possible values.
    If a single feature column is returned, the feature name should be a string, and the data returned as a series. If
    multiple feature columns are returned, the feature name should be a tuple, and the data returned as a dataframe
    * The decision function takes a tuple (function, function_parameter_dict) similarly as for the features
    * active_interval: Interval or List[Interval] is in which interval sampling the agent should be active

    This will all later be wrapped up in nice subclasses or factories

    For the feature functions and the decision functions the first argument should always be the data_df

    """

    def set_agent_properties(self, **configuration_dict) -> None:
        pass

    agent_name = 'SingleAssetFeatureDecisionAgent'
    run_interval = Interval.five_min

    def __init__(self, agent_name: str,
                 asset: Asset,
                 feature_dict: Dict[str or Tuple[str], Tuple[Callable, Dict[str, Collection[Any]]]],
                 decision_func: Callable,
                 decision_func_var_dict: Dict[str, Collection[Any]],
                 interval: Interval or List[Interval]):
        self.agent_name = agent_name
        self.fit_configuration = {
            'asset': asset,
            'feature_dict': feature_dict,
            'decision_func': decision_func,
            'decision_func_var_dict': decision_func_var_dict,
            'interval': to_list(interval),
        }
        self.asset = asset
        self.feature_dict = {feature_name: (feature_func,
                                            {feature_func_var_name: value_list[0]
                                             for feature_func_var_name, value_list in feature_parameter_dict.items()})
                             for feature_name, (feature_func, feature_parameter_dict) in feature_dict.items()}
        self.decision_func = decision_func
        self.decision_func_var_dict = {decision_func_var_name: value_list[0]
                                       for decision_func_var_name, value_list in decision_func_var_dict.items()}
        self.run_interval = self.fit_configuration['interval'][0]

    def step(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
             current_time: Optional[datetime.datetime] = None) -> None:
        pass

    def execute(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
                start_time: Optional[datetime.datetime] = None) -> List[Order]:
        data_df = accessor.get_feature_asset_data(self.asset, interval=self.run_interval)
        for feature_name, (feature_func, feature_func_var_dict) in self.feature_dict.items():
            if isinstance(feature_name, tuple):
                feature_name = list(feature_name)
            data_df[feature_name] = feature_func(data_df=data_df, **feature_func_var_dict)
        # Only do calculations for the period (earlier data only used for feature calculations)
        if start_time is not None:
            data_df = data_df[data_df.end_time > start_time]
        # Calculate the decisions
        data_df['decision'] = self.decision_func(data_df=data_df, **self.decision_func_var_dict)
        decision_series = data_df.set_index('end_time')['decision']
        decision_series = decision_series.dropna()
        # Remove consecutive duplicate orders
        decision_series = decision_series.loc[decision_series.shift(1) != decision_series]
        # create orders
        orders = [MarketOrder(self.asset, val, -1, index) for index, val in decision_series.iteritems()]
        if len(orders) > 0 and orders[0].action == 'sell' and (
                self.asset not in portfolio_handler.portfolio.keys() or portfolio_handler.portfolio[self.asset] == 0):
            orders = orders[1:]
        portfolio_handler.execute_order(orders)
        return orders

    def set_agent_conf(self, conf):
        for setting in conf:
            if setting[0] == 'feature':
                self.feature_dict[setting[1]][1][setting[2]] = setting[3]
            elif setting[0] == 'decision':
                self.decision_func_var_dict[setting[1]] = setting[2]
            elif setting[0] == 'interval':
                self.run_interval = setting[1]
            else:
                raise NotImplementedError("Unknown handling of setting {}".format(setting))

    def fit(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
            courtage: Optional[float or Dict] = None, return_all_res: bool = False) -> Dict[str, Any]:
        init_value = 10000
        configs = []
        for feature_name, (feature_func, feature_var_dict) in self.fit_configuration['feature_dict'].items():
            for feature_var_name, feature_var_values in feature_var_dict.items():
                configs.append([('feature', feature_name, feature_var_name, feature_var_value) for feature_var_value in
                                feature_var_values])
        # Then do the same with decision-func_parameters
        for decision_var_name, decision_var_values in self.fit_configuration['decision_func_var_dict'].items():
            configs.append(
                [('decision', decision_var_name, decision_var_value) for decision_var_value in decision_var_values])
        configs.append([('interval', interval) for interval in self.fit_configuration['interval']])
        all_configs = list(itertools.product(*configs))

        res = {}
        all_res = {}
        for conf in tqdm(all_configs):
            # Set the agent parameter correctly
            self.set_agent_conf(conf)
            portfolio_handler = BacktestPortfolioHandler(init_value, accessor, courtage=courtage)
            self.execute(accessor, portfolio_handler, start_time=start_time)
            res[conf] = portfolio_handler.get_average_daily_ratio()
            all_res[conf] = portfolio_handler.get_total_ratio()
        best_res = sorted(res.items(), key=lambda x: x[1])[-1]
        logger.debug('Best res found with return {} with conf {}'.format(best_res[1], best_res[0]))
        self.set_agent_conf(best_res[0])
        # Go back from single value features and decision function values to multi-values to be called using init

        if not return_all_res:
            feature_dict = copy.deepcopy(self.fit_configuration['feature_dict'])
            decision_var_dict = copy.deepcopy(self.fit_configuration['decision_func_var_dict'])
            interval = []
            for setting in best_res[0]:
                if setting[0] == 'feature':
                    feature_dict[setting[1]][1][setting[2]] = [setting[3]]
                elif setting[0] == 'decision':
                    decision_var_dict[setting[1]] = [setting[2]]
                elif setting[0] == 'interval':
                    interval = setting[1]
                else:
                    raise NotImplementedError("Unknown handling of setting {}".format(setting))
            return {'agent_name': self.agent_name,
                    'asset': self.asset,
                    'feature_dict': feature_dict,
                    'decision_func': self.decision_func,
                    'decision_func_var_dict': decision_var_dict,
                    'interval': interval}
        else:
            return all_res

    def visualize(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
                  include_buy_sell_signals: bool = True, courtage_settings=None,  plotlib: str = 'plotly',
                  **plotlib_args) -> None:
        if plotlib != 'plotly':
            raise NotImplementedError("plotlib {} not implemented".format(plotlib))
        data_df = accessor.get_feature_asset_data(self.asset, self.run_interval)
        feature_data_dict = {}
        for feature_name, (feature_func, feature_func_var_dict) in self.feature_dict.items():
            feature_data_dict[feature_name] = feature_func(data_df=data_df, **feature_func_var_dict)
            feature_data_dict[feature_name].index = data_df.end_time.loc[feature_data_dict[feature_name].index]
        if start_time is not None:
            data_df = data_df[data_df.end_time >= start_time]
            for feature_name in feature_data_dict.keys():
                feature_data_dict[feature_name] = feature_data_dict[feature_name][
                    feature_data_dict[feature_name].index >= start_time]
        # Handle features with multiple output columns
        for feature_name, feature_data in list(feature_data_dict.items()):
            if isinstance(feature_name, tuple):
                for i in range(len(feature_name)):
                    feature_data_dict[feature_name[i]] = feature_data.iloc[:, i]
                feature_data_dict.pop(feature_name)
        data_df = data_df.set_index('end_time')
        candlestick_data = data_df[['open', 'high', 'low', 'close']]
        if include_buy_sell_signals:
            test_portfolio = BacktestPortfolioHandler(10000, accessor, courtage=courtage_settings)
            orders = self.execute(accessor, test_portfolio, start_time)
            buy_indexes = [order.creation_time for order in orders if order.action == 'buy']
            buy_series = data_df.close[buy_indexes]
            sell_indexes = [order.creation_time for order in orders if order.action == 'sell']
            sell_series = data_df.close.loc[sell_indexes]

            # Also include the portfolio value, but normalized to the value
            time_zero_value = accessor.get_time_restricted_instance(start_time=start_time).get_feature_asset_data(
                asset=self.asset, interval=self.run_interval).close.iloc[0]
            port_value = test_portfolio.calculate_value(start_time=start_time)
            if port_value.shape[0] > 1:
                feature_data_dict['relative_portfolio_value'] = port_value/port_value.iloc[0]*time_zero_value
        else:
            buy_series = None
            sell_series = None
        feature_visualization(candlestick_data_df=candlestick_data, single_value_data_dict=feature_data_dict,
                              buy_signal_series=buy_series, sell_signal_series=sell_series, plotlib=plotlib)

    def __repr__(self):
        return self.agent_name


class CrossoverAgent(SingleAssetFeatureDecisionAgent):

    def __init__(self, asset: Asset,  short_periods: Tuple[int] = (9,), long_periods: Tuple[int] = (55,),
                 moving_average_method: Tuple[str] = ('ema', ), interval: Interval or List[Interval] = Interval.hourly):
        super().__init__(agent_name='EmaAgent',
                         asset=asset,
                         feature_dict={'ma_short': (moving_average_feature_func, {'period': short_periods,
                                                                                  'method': moving_average_method}),
                                       'ma_long': (moving_average_feature_func, {'period': long_periods,
                                                                                 'method': moving_average_method})},
                         decision_func=crossover_decision_func,
                         decision_func_var_dict={'short_col_name': ['ma_short'], 'long_col_name': ['ma_long']},
                         interval=interval)


class TripleCrossoverAgent(SingleAssetFeatureDecisionAgent):

    def __init__(self, asset: Asset, short_periods: Tuple[int] = (9,), medium_periods: Tuple[int] = (21,),
                 long_periods: Tuple[int] = (55,),
                 moving_average_method: Tuple[str] = ('ema', ), interval: Interval or List[Interval] = Interval.hourly):
        super().__init__(
            agent_name='TripleEmaAgent',
            asset=asset,
            feature_dict={'ma_short': (moving_average_feature_func, {'period': short_periods,
                                                                     'method': moving_average_method}),
                          'ma_medium': (moving_average_feature_func, {'period': medium_periods,
                                                                      'method': moving_average_method}),
                          'ma_long': (moving_average_feature_func, {'period': long_periods,
                                                                    'method': moving_average_method})},
            decision_func=triple_crossover_decision_func,
            decision_func_var_dict={'short_col_name': ['ma_short'],
                                    'medium_col_name': ['ma_medium'],
                                    'long_col_name': ['ma_long']},
            interval=interval)


class RsiCrossoverAgent(SingleAssetFeatureDecisionAgent):

    def __init__(self, asset, periods: Tuple[int] = (12,), buy_percentages: Tuple[float] = (30, ),
                 sell_percentages: Tuple[float] = (70, ), interval: Interval or List[Interval] = Interval.hourly):
        super().__init__(
            agent_name='RsiAgent',
            asset=asset,
            feature_dict={'rsi': (rsi_feature_func, {'period': periods})},
            decision_func=channel_decision_func,
            decision_func_var_dict={'col_name': ['rsi'], 'buy_value': buy_percentages, 'sell_value': sell_percentages},
            interval=interval)


class MacdCrossoverAgent(SingleAssetFeatureDecisionAgent):

    def __init__(self, asset: Asset, short_periods: Tuple[int] = (12, ), long_periods: Tuple[int] = (26, ),
                 signal_periods: Tuple[int] = (9, ), interval: Interval or List[Interval] = Interval.hourly):
        super().__init__(agent_name='MacdCrossover',
                         asset=asset,
                         feature_dict={('macd', 'macd_signal', 'macd_histogram'): (macd_feature_func,
                                                                                   {'short_period': short_periods,
                                                                                    'long_period': long_periods,
                                                                                    'signal_period': signal_periods})},
                         decision_func=crossover_decision_func,
                         decision_func_var_dict={'short_col_name': ['macd'], 'long_col_name': ['macd_signal']},
                         interval=interval)
