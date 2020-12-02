from typing import List, Optional, Dict
import datetime
from atapy.agent import Agent
from atapy.agents.buy_and_hold_agent import BuyAndHoldAgent
from atapy.data_accessor import DataAccessor
from atapy.asset import Asset
from atapy.portfolio_handlers.backtest_portfolio_handler import BacktestPortfolioHandler
import pandas as pd
from atapy.interval import Interval
import logging

INIT_BACKTEST_VALUE: int = 10000
logger = logging.getLogger()


class AgentHandler:

    def __init__(self, agent: List[Agent] or Agent, accessor: DataAccessor):
        if isinstance(agent, list):
            self.agents = agent
        else:
            self.agents = [agent]
        self.accessor = accessor

    def sliding_window_backtest(
            self, training_window_days: int, testing_window_days: int,  start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None, courtage: Optional[float or Dict] = None) -> Dict:
        # Performs a sliding window training and testing setup where the agents are trained on a set of data, then
        # evaluated on the next testing_window_days nr of days, whereafter the window is moved forward
        # testing_window_days nr of days

        results = {}
        for agent in self.agents:
            portfolio = BacktestPortfolioHandler(initial_currency_value=INIT_BACKTEST_VALUE,
                                                 accessor=self.accessor.get_time_restricted_instance(
                                                     start_time=start_time, end_time=end_time),
                                                 courtage=courtage)
            prev_portfolio_value = INIT_BACKTEST_VALUE
            sliding_window_test_start_time = start_time
            sliding_window_test_end_time = min(
                sliding_window_test_start_time + datetime.timedelta(days=testing_window_days), end_time)
            best_conf = []
            logger.info("Running agent: {}".format(agent))
            while end_time > sliding_window_test_start_time:
                sliding_window_train_end_time = sliding_window_test_start_time
                sliding_window_train_start_time = sliding_window_train_end_time - datetime.timedelta(
                    training_window_days)
                train_accessor = self.accessor.get_time_restricted_instance(end_time=sliding_window_test_start_time)
                test_accessor = self.accessor.get_time_restricted_instance(end_time=sliding_window_test_end_time)
                best_conf.append(agent.fit(train_accessor, start_time=sliding_window_train_start_time,
                                           courtage=courtage))
                orders = agent.execute(test_accessor, portfolio_handler=portfolio,
                                       start_time=sliding_window_test_start_time)

                current_portfolio_value = portfolio.calculate_value().loc[sliding_window_test_end_time]
                logger.info('train {} to {}, test {} to {}'.format(
                    sliding_window_train_start_time, sliding_window_train_end_time, sliding_window_test_start_time,
                    sliding_window_test_end_time
                ))
                logger.info('portfolio_val_change: {}, nr_of_trades: {}, Best conf: {}'.format(
                    current_portfolio_value - prev_portfolio_value, len(orders), best_conf[-1]))
                prev_portfolio_value = current_portfolio_value
                sliding_window_test_start_time = sliding_window_test_start_time + datetime.timedelta(
                    days=testing_window_days)
                sliding_window_test_end_time = min(
                    sliding_window_test_end_time + datetime.timedelta(days=testing_window_days), end_time)
            results[agent] = portfolio
        return results

