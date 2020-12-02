from typing import Optional, Dict, Any, Type, Tuple
import datetime
import logging
from abc import ABC, abstractmethod
from atapy.constants import *
from atapy.utils import load_classes_in_dir
from atapy.data_accessor import DataAccessor
from atapy.portfolio_handlers.backtest_portfolio_handler import BacktestPortfolioHandler
from atapy.interval import Interval
from typing import Dict


logger = logging.getLogger()


class Agent(ABC, object):
    """
    An initial agent that only trades on the ema_crossover (daily)
    I.e. if the short goes above the long, buy, if it goes below, sell
    """
    interval_size = Interval.five_min  # How often the agent should be called for stepwise assessment

    @property
    @abstractmethod
    def agent_name(self):
        raise RuntimeError

    @abstractmethod
    def __repr__(self):
        raise RuntimeError

    @classmethod
    def factory(cls, agent_name: str, **agent_args):
        load_classes_in_dir(Path(AGENTS_DIR))
        agent: Type[Agent]
        for agent in cls.__subclasses__():
            if agent.agent_name == agent_name:
                return agent(**agent_args)
        raise NotImplementedError("Subclass {} of Agent is not implemented".format(agent_name))

    @abstractmethod
    def execute(self, accessor: DataAccessor, portfolio_handler: BacktestPortfolioHandler,
                start_time: Optional[datetime.datetime] = None) -> None:
        """
        TODO: REWRITE
        Performs a backtest of the strategy from the start_time to the end_time using the data available in the accessor
        updating the portfolio_handler with the trades that would have taken place
        :param accessor:
        :param portfolio_handler:
        :param start_time:
        :return:
        """
        pass

    @abstractmethod
    def fit(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
            courtage: Optional[float or Dict] = None, return_all_res: bool = False) -> Dict[Tuple[str], Any]:
        """
        Trains the agent on the data available in the accessor and returns the best found parameters. These parameters
        should also be the expected input to its init method. The training should take place on the data in the accessor
        from the given start_time til the end of all data available in the accessor
        """
        pass

    @abstractmethod
    def visualize(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
                  include_buy_sell_signals: bool = True, courtage_settings=None, plotlib: str = 'plotly'):
        """ Visualizes the agent over the given period (start_time til end of accessor data) including important
        features and buy and sell signals/portfolio value changes etc """
        pass

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

