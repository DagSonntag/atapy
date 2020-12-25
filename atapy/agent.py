from typing import Optional, Dict, Any, Type, Tuple, List
import datetime
import logging
from abc import ABC, abstractmethod
from atapy.constants import *
from atapy.utils.method_utils import load_classes_in_dir
from atapy.utils.datetime_utils import to_utc_milliseconds, to_datetime
from atapy.data_accessor import DataAccessor
from atapy.portfolio_handler import PortfolioHandler
from atapy.interval import Interval
from atapy.order import Order


logger = logging.getLogger()


class Agent(ABC, object):
    """
    The interface class for the agent with the methods and properties required for implementation, as well as some
    general methods and the factory to create agents by name.

    An agent (sub)class is responsible for:

    * Calculate the actions taken by the agent, i.e. execute orders on a portfolio given selected existing data
    * Train (fit) the agent given selected existing data
    * Visualize the agents behaviour given selected existing data
    """

    @property
    @abstractmethod
    def run_interval(self) -> Interval:
        """ How often the agent should be executed to be up to date. """
        raise RuntimeError

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """ The name of the agent. Used when instantiated from the factory. """
        raise RuntimeError

    @abstractmethod
    def __repr__(self) -> str:
        raise RuntimeError

    @classmethod
    def factory(cls, agent_name: str, *agent_args, **agent_kwargs):
        """
        Create an agent (subclass) using the agent name. For the list of necessary and optional arguments the specific
        subclass will have to be checked.
        """
        load_classes_in_dir(Path(AGENTS_DIR))
        agent: Type[Agent]
        for agent in cls.__subclasses__():
            if agent.agent_name == agent_name:
                return agent(*agent_args, **agent_kwargs)
        raise NotImplementedError("Subclass {} of Agent is not implemented".format(agent_name))

    def execute_on_last(self, accessor: DataAccessor, portfolio_handler: PortfolioHandler) -> List[Order]:
        """ A helper method that executes the agent on the latest data only (given its run_interval)"""
        current_time_in_utc_seconds = to_utc_milliseconds(datetime.datetime.now())/1000
        start_time = to_datetime(int(current_time_in_utc_seconds
                                     - current_time_in_utc_seconds % self.run_interval.in_seconds())*1000)
        return self.execute(accessor, portfolio_handler, start_time)

    @abstractmethod
    def execute(self, accessor: DataAccessor, portfolio_handler: PortfolioHandler,
                start_time: Optional[datetime.datetime] = None) -> List[Order]:
        """
        Calculates actions to take from the start_time til the end_time (of the data in the accessor) and executes these
        in the provided portfolio_handler. This methods can be used both in a backtest scenario, but also in a realtime
        stepwise execution manner (with a tight start_time). Note tha the method both executes the orders for the
        duration on the provided portfolio, as well as return them in a list for later usage.
        """
        pass

    @abstractmethod
    def set_agent_properties(self, **configuration_dict) -> None:
        """ Sets the internal agent properties to the given configuration """
        pass

    @abstractmethod
    def fit(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
            courtage: Optional[float or Dict] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Trains the agent on the data available in the accessor (from the start_time onward) and returns the best found
        parameters (or result for all parameters if the flag-parameter is set accordingly) given the courtage settings
        provided.

        The method returns a list of tuples containing the agents configuration_dict and fitting value as follows:

        * The configuration_dict is the input to the agent init and the set_agent_properties method.
        * The fitting value is the metric used by the fitting method where higher is better.
        * The list should be sorted such that the highest value (best fitted model) is last.

        Moreover,the agent has, after the fitting, the configuration that gives the best fitting value.
        """
        pass

    @abstractmethod
    def visualize(self, accessor: DataAccessor, start_time: Optional[datetime.datetime] = None,
                  include_buy_sell_signals: bool = True, courtage_settings: Optional[float or Dict] = None,
                  plotlib: str = 'plotly', **plotlib_args) -> None:
        """
        Visualizes the agent behaviour over the given period (start_time til end of accessor data). I.e. visualizes
        important features and, if include_buy_and_sell_signals=True, buy and sell signals/portfolio value changes etc.
        to allow the viewer to understand what is going on.

        Additional arguments can be provided in plotlib_args depending on the plot library such as 'ax' for matplotlib.
        """
        pass

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

