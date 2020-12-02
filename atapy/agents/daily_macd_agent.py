from typing import List
import pandas as pd
import datetime

from atapy.data_accessor import DataAccessor
from atapy.order import Order


class daily_macd_agent:
    """
    An initial agent that only trades on the macd values (daily)
    Zero crosses strategy: If the macd goes from above to below 0, sell, if it goes from below to above, buy
    """
    interval_size = 'daily'
    agent_name = 'macd'

    def __init__(self, exchange, asset_symbol, asset_type, macd_strategy = 'zero_line_crossover'):
        # Here we will also need more trade info, such as cost per trade etc
        self.exchange = exchange
        self.asset_symbol = asset_symbol
        self.asset_type = asset_type

        self.macd_strategy = macd_strategy

    def step(self, accessor: DataAccessor, current_time: datetime.datetime = None) -> List[Order]:
        daily_df = accessor.get_feature_asset_data(self.exchange, self.asset_symbol, self.asset_type)
        # Filter away data after the given time
        if current_time is not None:
            daily_df = daily_df[daily_df.end_time <= current_time]
        # Calculate if there has been a change in macd
        if self.macd_strategy == 'zero_line_crossover':
            if daily_df.macd_difference.iloc[-2] > 0 and daily_df.macd_difference.iloc[-1] < 0:
                # Crossing from positive to negative Sell order
                return [Order(order_type='sell')]
            elif daily_df.macd_difference.iloc[-2] < 0 and daily_df.macd_difference.iloc[-1] > 0:
                # Crossing from negative to positive: Buy order
                return [Order(order_type='buy')]
            else:
                return []
        else:
            raise NotImplementedError

    def backtest(self, accessor: DataAccessor, start_time: datetime.datetime,
                 end_time: datetime.datetime) -> pd.Series(Order):
        pass

    def train(self, accessor: DataAccessor):
        pass
