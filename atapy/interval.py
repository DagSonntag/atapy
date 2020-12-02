from enum import Enum
import pandas as pd


class Interval(Enum):
    """ An enum class for the different supported intervals """
    daily = 24 * 60 * 60
    hourly = 60 * 60
    five_min = 5 * 60
    minute = 60

    def in_minutes(self):
        """ Returns the interval in minutes """
        return self.value / 60.0

    def in_seconds(self):
        """ Returns the interval in seconds """
        return self.value

    def datetime_identifiers(self, time_series: pd.Series):
        """ Returns unique identifiers for the different intervals in a pandas series of timestamps.
        Useful in for example groupby expressions """
        if self.name == 'daily':
            return time_series.dt.date
        elif self.name == 'hourly':
            return [time_series.dt.date, time_series.dt.hour]
        elif self.name == 'five_min':
            return [time_series.dt.date, time_series.dt.hour, time_series.dt.minute.floordiv(5)]
        elif self.name == 'minute':
            return [time_series.dt.date, time_series.dt.hour, time_series.dt.minute.floordiv(1)]

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

