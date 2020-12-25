import datetime
import pytz
from typing import Optional

""" Common utility functions related to datetime handling """


def to_utc_milliseconds(dt: datetime.datetime):
    """ Convert a datetime object to milliseconds utc aka UNIX time """
    epoch = datetime.datetime.utcfromtimestamp(0)
    if dt.tzinfo is not None:
        epoch = epoch.replace(tzinfo=dt.tzinfo)
    return int((dt - epoch).total_seconds() * 1000)


def to_tzinfo(time_zone: str or datetime.tzinfo) -> datetime.tzinfo:
    """ A helper method to return the time_zone of an asset"""
    if isinstance(time_zone, str):
        return pytz.timezone(time_zone)
    elif isinstance(time_zone, datetime.tzinfo):
        return time_zone
    else:
        raise NotImplementedError("Unable to handle type {}".format(type(time_zone)))

def add_time_zone_to_time(time: datetime.time, time_zone: datetime.datetime.tzinfo):
    """ Adds a timezone to a naive time object """
    return time_zone.localize(datetime.datetime.combine(datetime.datetime.today(), time)).timetz()


def to_datetime(time_var: str or datetime.datetime or datetime.date,
                time_zone: Optional[str or pytz.tzinfo.DstTzInfo or pytz.tzinfo.StaticTzInfo] = None):
    """ Converts common time and date formats to datetime by detecting how it is formatted """
    if isinstance(time_var, str):
        # Format 'YYMMDD HH:mm:ss'
        if len(time_var) == 17 and time_var[8] == " " and time_var[11] == ":" and time_var[14] == ":":
            ret_datetime = datetime.datetime.strptime(time_var, '%Y%m%d %H:%M:%S')
        # Format YYYY-MM-DD HH:mm:ss
        elif len(time_var) == 19 and time_var[4] == time_var[7] == "-" and time_var[13] == time_var[16] == ":":
            ret_datetime = datetime.datetime.strptime(time_var, '%Y-%m-%d %H:%M:%S')
        elif len(time_var) == 6:  # Format YYMMDD
            ret_datetime = datetime.datetime.strptime(time_var, '%Y%m%d')
        elif len(time_var) == 10 and time_var[4] == "-" and time_var[7] == "-":  # Format YYYY-MM-DD
            ret_datetime = datetime.datetime.strptime(time_var, '%Y-%m-%d')
        else:
            raise NotImplementedError(
                "Unknown conversion to date or datetime from string {}".format(time_var))
    elif isinstance(time_var, datetime.datetime):
        ret_datetime = time_var
    elif isinstance(time_var, datetime.date):
        ret_datetime = datetime.datetime.combine(time_var, datetime.datetime.min.time())
    elif isinstance(time_var, int):
        ret_datetime = pytz.timezone('UTC').localize(datetime.datetime.utcfromtimestamp(time_var/1000))

    else:
        raise NotImplementedError(
            "Currently cannot convert the class {} to datetime, please implement".format(type(time_var)))
    return set_time_zone(ret_datetime, time_zone)


def set_time_zone(dt: datetime.datetime, time_zone: Optional[str or pytz.tzinfo.DstTzInfo or pytz.tzinfo.StaticTzInfo]):
    """ Adds a timezone to a naive datetime object, or, if it is  aware, converts it to the given time zone """
    if time_zone is None:
        return dt
    elif isinstance(time_zone, str):
        time_zone = pytz.timezone(time_zone)
    elif isinstance(time_zone, (pytz.tzinfo.DstTzInfo, pytz.tzinfo.StaticTzInfo)):
        pass
    else:
        raise NotImplementedError('Currently unable to handle time_zone type {}'.format(type(time_zone)))
    if dt.tzinfo is None:
        return time_zone.localize(dt)
    else:
        return dt.astimezone(time_zone)

