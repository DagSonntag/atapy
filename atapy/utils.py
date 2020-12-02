import tensorflow as tf
import datetime
from typing import TYPE_CHECKING, Callable, Dict
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
import pandas as pd

if TYPE_CHECKING:
    from atapy.data_accessor import DataAccessor

""" Move the tensorflow utils methods to another file"""
BATCH_SIZE = 64

def features_only(x, y):
    return x

def targets_only(x, y):
    return y

def dataset_creator(accessor: 'DataAccessor', single_instance_generator_function: Callable,
                    batch_size: int = BATCH_SIZE, prefetch: int = 2, randomize: bool = True,
                    **generator_args) -> tf.data.Dataset:
    generator_creator = single_instance_generator_function(accessor, randomize=randomize, **generator_args)
    x, y = next(generator_creator())
    x_shapes = {}
    x_types = {}
    for key, tensor in x.items():
        x_shapes.update({key: tensor.shape})
        x_types.update({key: tensor.dtype})
    y_shapes = {}
    y_types = {}
    for key, tensor in y.items():
        y_shapes.update({key: tensor.shape})
        y_types.update({key: tensor.dtype})
    dataset = tf.data.Dataset.from_generator(generator_creator,
                                             output_types=(x_types, y_types),
                                             output_shapes=(x_shapes, y_shapes))
    return dataset.batch(batch_size).prefetch(prefetch)


def to_datetime(time_var: str or datetime.datetime or datetime.date):
    """ Converts common time and date formats to datetime by detecting how it is formatted """
    if isinstance(time_var, str):
        # Format 'YYMMDD HH:mm:ss'
        if len(time_var) == 17 and time_var[8] == " " and time_var[11] == ":" and time_var[14] == ":":
            return datetime.datetime.strptime(time_var, '%Y%m%d %H:%M:%S')
        # Format YYYY-MM-DD HH:mm:ss
        if len(time_var) == 19 and time_var[4] == time_var[7] == "-" and time_var[13] == time_var[16] == ":":
            return datetime.datetime.strptime(time_var, '%Y-%m-%d %H:%M:%S')
        if len(time_var) == 6:  # Format YYMMDD
            return datetime.datetime.strptime(time_var, '%Y%m%d')
        elif len(time_var) == 10 and time_var[4] == "-" and time_var[7] == "-":  # Format YYYY-MM-DD
            return datetime.datetime.strptime(time_var, '%Y-%m-%d')
        else:
            raise NotImplementedError(
                "Unknown conversion to date or datetime from string {}".format(time_var))
    elif isinstance(time_var, datetime.datetime):
        return time_var
    elif isinstance(time_var, datetime.date):
        return datetime.datetime.combine(time_var, datetime.datetime.min.time())
    elif isinstance(time_var, int):
        if time_var > 1000000000:
            return datetime.datetime.utcfromtimestamp(time_var/1000)
        else:
            return datetime.datetime.utcfromtimestamp(time_var)
    else:
        raise NotImplementedError(
            "Currently cannot convert the class {} to datetime, please implement".format(type(time_var)))


def load_classes_in_dir(dir_path: Path) -> None:
    """ Loads all python classes in a given directory (in filenames ending with .py) """
    for py_file_name in [f.name for f in dir_path.iterdir()
                         if f.suffix == '.py' and f.name != '__init__.py']:
        # For the class_path we should only include the part of the path from classes and forward
        parts = dir_path.parts
        index = max(loc for loc, val in enumerate(parts) if val == 'atapy')
        mod = __import__('.'.join(parts[index:]), fromlist=[py_file_name])
        classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for sub_cls in classes:
            setattr(sys.modules[__name__], sub_cls.__name__, sub_cls)


def to_list(var):
    """ Create a list of an item if it wasn't already"""
    if isinstance(var, list):
        return var
    else:
        return [var]


def feature_visualization(candlestick_data_df: pd.DataFrame = None,
                          single_value_data_dict: Dict[str, pd.Series] = None,
                          buy_signal_series: pd.Series = None,
                          sell_signal_series: pd.Series = None,
                          plotlib: str='plotly'):
    """ A helper method that splits the single_value_dict into macd and rsi plots and calls the correct library
    visualization method"""
    macd_data_dict = {col_name: single_value_data_dict[col_name]
                      for col_name in single_value_data_dict.keys() if 'macd' in col_name}
    rsi_data_dict = {col_name: single_value_data_dict[col_name]
                     for col_name in single_value_data_dict.keys() if 'rsi' in col_name}
    single_value_data_dict = {col_name: single_value_data_dict[col_name]
                              for col_name in single_value_data_dict.keys() if
                              'rsi' not in col_name and 'macd' not in col_name}
    plotly_visualization(candlestick_data_df, single_value_data_dict, buy_signal_series, sell_signal_series,
                         macd_data_dict, rsi_data_dict)


def plotly_visualization(candlestick_data_df: pd.DataFrame = None,
                         single_value_data_dict: Dict[str, pd.Series] = None,
                         buy_signal_series: pd.Series = None,
                         sell_signal_series: pd.Series = None,
                         macd_data_dict: Dict[str, pd.Series] = None,
                         rsi_data_dict: Dict[str, pd.Series] = None
                         ):
    """ A helper method to plot the data in a plotly fig """

    def add_buy_and_sell_signals(fig, buy_series, sell_series, value_series, row):
        if buy_series is not None:
            fig.add_trace(
                go.Scatter(name='buy', mode='markers', x=buy_series.index,
                           y=value_series.loc[buy_series.index].values,
                           marker={"size": 12, "color": "green"}, marker_symbol='triangle-up'), row=row, col=1)
        if sell_series is not None:
            fig.add_trace(
                go.Scatter(name='sell', mode='markers', x=sell_series.index,
                           y=value_series.loc[sell_series.index].values,
                           marker={"size": 12, "color": "red"}, marker_symbol='triangle-down'), row=row, col=1)

    n_rows = 2
    titles = ["Price plot", "", ]
    specs = [[{"rowspan": 2}], [{}]]
    macd_row = rsi_row = 0
    if macd_data_dict is not None and len(macd_data_dict) > 0:
        n_rows += 1
        macd_row = n_rows
        titles.append('MACD')
        specs.append([{}])
    if rsi_data_dict is not None and len(rsi_data_dict) > 0:
        n_rows += 1
        rsi_row = n_rows
        titles.append('RSI')
        specs.append([{}])

    fig = make_subplots(
        rows=n_rows, cols=1, subplot_titles=titles,
        specs=specs, shared_yaxes=False,
        shared_xaxes=True, vertical_spacing=0.1
    )
    # General layout settings
    fig.update_layout(height=1400, width=1400)
    fig.update_layout(legend_orientation="h", showlegend=True)
    fig.update_xaxes(rangeslider=dict(visible=True), rangeslider_thickness=0.03)
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                x=0, y=1,
                buttons=list([
                    dict(count=1, label="1day", step="day", stepmode="backward"),
                    dict(count=7, label="1week", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")]))))

    # Start with candlestick data
    fig.add_trace(
        go.Candlestick(name='candlestick', x=candlestick_data_df.index, open=candlestick_data_df.open,
                       close=candlestick_data_df.close, low=candlestick_data_df.low,
                       high=candlestick_data_df.high), row=1, col=1)
    # Price based data
    if single_value_data_dict is not None:
        for key, series in single_value_data_dict.items():
            fig.add_trace(go.Scatter(name=key, x=series.index, y=series.values), row=1, col=1)
    add_buy_and_sell_signals(fig, buy_signal_series, sell_signal_series, candlestick_data_df.close, 1)

    # Layout for top plot
    fig.update_layout(yaxis=dict(anchor='free'))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), rangeslider_thickness=0.03))
    # Macd data
    if macd_data_dict is not None and len(macd_data_dict) > 0:
        fig.add_trace(go.Scatter(name='macd', x=macd_data_dict['macd'].index, y=macd_data_dict['macd'].values),
                      row=macd_row, col=1)
        fig.add_trace(go.Scatter(name='macd_signal', x=macd_data_dict['macd_signal'].index,
                                 y=macd_data_dict['macd_signal'].values),
                      row=macd_row, col=1)
        fig.add_trace(go.Bar(name='macd_histogram', x=macd_data_dict['macd_histogram'].index,
                             y=macd_data_dict['macd_histogram'].values),
                      row=macd_row, col=1)
        add_buy_and_sell_signals(fig, buy_signal_series, sell_signal_series, macd_data_dict['macd'], macd_row)

    # RSI data
    if rsi_data_dict is not None and len(rsi_data_dict) > 0:
        for key, series in rsi_data_dict.items():
            fig.add_trace(go.Scatter(name=key, x=series.index, y=series.values), row=rsi_row, col=1)
        zero_series = candlestick_data_df.close.copy()
        zero_series.iloc[:] = 0.5
        add_buy_and_sell_signals(fig, buy_signal_series, sell_signal_series, zero_series, rsi_row)
    # fig.update_layout(legend_orientation="h", xaxis_rangeslider_visible=True, overwrite=True)
    for i in range(n_rows):
        xaxis_name = 'xaxis' if i == 0 else f'xaxis{i + 1}'
        getattr(fig.layout, xaxis_name).showticklabels = True
    fig.show()


def check_types(data_df: pd.DataFrame, correct_types: Dict):
    """ Checks that the given data_df contains the right columns and data types. If not it tries to convert them """
    if set(data_df.columns) != set(correct_types.keys()):
        raise ValueError("Data provided with incorrect columns. Expecting {}".format(
            correct_types.keys()))
    if data_df.dtypes.to_dict() != correct_types:
        data_df = data_df.astype(correct_types)
    return data_df

def to_utc_milliseconds(dt: datetime.datetime):
    """ Convert a datatime object to miliseconds utc aka UNIX time """
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)

def round_up_til_full_minute(dt: pd.Timestamp):
    """ Returns the timestamp of the next even minute (i.e. the timestamp where seconds = 0) """
    if dt.second == 0:
        return dt
    else:
        return dt + datetime.timedelta(minutes=1, seconds=-dt.second)
