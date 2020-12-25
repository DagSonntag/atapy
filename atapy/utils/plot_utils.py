import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict

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