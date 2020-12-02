import logging
import ib_insync as ibi
from ib_insync import Contract
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import datetime
import re
from typing import Optional

from dateutil.relativedelta import relativedelta
from atapy.data_cleaning import get_liquid_hours, add_start_and_end_time, filter_out_non_traded_hours, \
    fill_missing_intervals, aggregate_intraday_data
from atapy.data_accessor import RAW_DATA_TYPES, FEATURE_DATA_TYPES
from atapy.utils import to_datetime
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.market_data_connection import MarketDataConnection
from atapy.broker_classes.interactive_brokers_connection import InteractiveBrokersConnection, RECONNECT_WAIT_TIME, \
    DATA_REQUEST_TIMEOUT, OPTIMAL_BATCH_SIZES, BARSIZE_SETTING_CONVERSION_DICT

logger = logging.getLogger()


class InteractiveBrokersDataConnection(MarketDataConnection, InteractiveBrokersConnection):
    """
    A asset data connection class to help with downloading and updating information from the interactive brokers
    API (TWS API) using the ib_insync connector. This class mainly handles managing a specific file structure and
    keeping it up to date by being run in desired intervals.
    """

    connection_name = 'InteractiveBrokers'

    def collect_asset_information(self, asset_type: str = 'stk', rediscover_exchanges=False):

        if asset_type != 'stk':
            raise NotImplementedError("Collection of asset information currently only available for stocks (stk)")

        """
        Step 1, collect all the known exchanges.
        This is trickier than it sounds:
        * Not possible through the online listing, multiple important exchanges missing there such as CPH
        * Not possible through scanning. Multiple location exchanges missing such as IBIS2 and BMW2
        * Remaining possibility is to scan through all 3 letter/number combinations and see what exchanges pop up
        """
        known_exchanges = self.ib2mic.keys()

        # Step 2, Collect all contract on the exchanges
        logger.debug('Collecting stocks per exchange')
        # https://www.interactivebrokers.com/en/index.php?f=2222&exch=amex&showcategories=STK#productbuffer
        contracts_listing_page_href_template = "https://www.interactivebrokers.com/en/index.php?f=2222&exch={exchange}&showcategories={asset_type}#productbuffer"
        all_contracts_list = []
        base_page = 'https://www.interactivebrokers.com'
        for exchange in tqdm(known_exchanges, 'Collecting stocks by exchange'):
            contracts_listing_page_href = contracts_listing_page_href_template.format(exchange=exchange,
                                                                                      asset_type=asset_type)
            on_final_pagination_page = False
            table_values = []
            contract_ids = []
            while not on_final_pagination_page:
                contracts_page_soup = BeautifulSoup(requests.get(contracts_listing_page_href).text, features="html5lib")
                # Note that first page href is disabled (on first page). Hence 2 pages means only 1 page is available.
                # And in such a case there are always 3 pages.
                # On only one page only 2 pages are available (initial and disabled)
                if (contracts_page_soup.find('ul', class_="pagination") is None
                        or len(contracts_page_soup.find('ul', class_="pagination").find_all('li')) <= 2
                        or contracts_page_soup.find('ul', class_="pagination").find_all('li')[-1]['class'] == [
                            'disabled']):
                    on_final_pagination_page = True
                else:
                    contracts_listing_page_href = base_page + \
                                                  contracts_page_soup.find('ul', class_="pagination").find_all('li')[
                                                      -1].find('a', href=True)['href']
                table_rows = contracts_page_soup.find_all('table', class_="table table-striped table-bordered")[
                    -1].find_all('tr')
                contract_ids.extend([row.find('a', class_='linkexternal')['href'].split('conid=')[1].split("',")[0]
                                     for row in table_rows[1:]])
                table_columns = [td.contents[0].strip() for td in table_rows[0].find_all('th')]
                table_values.extend([[td.text.replace("\n", "") for td in tr.find_all('td')] for tr in table_rows[1:]])
            contract_df = pd.DataFrame(table_values, columns=table_columns)
            contract_df['contract_id'] = contract_ids
            contract_df['ib_exchange'] = exchange
            contract_df['mic_code'] = self.ib2mic[exchange]
            all_contracts_list.append(contract_df)
        all_contracts_df = pd.concat(all_contracts_list)
        logger.info("{} nr of contracts found with {} unique ids".format(
            all_contracts_df.shape[0], len(all_contracts_df.contract_id.unique())))
        all_contracts_df = all_contracts_df.rename(
            columns={'IB Symbol': 'ib_symbol', 'Product Description': 'full_name', 'Symbol': 'symbol',
                     'Currency': 'currency'})
        all_contracts_df['contract_id'] = all_contracts_df['contract_id'].astype(int)
        # Step 3, Collect detailed info for each contract
        contract_details_list = []
        for contract_id in tqdm(set(all_contracts_df.contract_id), 'Downloading detailed information'):
            self.ensure_connection()
            contract_details = self.ib.reqContractDetails(Contract(conId=contract_id))
            if len(contract_details) > 0:
                contract_details_list.append(contract_details[0].__dict__)
            else:
                logger.warning("Contract {} returned no data ({})".format(
                    contract_id,
                    all_contracts_df['full_name'][all_contracts_df.contract_id == contract_id].iloc[0]))

        contract_details_df = pd.DataFrame(contract_details_list)
        contract_details_df = pd.concat(
            [pd.DataFrame([val.__dict__ for val in contract_details_df.contract.to_list()]), contract_details_df],
            axis=1)

        # Only keep releveant columns and rename
        columns_to_keep = ['secType', 'conId', 'symbol', 'validExchanges', 'currency', 'industry', 'category',
                           'subcategory', 'liquidHours']
        contract_details_df = contract_details_df[columns_to_keep]
        contract_details_df = contract_details_df.rename(columns={
            'secType': 'security_type', 'conId': 'contract_id', 'symbol': 'ib_symbol',
            'validExchanges': 'valid_exchanges',
            'barCount': 'nr_of_trades', 'liquidHours': 'liquid_hours'})

        # Step 3, Extract the actual traded exchanges and merge in the real (local) contract names.
        contract_details_df['ib_exchange'] = contract_details_df.valid_exchanges.str.split(",")
        all_contract_details_df = contract_details_df.explode('ib_exchange')
        all_contract_details_df = all_contract_details_df[all_contract_details_df.ib_exchange.isin(known_exchanges)]
        all_contract_details_df = all_contract_details_df.drop(columns='valid_exchanges')

        all_contract_details_df = all_contract_details_df.merge(all_contracts_df,
                                                                on=['ib_symbol', 'ib_exchange', 'currency',
                                                                    'contract_id'])

        # Step 4, calculate liquid hours
        def calc_liquid_hours(liquid_hours_str):
            if liquid_hours_str is not None:
                _start_times = [a[1:5] for a in re.compile(':....-').findall(liquid_hours_str)]
                _end_times = [a[1:5] for a in re.compile(':....;|:....$').findall(liquid_hours_str)]
                _start_times = sorted(list(set(_start_times)))
                _end_times = sorted(list(set(_end_times)))

                final_str = ("{}-{}".format(_start_times[0], _end_times[0]) if len(_start_times) == 1
                             else "{}-{},{}-{}".format(_start_times[0], _end_times[0], _start_times[1], _end_times[1]))
                return final_str
            else:
                return None

        all_contract_details_df['liquid_hours'] = all_contract_details_df.liquid_hours.apply(calc_liquid_hours)

        # Step 5, save the data
        self.data_accessor.write_custom_table('ib_symbol_translation', all_contract_details_df[
            ['security_type', 'contract_id', 'ib_symbol', 'currency', 'ib_exchange', 'symbol', 'mic_code']])
        all_contract_details_df = all_contract_details_df.rename(columns={'mic_code': 'exchange'})
        self.data_accessor.write_asset_information(all_contract_details_df[
                                                       ['security_type', 'exchange', 'symbol', 'currency', 'full_name',
                                                        'industry', 'category', 'subcategory', 'liquid_hours']])
        return all_contract_details_df

    def collect_historical_asset_data(self, asset: Asset, redownload_data=False, include_minute_data=False,
                                      update_feature_data=False):
        def download_historical_data_safe(_contract, end_date, duration_str, bar_size_setting):
            logger.debug("Downloading data for {}: end date: {}, duration: {}, bar size {}".format(
                asset, end_date, duration_str, bar_size_setting))
            new_data = []
            error_code = None
            try_download = True
            while try_download:
                try:
                    self.ensure_connection()
                    new_data = self.ib.reqHistoricalData(contract=_contract,
                                                         endDateTime=end_date,
                                                         durationStr=duration_str,
                                                         barSizeSetting=bar_size_setting,
                                                         whatToShow='TRADES',
                                                         useRTH=False,
                                                         timeout=DATA_REQUEST_TIMEOUT)
                    logger.debug("{} records downloaded".format(len(new_data)))
                except ConnectionError as e:
                    logger.error("Connection broken, retrying")
                    time.sleep(RECONNECT_WAIT_TIME)
                    continue
                except TimeoutError as e:
                    logger.error("Timeout error, retrying")
                    time.sleep(RECONNECT_WAIT_TIME)
                    continue
                else:
                    # Handle logged errors
                    req_id = new_data.reqId
                    if req_id in self.error_events.keys():
                        # Known error events that can occur:
                        # 162 - No historical data - ignore
                        # 165 - Server connection unsuccessful
                        # 185 - No security definition - wait, reconnect, ignore
                        # 200 - No security definition bad request - raise exception, no data will be available
                        event_data = self.error_events[req_id]
                        if event_data[0] in [162]:
                            error_code = 162
                        elif event_data[0] in [165, 185]:
                            self.ib.disconnect()
                            time.sleep(RECONNECT_WAIT_TIME)
                            continue
                        elif event_data[0] in [200]:
                            raise RuntimeError('No data available for contract {}:{}'.format(_contract, event_data))
                        else:
                            raise Exception("Unknown error on downloading historical data {}".format(event_data))
                try_download = False
            return new_data, error_code

        def get_start_date(_interval: Interval, _redownload_data: bool):
            existing_df = self.data_accessor.get_raw_asset_data(asset, _interval)
            if not _redownload_data and existing_df.shape[0] != 0:
                return existing_df.date.max().date()
            else:
                existing_daily_df = self.data_accessor.get_raw_asset_data(asset, Interval.daily)
                if existing_daily_df.shape[0] > 0 and _interval != Interval.daily:
                    return existing_daily_df.date.min().date()
                else:
                    start_time = self.ib.reqHeadTimeStamp(contract, 'TRADES', False)
                    if isinstance(start_time, datetime.datetime):
                        return start_time.date()
                    else:
                        return to_datetime('1990-01-01').date()

        def calc_duration(start_date, end_date):
            diff_time = relativedelta(end_date, start_date)
            if diff_time.years > 0:
                return "{} Y".format(
                    diff_time.years + 1 if diff_time.months > 0 or diff_time.days > 0 else diff_time.years)
            else:
                return "{} D".format((end_date - start_date).days + 1)

        def check_for_new_data(_all_trades_per_day, existing_trades_per_day, start_date, end_date):
            if _all_trades_per_day is not None:
                return not _all_trades_per_day[
                    (_all_trades_per_day.index >= start_date) & (_all_trades_per_day.index <= end_date)].equals(
                    existing_trades_per_day[
                        (existing_trades_per_day.index >= start_date) & (existing_trades_per_day.index <= end_date)])
            return True

        def download_historical_data_batchwise(_contract: ibi.Contract, start_date: datetime.date,
                                               interval_size: Interval,
                                               batch_size: int,
                                               _all_trades_per_day: Optional[pd.Series],
                                               existing_trades_per_day: Optional[pd.Series]) -> pd.DataFrame:
            # Download in reverse, starting with the current date back to the start date,
            # then go backwards in time until a 162 error is given, or the startdate is hit
            data_list = []
            earliest_downloaded_date = datetime.date.today()
            bar_size_setting = BARSIZE_SETTING_CONVERSION_DICT[interval_size]
            end_loop = False
            while not end_loop:
                if earliest_downloaded_date < start_date:
                    break
                if (earliest_downloaded_date - start_date).days > batch_size:
                    batch_start_date = earliest_downloaded_date - datetime.timedelta(days=batch_size)
                else:
                    batch_start_date = start_date
                    end_loop = True
                if check_for_new_data(_all_trades_per_day, existing_trades_per_day,
                                      batch_start_date,
                                      earliest_downloaded_date):
                    data, error = download_historical_data_safe(_contract=_contract,
                                                                end_date=earliest_downloaded_date,
                                                                duration_str=calc_duration(
                                                                    batch_start_date, earliest_downloaded_date),
                                                                bar_size_setting=bar_size_setting)
                    if error == 162:  # No more data available
                        break
                    earliest_downloaded_date = to_datetime(data[0].date).date() - datetime.timedelta(days=1)
                    data_list.extend(data)
                else:
                    earliest_downloaded_date = earliest_downloaded_date - datetime.timedelta(days=batch_size)

            if len(data_list) == 0:
                res_df = self.data_accessor.get_empty_raw_asset_data()
            else:
                res_df = pd.DataFrame([val.__dict__ for val in data_list]).rename(columns={'barCount': 'nr_of_trades'})
                res_df['date'] = pd.to_datetime(res_df['date'])
            logger.info("{} nr of {} entries downloaded for {}".format(res_df.shape[0], interval_size, asset))
            return res_df.drop_duplicates().sort_values('date')

        def calc_trades_per_day(_asset, _interval):
            df = self.data_accessor.get_raw_asset_data(_asset, _interval)
            df['dt'] = df.date.dt.date
            return df.groupby('dt').nr_of_trades.sum()

        contract = self._get_ib_contract(asset=asset)
        # Daily data - First try to take it all at once, if that fails, split by year
        daily_start_date = get_start_date(Interval.daily, redownload_data)
        daily_df = download_historical_data_batchwise(contract, daily_start_date, Interval.daily,
                                                      (datetime.date.today()-daily_start_date).days, None, None)
        if daily_df.shape[0] == 0:
            # This can happen if the stock is very new, or has been made available on the exchange recently
            daily_df = download_historical_data_batchwise(contract, daily_start_date, Interval.daily, 5, None, None)
        self.clean_data_and_save(asset, daily_df, interval=Interval.daily, save_feature_data=update_feature_data)
        all_trades_per_day = calc_trades_per_day(asset, Interval.daily)
        # Intraday data
        for interval in Interval:
            if interval == Interval.daily:
                continue
            if interval == Interval.minute and not include_minute_data:
                continue
            else:
                data_df = download_historical_data_batchwise(
                    _contract=contract, start_date=get_start_date(interval, redownload_data), interval_size=interval,
                    batch_size=OPTIMAL_BATCH_SIZES[interval],
                    _all_trades_per_day=all_trades_per_day,
                    existing_trades_per_day=calc_trades_per_day(asset, interval))
                self.clean_data_and_save(asset, data_df, interval, update_feature_data)

    def clean_data_and_save(self, asset: Asset, data_df: pd.DataFrame, interval: Interval,
                            save_feature_data: bool = True):
        """ Saves the raw data for the asset, as well as performs data cleaning and saves the feature data
        for five min interval data (that aggregates to other intervals"""
        self.data_accessor.write_raw_asset_data(asset, interval, data_df.astype(RAW_DATA_TYPES))
        if save_feature_data and interval == Interval.five_min:
            data_df = add_start_and_end_time(data_df, interval)
            asset_info = self.data_accessor.get_asset_information(asset=asset)
            liquid_hours = asset_info.liquid_hours.iloc[0]
            start_daily_trade_time, end_daily_trade_time = get_liquid_hours(liquid_hours)
            data_df = filter_out_non_traded_hours(data_df=data_df, start_trade_time=start_daily_trade_time,
                                                  end_trade_time=end_daily_trade_time)
            traded_dates = data_df.date.drop_duplicates()
            data_df = fill_missing_intervals(data_df, interval, traded_dates, start_daily_trade_time,
                                             end_daily_trade_time)
            dfs = aggregate_intraday_data(data_df, interval)
            for agg_interval, interval_df in dfs.items():
                self.data_accessor.write_feature_asset_data(asset, agg_interval, interval_df.astype(FEATURE_DATA_TYPES))
