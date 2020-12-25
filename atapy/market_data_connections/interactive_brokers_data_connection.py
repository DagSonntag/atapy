import logging
from ib_insync import Contract
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import datetime
import re
from typing import Tuple
import math

from atapy.data_accessor import RAW_DATA_COLUMN_TYPES, ASSET_INFO_COLUMN_TYPES
from atapy.utils.datetime_utils import to_datetime
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

    def collect_asset_information(self, asset_type: str = 'stk') -> None:
        """
        Collects asset information from Interactive brokers using the reqContractDetails API-function. However, the
        stocks themselves first needs to be collected by crawling the interactive brokers homepage where they list
        their data.
        Note the following:

        * Only the exchanges in the EXCHANGES_TRANSLATION_TABLE will be crawled and collected for.
        * Currently only stocks are supported. For additional instruments additional work may be needed.
        * | The method may throw several error log messages about stocks, missing.
          | This is due to inconsistencies in the data provided by interactive brokers (also shown in in TWS interface)

        """
        if asset_type != 'stk':
            raise NotImplementedError("Collection of asset information currently only available for stocks (stk)")

        # Step 1, For all known_exchanges, crawl the IB homepage and collect the assets available there
        known_exchanges = self.ib2mic.keys()
        logger.debug('Collecting stocks per exchange')
        # https://www.interactivebrokers.com/en/index.php?f=2222&exch=amex&showcategories=STK#productbuffer
        base_page = 'https://www.interactivebrokers.com'
        contracts_listing_page_href_template = (
            base_page + "/en/index.php?f=2222&exch={exchange}&showcategories={asset_type}#productbuffer")
        all_contracts_list = []
        for exchange in tqdm(known_exchanges, 'Collecting stocks by exchange'):
            contracts_listing_page_href = contracts_listing_page_href_template.format(exchange=exchange,
                                                                                      asset_type=asset_type)
            on_final_pagination_page = False
            table_values = []
            contract_ids = []
            table_columns = ['IB Symbol', 'Product Description', 'Symbol', 'Currency']
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
                table_rows = contracts_page_soup.find_all(
                    'table', class_="table table-striped table-bordered")[-1].find_all('tr')
                contract_ids.extend([row.find('a', class_='linkexternal')['href'].split('conid=')[1].split("',")[0]
                                     for row in table_rows[1:]])
                table_values.extend([[td.text.replace("\n", "") for td in tr.find_all('td')] for tr in table_rows[1:]])
            contract_df = pd.DataFrame(table_values, columns=table_columns)
            contract_df['contract_id'] = contract_ids
            contract_df['ib_exchange'] = exchange
            contract_df['mic_code'] = self.ib2mic[exchange]
            all_contracts_list.append(contract_df)
        all_contracts_df = pd.concat(all_contracts_list)
        logger.debug("{} nr of contracts found with {} unique ids".format(
            all_contracts_df.shape[0], len(all_contracts_df.contract_id.unique())))
        all_contracts_df = all_contracts_df.rename(
            columns={'IB Symbol': 'ib_symbol', 'Product Description': 'full_name', 'Symbol': 'symbol',
                     'Currency': 'currency'})
        all_contracts_df['contract_id'] = all_contracts_df['contract_id'].astype(int)

        # Step 2, Collect detailed info for each contract
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
                           'subcategory', 'liquidHours', 'timeZoneId']
        contract_details_df = contract_details_df[columns_to_keep]
        contract_details_df = contract_details_df.rename(columns={
            'secType': 'asset_type', 'conId': 'contract_id', 'symbol': 'ib_symbol',
            'validExchanges': 'valid_exchanges', 'barCount': 'nr_of_trades', 'liquidHours': 'liquid_hours',
            'timeZoneId': 'time_zone'})

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
        all_contract_details_df = all_contract_details_df.rename(columns={'mic_code': 'exchange'})
        self.data_accessor.write_custom_table('ib_symbol_translation', all_contract_details_df[
            ['asset_type', 'contract_id', 'ib_symbol', 'currency', 'ib_exchange', 'symbol', 'exchange']])
        self.data_accessor.write_asset_information(all_contract_details_df[ASSET_INFO_COLUMN_TYPES])

    def collect_historical_data(self, asset: Asset, redownload_data=False, update_feature_data=False,
                                intervals: Tuple[Interval] = (Interval.daily, Interval.hourly, Interval.five_min, )):
        contract = self._get_ib_contract(asset=asset)
        # All requests are handled in the local timezone
        local_time_zone = self.data_accessor.get_asset_local_tzinfo(asset)

        def get_start_time(interval):
            existing_df = self.data_accessor.get_raw_asset_data(asset, interval)
            if not redownload_data and existing_df.shape[0] != 0:
                return existing_df['date'].iloc[-1]
            else:
                start_time = self.ib.reqHeadTimeStamp(contract, whatToShow='TRADES', useRTH=False, formatDate=2)
                if isinstance(start_time, datetime.datetime):
                    return start_time.astimezone(local_time_zone)
                else:
                    return to_datetime('1990-01-01', time_zone=local_time_zone)

        def calc_duration(start_time: datetime.datetime, end_time: datetime.datetime, interval=None):
            diff_in_seconds = (end_time - start_time).total_seconds()
            if diff_in_seconds > 60 * 60 * 24 * 365:  # Year
                return "{} Y".format(math.ceil(diff_in_seconds / 60 / 60 / 24 / 365))
            elif diff_in_seconds > 60 * 60 * 24 or interval == Interval.daily:  # Day
                return "{} D".format(math.ceil(diff_in_seconds / 60 / 60 / 24))
            else:  # Seconds
                return "{} S".format(math.ceil(diff_in_seconds))

        def download_historical_data_safe(end_time, duration_str, bar_size_setting):
            logger.debug("Downloading data for {}: end date: {}, duration: {}, bar size {}".format(
                asset, end_time, duration_str, bar_size_setting))
            error_code = None
            # Loop over data_download attempts that either will end successfully or with an exception
            while True:
                try:
                    self.ensure_connection()
                    new_data = self.ib.reqHistoricalData(contract=contract,
                                                         endDateTime=end_time,
                                                         durationStr=duration_str,
                                                         barSizeSetting=bar_size_setting,
                                                         whatToShow='TRADES',
                                                         useRTH=False,
                                                         formatDate=2,
                                                         timeout=DATA_REQUEST_TIMEOUT)
                    logger.debug("{} records downloaded".format(len(new_data)))
                except (ConnectionError, TimeoutError) as e:
                    logger.error("{}: {}, retrying".format(self, type(e)))
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
                        # 366 - 'No historical data query found for ticker id:101'
                        event_data = self.error_events[req_id]
                        if event_data[0] in [162, 366]:
                            error_code = event_data[0]
                        elif event_data[0] in [165, 185]:
                            logger.error("{}: Error due to missing data or on serverside".format(self))
                            self.ib.disconnect()
                            time.sleep(RECONNECT_WAIT_TIME)
                            continue
                        elif event_data[0] in [200]:
                            raise RuntimeError('{}: No data available for contract {}:{}'.format(
                                self, contract, event_data))
                        else:
                            raise Exception("{}: Unknown error on downloading historical data {}".format(
                                self, event_data))
                return new_data, error_code

        def download_historical_data_batchwise(start_time: datetime.datetime, batch_size: int, interval: Interval
                                               ) -> pd.DataFrame:
            # Download in reverse, starting with the current date back to the start date,
            # then go backwards in time until a 162 error is given, or the startdate is hit

            data_list = []
            bar_size_setting = BARSIZE_SETTING_CONVERSION_DICT[interval]
            earliest_downloaded_time = datetime.datetime.now(local_time_zone)
            logger.debug("Downloading data for {} from {} to {}".format(asset, start_time, earliest_downloaded_time))
            end_loop = False
            while not end_loop:
                if earliest_downloaded_time < start_time:
                    break
                if (earliest_downloaded_time - start_time).total_seconds() / 60 / 60 / 24 > batch_size:
                    batch_start_time = earliest_downloaded_time - datetime.timedelta(days=batch_size)
                else:
                    batch_start_time = start_time
                    end_loop = True
                data, error = download_historical_data_safe(end_time=earliest_downloaded_time,
                                                            duration_str=calc_duration(
                                                                batch_start_time, earliest_downloaded_time, interval),
                                                            bar_size_setting=bar_size_setting)
                if error == 162:  # No more data available
                    break
                elif len(data) > 0:
                    earliest_downloaded_time = to_datetime(data[0].date, time_zone=local_time_zone)
                else:
                    earliest_downloaded_time = batch_start_time
                data_list.extend(data)

            if len(data_list) == 0:
                res_df = self.data_accessor.get_empty_raw_asset_data()
            else:
                res_df = pd.DataFrame([val.__dict__ for val in data_list]).rename(columns={'barCount': 'nr_of_trades'})
                if interval == Interval.daily:
                    res_df['date'] = pd.to_datetime(res_df['date']).dt.tz_localize(local_time_zone)
                else:
                    res_df['date'] = res_df['date'].dt.tz_convert(local_time_zone)
            logger.info("{} nr of {} entries downloaded for {}".format(res_df.shape[0], interval, asset))
            return res_df.drop_duplicates().sort_values('date')
        for _interval in intervals:
            data_df = download_historical_data_batchwise(
                start_time=get_start_time(_interval), batch_size=OPTIMAL_BATCH_SIZES[_interval],
                interval=_interval)
            self.data_accessor.write_raw_asset_data(asset, _interval, data_df.astype(RAW_DATA_COLUMN_TYPES))
            if update_feature_data:
                if _interval == Interval.five_min:
                    self.preprocess_raw_data(asset, False)

    def collect_realtime_data(self, asset: Asset) -> None:
        raise NotImplementedError

    def stop_collecting_realtime_data(self, asset: Asset) -> None:
        raise NotImplementedError
