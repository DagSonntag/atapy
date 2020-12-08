from unittest import TestCase
from atapy.data_accessor import DataAccessor
import pandas as pd
import pytz
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.utils import to_datetime


class TestFileAccessor(TestCase):

    def setUp(self) -> None:
        self.accessor = DataAccessor.factory('file', data_folder='test_data')
        self.asset1 = Asset(exchange='AEB', symbol='INGA', type='stk')

    def tearDown(self) -> None:
        self.accessor.clear()

    def test_write_asset_information(self):
        """ An accessor should be able to read and write asset information. This includes the columns:
        [security_type, exchange, symbol, currency, full_name, industry, category, subcategory, liquid_hours] """
        first_row = pd.Series(
            {'asset_type': 'STK', 'exchange': 'AEB', 'symbol': 'ASML', 'currency': 'EUR',
             'full_name': 'ASML HOLDING NV', 'industry': 'Technology', 'category': 'Semiconductors',
             'subcategory': 'Semiconductor Equipment', 'liquid_hours': '0900-1730', 'time_zone': 'MET'})
        self.accessor.write_asset_information(pd.DataFrame(first_row).T)
        second_row = pd.Series(
            {'asset_type': 'STK', 'exchange': 'AEB', 'symbol': 'INGA', 'currency': 'EUR',
             'full_name': 'ASML HOLDING NV', 'industry': 'Technology', 'category': 'Semiconductors',
             'subcategory': 'Semiconductor Equipment', 'liquid_hours': '0900-1730', 'time_zone': 'MET'})
        self.accessor.write_asset_information(pd.DataFrame(second_row).T)

    def test_get_asset_information(self):
        self.assertEqual(self.accessor.get_asset_information().shape[0], 0)
        self.assertGreaterEqual(self.accessor.get_asset_information().shape[1], 9)
        self.test_write_asset_information()
        self.assertEqual(self.accessor.get_asset_information().shape[0], 2)
        self.assertEqual(self.accessor.get_asset_information(asset_symbol='INGA').shape[0], 1)

    def test_write_raw_asset_data(self):
        """ An accessor should be able to read and write raw asset data. Asset data is identified by its asset and the
            interval and contain at least the columns ()"""
        df = pd.DataFrame({'date': {0: pd.Timestamp('1991-02-25 00:00:00'),
                                    1: pd.Timestamp('1991-02-26 00:00:00'),
                                    2: pd.Timestamp('1991-02-27 00:00:00'),
                                    3: pd.Timestamp('1991-02-28 00:00:00'),
                                    4: pd.Timestamp('1991-03-01 00:00:00')},
                           'open': {0: 46.1, 1: 45.8, 2: 45.2, 3: 45.4, 4: 45.0},
                           'high': {0: 46.1, 1: 46.0, 2: 45.4, 3: 45.5, 4: 46.1},
                           'low': {0: 46.1, 1: 45.8, 2: 45.0, 3: 45.0, 4: 45.0},
                           'close': {0: 46.1, 1: 45.8, 2: 45.2, 3: 45.4, 4: 45.0},
                           'volume': {0: 1046, 1: 3314, 2: 987, 3: 2853, 4: 3865},
                           'average': {0: 46.1, 1: 45.8, 2: 45.2, 3: 45.4, 4: 45.0},
                           'nr_of_trades': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}})
        df['date'] = df.date.dt.tz_localize(pytz.timezone('MET'))
        self.accessor.write_raw_asset_data(self.asset1, Interval.daily, data_df=df)

    def test_read_raw_asset_data(self):
        self.test_write_asset_information()
        self.assertEqual(self.accessor.get_raw_asset_data(self.asset1, Interval.daily).shape[0], 0)
        self.assertGreaterEqual(self.accessor.get_raw_asset_data(self.asset1, Interval.daily).shape[1], 8)
        self.test_write_raw_asset_data()
        self.assertEqual(self.accessor.get_raw_asset_data(self.asset1, Interval.daily).shape[0], 5)

    # def test_get_feature_asset_data(self):
    #     """ The accessor is responsible for preprocessing data into features when data is saved (optional) or when
    #     a preprocess method is called. The function used for preprocessing can be supplied in its init method """
    #     pass

    # def test_preprocess_asset(self):
    #     pass
    #

    def test_get_time_restricted_instance(self):
        """ An accessor should be restricting data access on request, allowing it to be passed to agents only able to
        access restricted data (in terms of time) """
        self.test_write_asset_information()
        self.test_write_raw_asset_data()
        new_accessor = self.accessor.get_time_restricted_instance(start_time=to_datetime('1991-02-26'),
                                                                  end_time=to_datetime('1991-02-28'))
        self.assertEqual(new_accessor.get_raw_asset_data(self.asset1, Interval.daily).shape[0], 2)

    def test_write_custom_table(self):
        """ An accessor must also be able to handle arbitrary tables used by the various data connections"""
        df = pd.DataFrame({'date': {0: pd.Timestamp('1991-02-25 00:00:00'),
                                    1: pd.Timestamp('1991-02-26 00:00:00'),
                                    2: pd.Timestamp('1991-02-27 00:00:00'),
                                    3: pd.Timestamp('1991-02-28 00:00:00'),
                                    4: pd.Timestamp('1991-03-01 00:00:00')},
                           'open': {0: 46.1, 1: 45.8, 2: 45.2, 3: 45.4, 4: 45.0},
                           'high': {0: 46.1, 1: 46.0, 2: 45.4, 3: 45.5, 4: 46.1},
                           'low': {0: 46.1, 1: 45.8, 2: 45.0, 3: 45.0, 4: 45.0},
                           'close': {0: 46.1, 1: 45.8, 2: 45.2, 3: 45.4, 4: 45.0},
                           'volume': {0: 1046, 1: 3314, 2: 987, 3: 2853, 4: 3865},
                           'average': {0: 46.1, 1: 45.8, 2: 45.2, 3: 45.4, 4: 45.0},
                           'nr_of_trades': {0: 3, 1: 1, 2: 1, 3: 2, 4: 1}})
        df['date'] = df.date.dt.tz_localize(pytz.timezone('MET'))
        self.accessor.write_custom_table('test_table', df)

    def test_read_custom_table(self):
        self.test_write_custom_table()
        self.assertEqual(self.accessor.read_custom_table('test_table').shape[0], 5)
        self.assertEqual(self.accessor.read_custom_table('test_table',
                                                         "SELECT * from df where nr_of_trades > 1").shape[0], 2)
