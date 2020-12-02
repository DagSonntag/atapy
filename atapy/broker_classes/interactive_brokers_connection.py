import logging
import numpy as np
import ib_insync as ibi
import pandas as pd
from tqdm import tqdm
import time
import string
import itertools
from io import StringIO

from atapy.data_accessor import DataAccessor
from atapy.asset import Asset
from atapy.interval import Interval
from atapy.exceptions import AssetNotFoundException

logger = logging.getLogger()

"""
The main exchanges table for interactive brokers. This translates interactive brokers internal codes
to mic codes used in this system (and used by other brokers). This information is not available anywhere and hence
has to be supplied here. 
Note 1: Only the exchanges included in the csv table below are used in the current state of the system. To add an 
exchange just add the add the full name, the mic code (searchable in google) and the matching IB code. The full list 
of IB exchanges are as follows:  
'AEB', 'AEQLIT', 'AMEX', 'ARCA', 'ARCAEDGE', 'ASX', 'ASXCEN', 'ATH', 'BACKTEST', 'BATECH', 'BATEDE', 'BATEEN', 'BATEES', 
'BATEUK', 'BATS', 'BELFOX', 'BEX', 'BM', 'BOVESPA', 'BSEI', 'BUX', 'BVL', 'BVME', 'BVME.ETF', 'BYX', 'CARISK', 'CBOE', 
'CDE', 'CFE', 'CFECRYPTO', 'CHINEXT', 'CHIXCH', 'CHIXDE', 'CHIXEN', 'CHIXES', 'CHIXJ', 'CHIXUK', 'CHX', 'CMECRYPTO', 
'CORPACT', 'CPH', 'CSFBALGO', 'DOLLR4LOT', 'DRCTEDGE', 'DTB', 'DXEDE', 'DXEEN', 'DXEES', 'EBS', 'ECBOT', 'EDGEA', 
'EDGX', 'EDXNO', 'ENEXT.BE', 'EURONEXT', 'FOXRIVER', 'FTA', 'FUNDSERV', 'FWB', 'FWB2', 'GETTEX', 'GETTEX2', 'GLOBEX', 
'HEX', 'HKFE', 'IBIS', 'IBIS2', 'ICECRYPTO', 'ICEEU', 'IDEM', 'IEX', 'IPE', 'ISE', 'ISED', 'ISLAND', 'JEFFALGO', 
'JPNNEXT', 'JSE', 'KSE', 'LMEOTC', 'LSE', 'LSEETF', 'LSEIOB1', 'LTSE', 'MEFFRV', 'MEMX', 'MEXDER', 'MEXI', 'MOEX', 
'MONEP', 'N.RIGA', 'N.TALLINN', 'N.VILNIUS', 'NASDAQ', 'NASDAQ.NMS', 'NASDAQ.SCM', 'NSE', 'NYBOT', 'NYMEX', 'NYSE', 
'NYSELIFFE', 'NYSENAT', 'OMS', 'OMXNO', 'ONE', 'OSE', 'OSE.JPN', 'OTCBB', 'OTCLNKECN', 'PEARL', 'PHLX', 'PINK', 
'PINK.CURRENT', 'PINK.INTPREMQX', 'PINK.INTPRIMQX', 'PINK.LIMITED', 'PINK.NOINFO', 'PINK.PREMQX', 'PINK.PRIMQX', 'PRA', 
'PREBORROW', 'PSE', 'PSX', 'PURE', 'SBF', 'SEHK', 'SEHK.CORPACT', 'SEHKNTL', 'SEHKSZSE', 'SFB', 'SGX', 'SNFE', 'SOFFEX', 
'SWB', 'SWB2', 'TASE', 'TGATE', 'TRQXCH', 'TRQXDE', 'TRQXEN', 'TRQXES', 'TRQXUK', 'TSE', 'TSEJ', 'VALUBOND', 
'VALUBONDG', 'VALUBONDM', 'VALUE', 'VENTURE', 'VIRTX', 'VSE', 'WSE', 'TPLUS1'
To find out more info about an exchange go to 
https://www.interactivebrokers.com/en/index.php?f=2222&exch=EXCHANGE&showcategories=STK#productbuffer and read more 
(with EXCHANGE replaced by the symbol above) or use the TWS Client. 
It should be said that not all of the exchanges that have pages are actually usable in the API (or TWS) such as NASDAQ
Note 2: The ISLAND exchange is selected as the NASDAQ exchange, even though other interfaces exists (such as BEX and 
IEX), since it offers most data
"""
EXCHANGES_CSV_STRING = """full_name, mic_code, ib_name\n
                          Euronext Amsterdam, XAMS, AEB\n
                          Euronext Paris, XPAR, SBF\n
                          Euronext Brussels, XBRU, ENEXT.BE\n
                          NASDAQ Copenhagen, XCSE, CPH\n
                          NASDAQ Stockholm, XSTO, SFB\n
                          NASDAQ Oslo, XOSL, OMXNO\n
                          NASDAQ Helsinki, XHEL, HEX\n
                          NASDAQ US, XNAS, ISLAND\n
                          """
EXCHANGES_TRANSLATION_TABLE = pd.read_csv(StringIO(EXCHANGES_CSV_STRING))
DATA_REQUEST_TIMEOUT = 60 * 10  # Time in seconds before a request is considered bad and a new attempt is made
RECONNECT_WAIT_TIME = 300  # The time to wait in seconds before reconnecting in case of disconnect

BARSIZE_SETTING_CONVERSION_DICT = {Interval.daily: '1 day', Interval.hourly: '1 hour', Interval.five_min: '5 mins',
                                   Interval.minute: '1 min'}
OPTIMAL_BATCH_SIZES = {Interval.daily: 365, Interval.hourly: 250, Interval.five_min: 190, Interval.minute: 190}


class InteractiveBrokersConnection:

    def __init__(self, data_accessor: DataAccessor, tws_ip='127.0.0.1', tws_port=7497, keep_portfolio_up_to_date=False):
        self.data_accessor = data_accessor
        self.keep_portfolio_up_to_date = keep_portfolio_up_to_date
        ibi.util.startLoop()
        self.ib = ibi.IB()
        self.tws_ip = tws_ip
        self.tws_port = tws_port
        self.error_events = {}
        self.reconnect()

    def catch_error_event(self, req_id, error_code, message, contract):
        self.error_events[req_id] = (error_code, message, contract)

    def reconnect(self):
        # First, make sure to clean up old connection:
        self.ib.disconnect()
        self.ib.errorEvent.clear()
        # Then create a new connection
        self.ib = ibi.IB()
        for i in range(10):
            connection_id = np.random.randint(100)
            self.ib.connect(self.tws_ip, self.tws_port, clientId=connection_id,
                            keepPortfolioUpToDate=self.keep_portfolio_up_to_date)
            if self.ib.isConnected():
                self.ib.errorEvent += self.catch_error_event
                break
            else:
                logger.info("Unable to connect to TWS on {}:{} using id {}. Trying again...".format(
                    self.tws_ip, self.tws_port, connection_id))
        if self.ib.isConnected():
            logger.info("Connected to TWS on {}:{} using id {}".format(self.tws_ip, self.tws_port, connection_id))
        else:
            raise ConnectionRefusedError("Unable to connect to TWS on {}:{}".format(self.tws_ip, self.tws_port))

    def __del__(self):
        logger.info("Removing IB connection due to garbage collection")
        self.ib.disconnect()
        self.ib.errorEvent.clear()

    def ensure_connection(self):
        # If not connected, reconnect
        while not self.ib.isConnected():
            logger.warning("Client disconnected, reconnecting")
            self.reconnect()
            if not self.ib.isConnected():
                time.sleep(RECONNECT_WAIT_TIME)
            else:
                logger.info("Client reconnected")

    def _discover_exchanges(self):
        #  Collect all contracts to find exchanges
        letters = list(string.ascii_uppercase + string.digits)
        search_combinations = list(itertools.product(letters, repeat=3))
        found_contracts = []
        for search_combination in tqdm(search_combinations, 'Collecting contracts'):
            try_download = True
            while try_download:
                try:
                    self.ensure_connection()
                    res = self.ib.reqMatchingSymbols("".join(search_combination))
                except ConnectionError as e:
                    logger.error("Connection broken, retrying")
                    time.sleep(RECONNECT_WAIT_TIME)
                    continue
                except TimeoutError as e:
                    logger.error("Timeout error, retrying")
                    time.sleep(RECONNECT_WAIT_TIME)
                    continue
                except Exception as e:
                    raise e
                else:
                    try_download = False
                    if res is None:
                        logger.error("No response received for combination {}".format("".join(search_combination)))
                    else:
                        found_contracts.extend([contract_description.contract for contract_description in res])
        contracts_df = pd.DataFrame(found_contracts)
        found_exchanges = list(contracts_df.primaryExchange.unique())

        # Then also add in exchanges that can be found from the reqScannerParameters. The additional exchanges added
        # are normally exchanges that are not the 'primary' exchange for the stocks
        import xml.etree.ElementTree as ET
        xml = self.ib.reqScannerParameters()
        tree = ET.fromstring(xml)
        loc_exchanges = sorted(list(set([e.text for e in tree.findall('.//routeExchange')])))
        known_exchanges = sorted(list(set(loc_exchanges).union(found_exchanges)))
        return known_exchanges

    def _get_ib_contract(self, asset: Asset):
        df = self.data_accessor.read_custom_table(
            'ib_symbol_translation',
            "Select * from ib_symbol_translation where security_type = '{}'".format(asset.type.upper())
            + " and symbol = '{}' and exchange = '{}' and currency = '{}'".format(
                asset.symbol, asset.exchange, asset.currency))
        if df.shape[0] == 0:
            raise AssetNotFoundException(asset)
        else:
            return ibi.Contract(conId=df.contract_id.iloc[0], symbol=df.ib_symbol.iloc[0],
                                exchange=df.ib_exchange.iloc[0], currency=df.currency.iloc[0])
