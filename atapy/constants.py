from pathlib import Path

LIBRARY_DIR = str(Path(__file__).parent)
AGENTS_DIR = LIBRARY_DIR + '/agents'
PORTFOLIO_HANDLERS_DIR = LIBRARY_DIR + '/portfolio_handlers'
MARKET_DATA_CONNECTIONS_DIR = LIBRARY_DIR + '/market_data_connections'
DATA_ACCESSORS_DIR = LIBRARY_DIR + '/data_accessors'
COLUMN_NAME_USED_BY_EMA = 'close'
