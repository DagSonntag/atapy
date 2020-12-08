from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """
    A class that identifies an asset. Key attributes are the exchange it is traded on, its (local) symbol on that
    exchange and the type of asset it is
    """
    exchange: str
    symbol: str
    type: str
