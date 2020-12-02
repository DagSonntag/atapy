from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """
    A class that identifies an asset. Key attributes are the exchange it is traded on, its (local) symbol on that
    exchange, the type of asset and the currency it is traded in.
    """
    exchange: str
    symbol: str
    type: str
    currency: str