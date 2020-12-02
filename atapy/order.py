from dataclasses import dataclass, fields
import datetime
from atapy.asset import Asset


@dataclass(frozen=True)
class Order:
    order_type: str = ""
    asset: Asset = None
    action: str = ""
    quantity: int = 0
    creation_time: datetime.datetime = None
    price: float = 0.0

    def __str__(self):
        values = [getattr(self, field.name) for field in fields(self)]
        attr = {field.name: value for field, value in zip(fields(self), values)
                if value != field.default and value == value and value != []}
        return "{}({})".format(self.__class__.__qualname__, ', '.join("{}={!r}".format(k, v) for k, v in attr.items()))

    __repr__ = __str__

    def __eq__(self, other):
        return id(self) == id(other)

    def __lt__(self, other):
        return self.creation_time < other.creation_time

    def __le__(self, other):
        return self.creation_time <= other.creation_time


class MarketOrder(Order):
    def __init__(self, asset, action, quantity, creation_time):
        Order.__init__(self, order_type='MKT', asset=asset, action=action, quantity=quantity,
                       creation_time=creation_time)
