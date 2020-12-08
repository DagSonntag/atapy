from unittest import TestCase
from atapy.asset import Asset


class TestAsset(TestCase):

    def setUp(self) -> None:
        self.asset1 = Asset('AEB', 'INGA', 'stk')
        self.asset2 = Asset(symbol='ASML', exchange='AEB', type='stk')

    def test_params(self):
        """ An asset is identified by its exchange, its symbol, its type and its currency (optional) """
        self.assertEqual(self.asset1.exchange, 'AEB')
        self.assertEqual(self.asset1.symbol, 'INGA')
        self.assertEqual(self.asset1.type, 'stk')

    def test___eq__(self):
        """ Two assets are the same if they have the same parameters """
        self.assertEqual(self.asset1, Asset(symbol='INGA', exchange='AEB', type='stk'))
        self.assertNotEqual(self.asset1, self.asset2)

    def test___hash__(self):
        hash(self.asset1)

