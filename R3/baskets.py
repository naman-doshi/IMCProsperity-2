import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections
import pandas as pd
import math
import copy

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()
empty = {'STRAWBERRIES': [], 'CHOCOLATE': [], 'ROSES': [], 'GIFT_BASKET': []}

class Trader:

    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE':240, 'STRAWBERRIES': 300, 'ROSES':60, 'GIFT_BASKET': 60}
    orders = copy.deepcopy(empty)
    sd = 75
    mean = 0.5
    tradeOpen = False
    leftToBuy = {}
    leftToSell = {}

    df = pd.DataFrame(columns=['combined', 'basket', 'zscore', 'zscorema'])

    def pred(self, straw, choc, rose):
        return straw * 6 + choc * 4 + rose + 380
    
    def record(self, state):
        depths = state.order_depths
        bask, straw, choc, rose = 0, 0, 0, 0
        for symbol, order_depth in depths.items():
            midprice = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
            if symbol == 'STRAWBERRIES':
                straw = midprice
            elif symbol == 'CHOCOLATE':
                choc = midprice
            elif symbol == 'ROSES':
                rose = midprice
            elif symbol == 'GIFT_BASKET':
                bask = midprice
        
        comb = self.pred(straw, choc, rose)
        diff = comb - bask
        zscore = (diff - self.mean) / self.sd
        self.df.loc[len(self.df)] = {'combined': comb, 'basket': bask, 'zscore': zscore, 'zscorema': 0}
        self.df = self.df.reset_index(drop=True)
        self.df['zscorema'] = self.df['zscore'].rolling(window=50).mean()

    def updateBuys(self, state):
        for i in range(len(self.leftToBuy.items())):
            product, volume = list(self.leftToBuy.items())[i]
            depths = state.order_depths[product]
            order = []
            osell = collections.OrderedDict(sorted(depths.sell_orders.items()))

            for price, quantity in osell.items():
                volume = self.leftToBuy[product]
                vol = min(-quantity, volume)
                self.leftToBuy[product] -= vol
                if vol != 0:
                    order.append(Order(product, price, vol))

            self.orders[product] = order

    def updateSells(self, state):
        for i in range(len(self.leftToSell.items())):
            product, volume = list(self.leftToSell.items())[i]
            depths = state.order_depths[product]
            order = []
            obuy = collections.OrderedDict(sorted(depths.buy_orders.items(), reverse=True))

            for price, quantity in obuy.items():
                volume = self.leftToSell[product]
                vol = min(quantity, volume)
                self.leftToSell[product] -= vol
                if vol != 0:
                    order.append(Order(product, price, -vol))

            self.orders[product] = order

    def basketStrat(self, state):
        realz = self.df['zscorema'].to_list()
        zscores = self.df['zscore'].to_list()
        products = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        compProducts = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES']
        compPrice = self.df['combined'].to_list()
        basketPrice = self.df['basket'].to_list()

        if abs(realz[-1]) <= 0.1 and self.tradeOpen:
            self.tradeOpen = False
            self.leftToBuy = {}
            self.leftToSell = {}
            for product in products:
                pos = state.position.get(product, 0)
                if pos >= 0:
                    self.leftToSell[product] = pos
                else:
                    self.leftToBuy[product] = -pos
            logger.print("CLOSING TRADE")
            logger.print(self.leftToBuy)
            logger.print(self.leftToSell)

        if abs(zscores[-1]) < abs(zscores[-2]) and not self.tradeOpen and abs(zscores[-1]) > 1:
            self.tradeOpen = True
            logger.print("OPENING TRADE")
            self.leftToBuy = {}
            self.leftToSell = {}

            if compPrice < basketPrice:
                #buy all of these
                for product in compProducts:
                    pos = state.position.get(product, 0)
                    vol = self.POSITION_LIMIT[product] - pos
                    self.leftToBuy[product] = vol
                # sell basket
                pos = state.position.get('GIFT_BASKET', 0)
                vol = self.POSITION_LIMIT['GIFT_BASKET'] + pos
                self.leftToSell['GIFT_BASKET'] = vol
            
            else:
                # sell all of these
                for product in compProducts:
                    pos = state.position.get(product, 0)
                    vol = self.POSITION_LIMIT[product] + pos
                    self.leftToSell[product] = vol
                # buy basket
                pos = state.position.get('GIFT_BASKET', 0)
                vol = self.POSITION_LIMIT['GIFT_BASKET'] - pos
                self.leftToBuy['GIFT_BASKET'] = vol

        self.updateBuys(state)
        self.updateSells(state)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        self.orders = copy.deepcopy(empty)

        self.record(state)

        if state.timestamp >= 51:
            self.basketStrat(state)


        
        logger.flush(state, self.orders, conversions, trader_data)
        return self.orders, conversions, trader_data