import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections
import pandas as pd

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

class Trader:

    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100}
    starfruit_dim = 4
    curOrders = {}
    starfruit_cache = []
    orchids_cache = []
    sunlightInfo = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    orchid_ma_differential = 0
    INF = 1e9
  
    def processSunlight(self, state):
        cur = state.observations.conversionObservations['ORCHIDS'].sunlight
        self.sunlightInfo[0].append(cur)

        df = pd.DataFrame(columns=['0', '1', '2', '3'])
        df['0'] = self.sunlightInfo[0]
        df['1'] = df['0'].diff()
        df['2'] = df['1'].diff()
        df['3'] = df['2'].diff()

        logger.print(df['3'])

    def updateOrchidCache(self, state):

        buy = state.observations.conversionObservations['ORCHIDS'].bidPrice
        sell = state.observations.conversionObservations['ORCHIDS'].askPrice
        mid = (buy + sell) / 2

        self.orchids_cache.append(mid)

        df = pd.DataFrame(self.orchids_cache, columns=['mid'])
        df['madiff'] = df['mid'].diff()
        self.orchid_ma_differential = df['madiff'][len(df)-1]


    def orchidArbitrage(self, product, state):
        conversion = state.observations.conversionObservations[product]
        orders = []
        conv = 0

        # cover short sell
        buyP = conversion.askPrice + conversion.transportFees + conversion.importTariff
        # sell bought
        sellR = conversion.bidPrice - conversion.transportFees - conversion.exportTariff

        logger.print(buyP, sellR)
        logger.print(self.orchid_ma_differential)

        order_depths = state.order_depths.get(product, 0)
        osell = collections.OrderedDict(sorted(order_depths.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depths.buy_orders.items(), reverse=True))

        pos = 0

        # fill all sell orders below sellR
        sf = 0
        if self.orchid_ma_differential >= -2:
            for sell, vol in osell.items():
                if sell < sellR and pos < 100:
                    order_for = min(-vol, 100-pos)
                    pos += order_for
                    orders.append(Order(product, sell, order_for))
                    sf += 1

            if pos < 100:
                orders.append(Order(product, int(sellR-1), 100-pos))

        # fill all buy orders above buyP
        bf = 0

        if self.orchid_ma_differential <= 2:
            for buy, vol in obuy.items():
                if buy > buyP and pos > -100:
                    order_for = max(-vol, -100-pos)
                    pos += order_for
                    orders.append(Order(product, buy, order_for))
                    bf += 1

            bprice = list(obuy.keys())[0]

            if pos > -100:
                amt = 100+pos
                orders.append(Order(product, max(bprice+2, int(buyP+1)), -amt))
        
        self.curOrders[product] = orders

        conv = -state.position.get(product, 0)

        return conv


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        
        if state.timestamp != 0:
            self.starfruit_cache = json.loads(json.loads(state.traderData)["starfruit_cache"])
            self.orchids_cache = json.loads(json.loads(state.traderData)["orchids_cache"])

        # self.compute_orders_regression(state, 'AMETHYSTS', 9999, 10001, 20)
        # self.starfruitMM(state)
        self.updateOrchidCache(state)
        self.processSunlight(state)
        conversions = self.orchidArbitrage('ORCHIDS', state)

        orders = self.curOrders
        trader_data = json.dumps({
            "starfruit_cache": json.dumps(self.starfruit_cache),
            "orchids_cache": json.dumps(self.orchids_cache),
        })
    
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data