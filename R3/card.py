import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections
import pandas as pd
import math
import copy

empty_dict = {'STRAWBERRIES': 0, 'CHOCOLATE': 0, 'ROSES': 0, 'GIFT_BASKET': 0}

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

    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE':250, 'STRAWBERRIES':350, 'ROSES':60, 'GIFT_BASKET':60}
    std_threshold = 55
    orders = {}
    bt, st = -1, -1
    choc_cache = []
    choc_ma1 = 0
    choc_ma2 = 0
    straw_cache = []
    straw_ma1 = 0
    straw_ma2 = 0
    rose_cache = []
    rose_ma1 = 0
    rose_ma2 = 0
    buys, sells = [], []

    def update_choc(self, state):
        depth = state.order_depths['CHOCOLATE']
        osell = collections.OrderedDict(sorted(depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(depth.buy_orders.items(), reverse=True))
        mid = (next(iter(osell)) + next(iter(obuy)))/2

        if len(self.choc_cache) == 1000:
            self.choc_cache.pop(0)
            self.choc_cache.append(mid)
            self.choc_ma1 = sum(self.choc_cache)/1000
            self.choc_ma2 = sum(self.choc_cache[-300:])/300
        else:
            self.choc_cache.append(mid)

    def update_straw(self, state):
        depth = state.order_depths['STRAWBERRIES']
        osell = collections.OrderedDict(sorted(depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(depth.buy_orders.items(), reverse=True))
        mid = (next(iter(osell)) + next(iter(obuy)))/2

        if len(self.straw_cache) == 1000:
            self.straw_cache.pop(0)
            self.straw_cache.append(mid)
            self.straw_ma1 = sum(self.straw_cache)/1000
            self.straw_ma2 = sum(self.straw_cache[-100:])/100
        else:
            self.straw_cache.append(mid)

    def update_rose(self, state):
        depth = state.order_depths['ROSES']
        osell = collections.OrderedDict(sorted(depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(depth.buy_orders.items(), reverse=True))
        mid = (next(iter(osell)) + next(iter(obuy)))/2

        if len(self.rose_cache) == 300:
            self.rose_cache.pop(0)
            self.rose_cache.append(mid)
            self.rose_ma1 = sum(self.rose_cache)/300
            self.rose_ma2 = sum(self.rose_cache[-100:])/100
        else:
            self.rose_cache.append(mid)



    def compute_orders_basket(self, state):

        order_depth = state.order_depths
        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2

        res_buy = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 381
        res_sell = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 381

        trade_at = self.std_threshold

        if self.choc_ma1 != 0:
            if self.choc_ma2 < self.choc_ma1 - 3:
                vol = state.position.get('CHOCOLATE', 0) + self.POSITION_LIMIT['CHOCOLATE']
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol))
            
            elif self.choc_ma2 > self.choc_ma1 + 3:
                vol = self.POSITION_LIMIT['CHOCOLATE'] - state.position.get('CHOCOLATE', 0)
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))

        if self.straw_ma1 != 0:
            if self.straw_ma2 < self.straw_ma1 - 2:
                vol = state.position.get('STRAWBERRIES', 0) + self.POSITION_LIMIT['STRAWBERRIES']
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol))
            
            elif self.straw_ma2 > self.straw_ma1 + 2:
                vol = self.POSITION_LIMIT['STRAWBERRIES'] - state.position.get('STRAWBERRIES', 0)
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))

        if self.rose_ma1 != 0:
            if self.rose_ma2 < self.rose_ma1 - 4:
                vol = state.position.get('ROSES', 0) + self.POSITION_LIMIT['ROSES']
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol))
            
            elif self.rose_ma2 > self.rose_ma1 + 4:
                vol = self.POSITION_LIMIT['ROSES'] - state.position.get('ROSES', 0)
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))


        if res_sell > trade_at:
            vol = state.position.get('GIFT_BASKET', 0) + self.POSITION_LIMIT['GIFT_BASKET']
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
            # vol = state.position.get('CHOCOLATE', 0) + self.POSITION_LIMIT['CHOCOLATE']
            # orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol))
            
        
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - state.position.get('GIFT_BASKET', 0)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
            # vol = self.POSITION_LIMIT['CHOCOLATE'] - state.position.get('CHOCOLATE', 0)
            # orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))

        self.orders = orders
    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        
        self.update_choc(state)
        self.update_straw(state)
        self.update_rose(state)
        self.compute_orders_basket(state)

        logger.print(self.buys)
        logger.print(self.sells)

        orders = self.orders
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data