import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections
import pandas as pd
import numpy as np
import math

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

    mean = 3
    sd = 8
    curOrders = {}
    POSITION_LIMIT = {"COCONUT_COUPON": 600, "COCONUT": 300}

    def calculate_delta(self, day, S):
        K = 10000
        T = (250-day)/252
        sigma = 0.161615
        # Calculate d1
        d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        # Calculate delta using the CDF of the standard normal distribution
        delta = self.norm_cdf(d1)
        
        return delta

    def norm_cdf(self, x) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes(self, day, S: float):
        K = 10000
        T = (250-day)/252
        sigma = 0.161615
        # Compute d1 and d2
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * self.norm_cdf(d1) - K * self.norm_cdf(d2)
        return call_price
    
    def coconut_price(self, state):
        depth = state.order_depths["COCONUT"]
        buy = max(list(depth.buy_orders.keys()))
        sell = min(list(depth.sell_orders.keys()))
        if (buy == 0 or sell == 0):
            return 0
        return (buy + sell) / 2
    
    def coconut_strategy(self, state: TradingState):
        orders = []
        order_depth = state.order_depths
        orders = {'COCONUT_COUPON': [], 'COCONUT': []}
        prods = ["COCONUT_COUPON", "COCONUT"]
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price = {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            if (len(osell[p]) == 0 or len(obuy[p]) == 0):
                return

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))
            mid_price[p] = (best_sell[p] + best_buy[p])/2

        price = mid_price["COCONUT_COUPON"]
        theo = self.black_scholes(4 + state.timestamp / 1000000, self.coconut_price(state))
        dx = theo - price - self.mean
        dx /= self.sd
        
        delta = self.calculate_delta(4 + state.timestamp / 1000000, self.coconut_price(state))
        logger.print("delta: ", delta)
        if dx > 1:
            coup_pos = state.position.get("COCONUT_COUPON", 0)
            coup_qty = self.POSITION_LIMIT['COCONUT_COUPON'] - coup_pos
            coco_pos = state.position.get("COCONUT", 0)
            coco_qty = - self.POSITION_LIMIT['COCONUT'] - coco_pos
            
            if delta >= 0.5:
                # coco position maximised
                coup_lim = int(300 / delta)
                orders['COCONUT_COUPON'].append(Order("COCONUT_COUPON", worst_sell['COCONUT_COUPON'], coup_lim - coup_pos))
                orders['COCONUT'].append(Order("COCONUT", worst_buy['COCONUT'], coco_qty))
            else:
                # coup position maximised
                coco_lim = int(600 * delta)
                orders['COCONUT_COUPON'].append(Order("COCONUT_COUPON", worst_sell['COCONUT_COUPON'], coup_qty))
                orders['COCONUT'].append(Order("COCONUT", worst_buy['COCONUT'], -coco_lim - coco_pos))


        elif dx < -1:
            coup_pos = state.position.get("COCONUT_COUPON", 0)
            coup_qty = -self.POSITION_LIMIT['COCONUT_COUPON'] - coup_pos
            coco_pos = state.position.get("COCONUT", 0)
            coco_qty = self.POSITION_LIMIT['COCONUT'] - coco_pos

            if delta >= 0.5:
                # coco position maximised
                coup_lim = int(300 / delta)
                logger.print("coup_qty: ", coup_qty)
                orders['COCONUT_COUPON'].append(Order("COCONUT_COUPON", worst_buy['COCONUT_COUPON'], -coup_lim - coup_pos))
                orders['COCONUT'].append(Order("COCONUT", worst_sell['COCONUT'], coco_qty))
            else:
                # coup position maximised
                coco_lim = int(600 * delta)
                orders['COCONUT_COUPON'].append(Order("COCONUT_COUPON", worst_buy['COCONUT_COUPON'], coup_qty))
                orders['COCONUT'].append(Order("COCONUT", worst_sell['COCONUT'], coco_lim - coco_pos))
        
        self.curOrders = orders
  

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""
        self.coconut_strategy(state)
        orders = self.curOrders
    
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data