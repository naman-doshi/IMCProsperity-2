import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections
import pandas as pd
import math
import copy
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections
import pandas as pd
import numpy as np
import math

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

    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100, 'CHOCOLATE': 240, 'STRAWBERRIES':300, 'ROSES':60, 'GIFT_BASKET':60, "COCONUT_COUPON": 600, "COCONUT": 300}
    mean = 3
    sd = 8
    std_threshold = 55
    starfruit_dim = 4
    curOrders = {}
    starfruit_cache = []
    orchids_cache = []
    orchid_ma_differential = 0
    INF = 1e9

    def updateOrchidCache(self, state):
        buy = state.observations.conversionObservations['ORCHIDS'].bidPrice
        sell = state.observations.conversionObservations['ORCHIDS'].askPrice
        mid = (buy + sell) / 2
        if len(self.orchids_cache) >= 1:
            self.orchid_ma_differential = mid - self.orchids_cache[-1]
        self.orchids_cache.append(mid)
        

    def calc_next_price_starfruit(self):
        # starfruit cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price
        coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]
        intercept = 4.481696494462085
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]
        return int(round(nxt_price))
    
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders_regression(self, state, product, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []
        order_depth = state.order_depths.get(product, 0)

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = state.position.get(product, 0)
        po2 = cpos

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((po2<0) and (ask == acc_bid+1))) and cpos < LIMIT and abs(ask) != self.INF:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT and abs(bid_pr) != self.INF:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = po2
    
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((po2>0) and (bid+1 == acc_ask))) and cpos > -LIMIT and abs(bid) != self.INF:
                order_for = max(-vol, -LIMIT-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT and abs(sell_pr) != self.INF:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num
        

        self.curOrders[product] = orders

    def starfruitMM(self, state):

        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)

        _, bs_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.starfruit_cache.append((bs_starfruit+bb_starfruit)//2)

        starfruit_lb = -self.INF
        starfruit_ub = self.INF

        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit() - 1
            starfruit_ub = self.calc_next_price_starfruit() + 1
        
        self.compute_orders_regression(state, 'STARFRUIT', starfruit_lb, starfruit_ub, 20)

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

            sprice = list(osell.keys())[0]

            if pos < 100:
                orders.append(Order(product, min(sprice-2, int(sellR-1)), 100-pos))

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

        if res_sell > trade_at:
            vol = state.position.get('GIFT_BASKET', 0) + self.POSITION_LIMIT['GIFT_BASKET']
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
            # for product in ['STRAWBERRIES', 'CHOCOLATE', 'ROSES']:
            #     vol = self.POSITION_LIMIT[product] - state.position.get(product, 0)
            #     orders[product].append(Order(product, worst_sell[product], vol))
            
        
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - state.position.get('GIFT_BASKET', 0)
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
            # for product in ['STRAWBERRIES', 'CHOCOLATE', 'ROSES']:
            #     vol = state.position.get(product, 0) + self.POSITION_LIMIT[product]
            #     orders[product].append(Order(product, worst_buy[product], -vol))

        for product in prods:
            self.curOrders[product] = orders[product]

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
        theo = self.black_scholes(3 + state.timestamp / 1000000, self.coconut_price(state))
        dx = theo - price - self.mean
        dx /= self.sd
        
        delta = self.calculate_delta(3 + state.timestamp / 1000000, self.coconut_price(state))
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
        
        for product in prods:
            self.curOrders[product] = orders[product]
    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        
        if state.timestamp != 0:
            self.starfruit_cache = json.loads(json.loads(state.traderData)["starfruit_cache"])
            self.orchids_cache = json.loads(json.loads(state.traderData)["orchids_cache"])

        self.compute_orders_regression(state, 'AMETHYSTS', 9999, 10001, 20)
        self.starfruitMM(state)
        self.updateOrchidCache(state)
        conversions = self.orchidArbitrage('ORCHIDS', state)
        self.compute_orders_basket(state)
        self.coconut_strategy(state)

        orders = self.curOrders
        trader_data = json.dumps({
            "starfruit_cache": json.dumps(self.starfruit_cache),
            "orchids_cache": json.dumps(self.orchids_cache)
        })
    
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
    
    