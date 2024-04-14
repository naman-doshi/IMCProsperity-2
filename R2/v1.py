import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections

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

    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    starfruit_dim = 4
    curOrders = {}
    starfruit_cache = []
    orchids_cache = []
    INF = 1e9
    
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

        order_depths = state.order_depths.get(product, 0)
        osell = collections.OrderedDict(sorted(order_depths.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depths.buy_orders.items(), reverse=True))

        # fill all buy orders above buyP
        bf = 0
        pos = state.position.get('ORCHIDS', 0)
        for buy, vol in obuy.items():
            if buy > buyP:
                order_for = min(-vol, -100-pos)
                pos += order_for
                orders.append(Order(product, buy, order_for))
                bf += 1
        
        # fill all sell orders below sellR
        sf = 0
        for sell, vol in osell.items():
            if sell < sellR:
                order_for = min(vol, 100+pos)
                pos -= order_for
                orders.append(Order(product, sell, order_for))
                sf += 1

        # undercut lowest unfilled sell order
        price, vol = list(osell.items())[sf]
        if price - 1 > buyP:
            orders.append(Order(product, price-1, -100-pos))
        
        # undercut highest unfilled buy order
        price, vol = list(obuy.items())[bf]
        if price + 1 < sellR:
            orders.append(Order(product, price+1, 100+pos))

        self.curOrders[product] = orders

        conv = -state.position.get(product, 0)

        return conv


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        
        if state.timestamp != 0:
            self.starfruit_cache = json.loads(json.loads(state.traderData)["starfruit_cache"])
            # self.orchids_cache = json.loads(json.loads(state.traderData)["orchids_cache"])

        self.compute_orders_regression(state, 'AMETHYSTS', 9999, 10001, 20)
        self.starfruitMM(state)
        conversions = self.orchidArbitrage('ORCHIDS', state)

        orders = self.curOrders
        trader_data = json.dumps({
            "starfruit_cache": json.dumps(self.starfruit_cache),
            # "orchids_cache": json.dumps(self.orchids_cache)
        })
    
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data