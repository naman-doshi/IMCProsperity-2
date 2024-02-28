import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
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

logger = Logger()

class Trader:
    curOrders = {}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    bananas_dim = 4
    bananas_cache = []
    
    def staticMM(self, state, product, theo=10000):
        # theo is the theoretical price of pearls
        orders: list[Order] = []
        limit = self.POSITION_LIMIT[product]
        order_depth: OrderDepth = state.order_depths.get(product, 0)

        if order_depth == 0:
            return
        
        myPosition = state.position.get(product, 0)
        sells = order_depth.sell_orders # asks
        buys = order_depth.buy_orders # bids
        sellPrices = sorted(list(sells.keys()))
        buyPrices = sorted(list(buys.keys()))
        best_ask = sellPrices[0] if sellPrices else -1
        best_bid = buyPrices[-1] if buyPrices else -1

        theo -= 0.05 * myPosition

        if best_bid > theo:
            for p in buyPrices[::-1]:
                # sell as much as possible above theo price
                if p < theo:
                    break
                sell_q = min(buys[p], limit + myPosition)

                if sell_q:
                    orders.append(Order(product, p, -sell_q))
                    myPosition -= sell_q

                if myPosition <= -limit:
                    break

            p = best_bid+1

            if p != best_ask and myPosition > -limit:
                orders.append(Order(product, p, -limit-myPosition)) # keep probing

        if best_ask < theo:
            for p in sellPrices:
                # buy as much as possible below theo price
                if p > theo:
                    break

                buy_q = min(-sells[p], limit - myPosition)

                if buy_q:
                    orders.append(Order(product, p, buy_q))
                    myPosition += buy_q

                if myPosition >= limit:
                    break

            p = best_ask-1
            if p != best_bid and myPosition < limit:
                orders.append(Order(product, p, limit-myPosition))

        if best_bid < theo and best_ask > theo and best_bid != -1 and best_ask != -1:
            # money making portion

            qbuy = limit-myPosition
            qsell = limit+myPosition

            if best_bid + 1 < best_ask - 1:
                orders.append(Order(product, best_bid+1, qbuy))
                orders.append(Order(product, best_ask-1, -qsell))
            else:
                if myPosition>0:
                    orders.append(Order(product, best_ask-1, -myPosition))
                elif myPosition<0:
                    orders.append(Order(product, best_bid+1, -myPosition))
        self.curOrders[product] = orders
        return

    
    def calc_next_price_bananas(self):
        # bananas cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]
        intercept = 4.481696494462085
        nxt_price = intercept
        for i, val in enumerate(self.bananas_cache):
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
            if ((ask <= acc_bid) or ((po2<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = po2
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((po2>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num
        
        logger.print(orders)

        self.curOrders[product] = orders
    
    
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        self.staticMM(state, 'AMETHYSTS')

        if len(self.bananas_cache) == self.bananas_dim:
            self.bananas_cache.pop(0)

        _, bs_bananas = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_bananas = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.bananas_cache.append((bs_bananas+bb_bananas)/2)

        INF = 1e9

        bananas_lb = -INF
        bananas_ub = INF

        if len(self.bananas_cache) == self.bananas_dim:
            bananas_lb = self.calc_next_price_bananas()-1
            bananas_ub = self.calc_next_price_bananas()+1

        logger.print(bananas_lb, bananas_ub)
        
        self.compute_orders_regression(state, 'STARFRUIT', bananas_lb, bananas_ub, 20)

        orders = self.curOrders

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data