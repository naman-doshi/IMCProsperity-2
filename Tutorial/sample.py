from typing import Dict, List
from datamodel import *
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

class Trader:

    curOrders = {}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    bananaLim = 0
    bananaPrevP = None
    
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
    
    def MAMM(self, state, limit=20, product='STARFRUIT'):

        orders: list[Order] = []
        order_depth: OrderDepth = state.order_depths.get(product, 0)

        if order_depth == 0:
            print("Order book does not contain {}. PearlStrat strategy exiting".format(product))
            return
        myPosition = state.position.get(product, 0)
        sells = order_depth.sell_orders # asks
        buys = order_depth.buy_orders # bids
        sellPrices = sorted(list(sells.keys()))
        buyPrices = sorted(list(buys.keys()))
        best_ask = sellPrices[0] if sellPrices else -1
        best_bid = buyPrices[-1] if buyPrices else -1

        def tosum(D):
            # return the sum of the dot product of key value pairs in dictionary, and the sum of the values
            res, val = 0,0
            for x in D.keys():
                res += D[x]*x
                val += D[x]
            return res, val

        rb, vb = tosum(buys)
        rs, vs = tosum(sells)
        rs, vs = -rs, -vs

        theo = (rb+rs)/(vb+vs) #dynamic theo value

        print("Theo for banana before adjusting is {}".format(theo))
        if self.bananaPrevP:
            bChange = theo-self.bananaPrevP
        else:
            bChange = 0
        self.bananaPrevP = theo

        bb_q = buys[best_bid]
        ba_q = -sells[best_ask]

        theo -= 0.05*myPosition-0.15*bChange


        print("Theo for banana after adjusting is {}".format(theo))

        if best_bid >= theo:
            for p in buyPrices[::-1]:
                # sell as much as possible above theo price
                if p < theo:
                    break
                sell_q = min(buys[p], limit + myPosition)
                if sell_q:
                    print("*******Selling {} for price: {} and quantity: {}".format(product, p, sell_q))
                    orders.append(Order(product, p, -sell_q))
                    myPosition -= sell_q
                if myPosition <= -limit:
                    self.bananaLim += 1
                    break

            p = best_bid+1
            if p != best_ask and myPosition > -limit:
                orders.append(Order(product, p, -limit-myPosition)) #keep probing




        elif best_ask <= theo:
            for p in sellPrices:
                # buy as much as possible below theo price
                if p > theo:
                    break
                buy_q = min(-sells[p], limit - myPosition)
                if buy_q:
                    print("*******Buying {} for price: {} and quantity: {}".format(product, p, buy_q))
                    orders.append(Order(product, p, buy_q))
                    myPosition += buy_q
                if myPosition >= limit:
                    self.bananaLim += 1
                    break

            p = best_ask-1
            if p != best_bid and myPosition < limit:
                orders.append(Order(product, p, limit-myPosition))



        elif best_bid < theo and best_ask > theo and best_bid != -1 and best_ask != -1:
            # money making portion
            print("Potential buy or sell submitted.")

            qbuy = limit-myPosition
            qsell = limit+myPosition
            d_bid = theo-best_bid
            d_ask = best_ask-theo

            if best_bid + 1 != best_ask - 1:
                orders.append(Order(product, best_bid+1, qbuy))
                if qbuy == 0:
                    self.bananaLim+=1
                orders.append(Order(product, best_ask-1, -qsell))
                if qsell==0:
                    self.bananaLim+=1
            else:
                if myPosition>0:
                    orders.append(Order(product, best_ask-1, -myPosition))
                elif myPosition<0:
                    orders.append(Order(product, best_bid+1, -myPosition))
        if orders:
            self.curOrders[product] = orders
        return
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # self.staticMM(state, 'AMETHYSTS', 10000)
        self.staticMM(state, 'AMETHYSTS')
        self.MAMM(state, 20, 'STARFRUIT')


        # self.MAMM(state, 20, 'STARFRUIT')

				# Orders to be placed on exchange matching engine
        result = {}


        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return self.curOrders, conversions, traderData