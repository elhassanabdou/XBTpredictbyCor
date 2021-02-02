#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:40:41 2020

@author: elhassan
"""

import  krakenex
from decimal import Decimal as D
import pprint
import pandas as pd
import time
import numpy as np
from datetime import date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def get_ohlc_pd(pair, interval = 1, since=None,ascending=False):
    
    
    data = {arg:value for arg, value in locals().items() if arg !='self' and value is not None}
    #print(data)
    res = k.query_public('OHLC',data=data)
    if len(res['error']) > 0:
        raise KrakenAPIError(res['error'])
    pair = list(res['result'].keys())[0]
    ohlc = pd.DataFrame(res['result'][pair])
    last = res['result']['last']
    if ohlc.empty:
        return ohlc,last
    else:
        ohlc.columns=[
            'time', 'open', 'high', 'low', 'close',
            'vwap', 'volume', 'count',
        ]
        ohlc['dtime'] = pd.to_datetime(ohlc.time,unit='s')
        
        #ohlc.sort_values('dtime',ascending=ascending,inplace=True)
        ohlc.set_index('dtime',inplace=True)
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
            ohlc.loc[:, col] = ohlc[col].astype(float)
        #del ohlc['time']
        return ohlc, last


k = krakenex.API()
k.load_key('keys')

#balance = k.query_private('Balance')
#print(balance)

#orders = k.query_private('OpenOrders')

#balance = balance['result']

#orders = orders['result']
#print(orders)

#newbalance= dict()

#for currency in balance:
 #   newname = currency[1:] if len(currency) ==4 and currency !="KAVA" else currency
  #  newbalance[newname] = D(balance[currency])
#balance =newbalance


d =date(2020,10,24)
uxtime =str( time.mktime(d.timetuple()))
ohlcxbt,lastxbt = get_ohlc_pd("XBTUSD",1,uxtime)
ohlceth,lasteth = get_ohlc_pd("ETHUSD",1,uxtime)
ohlcxtz,lastxtz = get_ohlc_pd("XTZUSD",1,uxtime)
ohlckava,lastkava = get_ohlc_pd("KAVAUSD",1,uxtime)

del ohlcxbt['time']
del ohlcxbt['vwap']
del ohlcxbt['volume']
del ohlcxbt['count']
del ohlceth['time']
del ohlceth['vwap']
del ohlceth['volume']
del ohlceth['count']
del ohlcxtz['time']
del ohlcxtz['vwap']
del ohlcxtz['volume']
del ohlcxtz['count']
del ohlckava['time']
del ohlckava['vwap']
del ohlckava['volume']
del ohlckava['count']
whiteP, blackP, redP, greyP = '#FFFFFF', '#000000', '#FF4136', 'rgb(150,150,150)'
ts = {}
ts['BTC'] = ohlcxbt
ts['ETH'] = ohlceth
ts['XTZ'] = ohlcxtz
ts['KAVA'] = ohlckava
fig = go.Figure(data=go.Candlestick(x=ts['BTC'] .index,
                                    open = ts['BTC'] .iloc[:,0],
                                    high = ts['BTC'] .iloc[:,1],
                                    low = ts['BTC'] .iloc[:,2],
                                    close = ts['BTC'] .iloc[:,3],
                                    ))
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_layout(plot_bgcolor=whiteP, width=500)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=greyP)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=greyP)
fig.update_yaxes(title_text='BTC/USD')
cs = fig.data[0]
cs.increasing.fillcolor, cs.increasing.line.color = blackP, blackP
cs.decreasing.fillcolor, cs.decreasing.line.color = redP, redP

fig.show()


# In[2]:


fig = go.Figure(data=go.Candlestick(x=ts['XTZ'].index,
                                    open = ts['XTZ'].iloc[:,0],
                                    high = ts['XTZ'].iloc[:,1],
                                    low = ts['XTZ'].iloc[:,2],
                                    close = ts['XTZ'].iloc[:,3],
                                    ))
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_layout(plot_bgcolor=whiteP, width=500)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=greyP)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=greyP)
fig.update_yaxes(title_text='XTZ/USD')
cs = fig.data[0]
cs.increasing.fillcolor, cs.increasing.line.color = blackP, blackP
cs.decreasing.fillcolor, cs.decreasing.line.color = redP, redP

fig.show()


# In[3]:


btc = ts['BTC']  # assign BTC data to a new temporary variable

#btc.open.rolling(50, win_type='gaussian').sum(std=7)
#btc.close.rolling(50, win_type='gaussian').sum(std=7)
btc = btc[['open', 'close']]  # limit to Open and Close series
# add a new column with Close price 1 min ago
#btc.loc['BTCUSD_C_LAG1'] = btc['close'].shift(1)
btc['BTCUSD_C_LAG1'] = btc['close'].shift(1)
print(ts['BTC'])


# In[4]:


# define a supporting function
def rr(z):
    '''Calculates rate of return [percent].
       Works with two DataFrame's columns as an input.
    '''
    x, y = z[0], z[1]
    return 100*(x/y-1)
 
 
# calculate rate of return between:
btc['rate_of_reutrn'] = btc[['open', 'close']].apply(rr, axis=1)
 
# get rid of NaN rows
btc = btc.dropna()
display(btc)


# In[6]:


thr = 0.3 # 1-min rate of return greater than 'thr' percent
blue, orange, red, green = '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'
grey8 = (.8,.8,.8)
tmp = btc[btc.rate_of_reutrn > thr]
 
fig, ax = plt.subplots(1,1,figsize=(15,5))
ax.plot((btc.close), color=grey8)
ax.plot([tmp.index, tmp.index], [tmp.open, tmp.close], color=red)
ax.grid()
ax.legend(['BTCUSD Close Price', 'Triggers'])
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)


# In[7]:


print(tmp.index)


# In[8]:


ind_buy = tmp.index + pd.Timedelta(minutes = 1)
print(ind_buy)


# In[10]:


def check_stoploss(z, thr1=-0.15, thr2=-0.15):
    p1, p2 = z
    if p1 < thr1 or p2 < thr2:
        return False  # close position
    else:
        return True  # hold position open


# In[11]:


backtested_coins = ['BTC','ETH','XTZ','KAVA'] 
results = {}
 
for coin in backtested_coins:
 
    # read OHLC price time-series
    df = ts[coin] 
    #print (df)
    tradePnLs = list()
 
    for ib in range(len(ind_buy)):
        i = ind_buy[ib]
        try:
            op = df.loc[i][0]
            #print(op)
            # Trade No. 'ib' DataFrame
            tmp = df[df.index >= i]
            tmp['open'] = op  # trade's open price
            tmp['current_price'] = df['close']
            tmp['pnl'] = tmp['current_price'] / op - 1
            #print(tmp)

            fi = True
            out1 = list()
            out2 = list()
            for j in range(tmp.shape[0]):
                if fi:
                    maxPnL = tmp.pnl[j]
                    maxClose = tmp.iloc[j, 3]
                    fi = False
                else:
                    if tmp.pnl[j] > maxPnL:
                        maxPnL = tmp.pnl[j]
                        maxClose = tmp.iloc[j, 3]
                out1.append(maxPnL)
                out2.append(maxClose)  # close price
 
            tmp['maxPnL'] = out1
            tmp['maxClose'] = out2
            tmp['drawdown'] = tmp['current_price'] / tmp['maxClose'] - 1
            tmp['hold'] = tmp[['pnl', 'drawdown']].apply(check_stoploss, axis=1)

            # execute selling if detected
            sell_executed = True
            try:
                sell_df = tmp[tmp['hold'] == 0]
                sell_time, close_price = sell_df.index[0], sell_df.current_price[0]
                tmpT = tmp[tmp.index<= sell_time]
            except:
                sell_executed = False
 
            #display(tmp.iloc[:,:].head(10))
 
            plt.figure(figsize=(15,4))
            plt.grid()
            plt.plot(tmp.pnl, color=grey8, label = "Rolling trade's PnL (open trade)")
            if sell_executed:
                plt.plot(tmpT.pnl, color=blue, label = "Rolling trade's PnL (closed)")
                plt.title("Trade's final PnL = %.2f%%" % (100*tmpT.iloc[-1,6]))
                tradePnLs.append(tmpT.iloc[-1,6])
            else:
                plt.title("Current trade's PnL = %.2f%%" % (100*tmp.iloc[-1,6]))
                tradePnLs.append(tmp.iloc[-1,6])
            plt.plot(tmp.maxPnL, color=orange, label = "Rolling maximal trade's PnL")
            plt.plot(tmp.index, np.zeros(len(tmp.index)), '--k')
            plt.suptitle('Trade No. %g opened %s @ %.2f USD' % (ib+1, i, df.loc[i][0]))
            plt.legend()
            locs, labels = plt.xticks()
            plt.xticks(locs, [len(list(labels))*""])
            plt.show()
 
            plt.figure(figsize=(14.85,1.5))
            plt.grid()
            plt.plot(tmp.drawdown, color=red, label = "Rolling trade's drawdown")
            plt.plot(tmp.index, np.zeros(len(tmp.index)), '--k')
            plt.gcf().autofmt_xdate()
            myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
            plt.gca().xaxis.set_major_formatter(myFmt)
            plt.legend()
            plt.show()
 
            print("\n\n")
        except:
            pass
        c = 1000  # initial investment (fixed; per each trade)
        tradePnLs = np.array(tradePnLs)
        n_trades = len(tradePnLs)
        res = pd.DataFrame(tradePnLs, columns=['Trade_PnL'])
        res['Investment_USD'] = c
        res['Trade_ROI_USD'] = np.round(c * (tradePnLs + 1),2)
        res.index = np.arange(1,n_trades+1)
        res.index.name = 'Trade_No'
        ROI = res.Trade_ROI_USD.sum() - (n_trades * c)
        ROI_pct = 100 * (res.Trade_ROI_USD.sum() / (n_trades * c) - 1)
        tot_pnl = res.Trade_ROI_USD.sum()
        res.loc[res.shape[0]+1] = ['', np.round(n_trades * c,2), '']
        res.rename(index = {res.index[-1] : "Total Investment (USD)"}, inplace=True)
        res.loc[res.shape[0]+1] = ['', '', np.round(tot_pnl,2)]
        res.rename(index = {res.index[-1] : "Total PnL (USD)"}, inplace=True)
        res.loc[res.shape[0]+1] = ['', '', np.round(ROI,2)]
        res.rename(index = {res.index[-1] : "Total ROI (USD)"}, inplace=True)
        res.loc[res.shape[0]+1] = ['', '', np.round(ROI_pct,2)]
        res.rename(index = {res.index[-1] : "Total ROI (%)"}, inplace=True)   
        results[coin] = res


# In[12]:


print(results['BTC'])
print(results['ETH'])
print(results['XTZ'])
print(results['KAVA'])


# In[89]:


print(results)


# In[ ]:




