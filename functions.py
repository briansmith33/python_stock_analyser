import pandas as pd
from bs4 import BeautifulSoup
import requests
from pytz import timezone
import datetime as dt
import pandas_datareader.data as web
import time
import matplotlib.pyplot as plt
from matplotlib import style
import re
from textblob import TextBlob
import bs4 as bs
import pickle
import numpy as np
import sklearn
from sklearn import linear_model
from pathlib import Path

style.use('dark_background')
stocks = ['ADBE', 'FB', 'NFLX', 'CHTR', 'AMZN', 'FISV', 'BMY', 'XOM', 'NVDA', 'QCOM']
# ****AVGO
all_symbols = ['TIF', 'GOOG', 'EXPE', 'NFLX', 'BABA', 'ZG', 'AMZN', 'BMY', 'TDG', 'NVDA', 'VRTX', 'ADBE', 'JPM', 'MSCI',
               'EQIX', 'LRCX', 'QCOM', 'AAPL', 'ANET', 'FISV', 'FTNT', 'SHOP']
long = ['BMY', 'TDG', 'NVDA', 'VRTX', 'ADBE', 'JPM', 'MSCI', 'EQIX', 'NFLX', 'AMZN']
short = ['BMY', 'LRCX', 'QCOM', 'GOOG', 'AAPL', 'MSCI', 'ANET', 'FISV', 'FTNT', 'SHOP']

names = ['Bristol-Myers Squibb', 'TransDigm Group', 'NVIDIA', 'Vertex Pharmaceuticals', 'Adobe', 'JPMorgan Chase',
         'MSCI', 'Equinix', 'Netflix', 'Amazon']


class Trend:
    # Average Directional Movement Index
    '''
    if ADX > 25:
        buy()
    '''
    '''
    *The price is moving up when +DI is above -DI, and the price is moving down when 
    -DI is above +DI.

    *Crosses between +DI and -DI are potential trading signals as bears or bulls gain 
    the upper hand.

    *The trend has strength when ADX is above 25. The trend is weak or the price is 
    trendless when ADX is below 20, according to Wilder.
    '''
    '''
    The average directional index (ADX) is a technical analysis indicator 
    used by some traders to determine the strength of a trend. The trend 
    can be either up or down, and this is shown by two accompanying indicators, 
    the Negative Directional Indicator (-DI) and the Positive Directional Indicator (+DI). 
    Therefore, ADX commonly includes three separate lines. These are used to help
    assess whether a trade should be taken long or short, or if a trade should be
    taken at all.

    *Designed by Welles Wilder for commodity daily charts, but can be used in other 
    markets or other timeframes.

    *The price is moving up when +DI is above -DI, and the price is moving down when 
    -DI is above +DI.

    *Crosses between +DI and -DI are potential trading signals as bears or bulls gain 
    the upper hand.

    *The trend has strength when ADX is above 25. The trend is weak or the price is 
    trendless when ADX is below 20, according to Wilder.

    *Non-trending doesn't mean the price isn't moving. It may not be, but the price 
    could also be making a trend change or is too volatile for a clear direction to 
    be present.
    '''
    @staticmethod
    def ADX(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))

        df.set_index('Date', inplace=True)
        main_df = pd.DataFrame(columns=['+DM', '-DM'])
        x = 0
        while x < len(df) - 1:
            today_high = df['High'][x + 1]
            yesterday_high = df['High'][x]
            today_low = df['Low'][x + 1]
            yesterday_low = df['Low'][x]

            upMove = today_high - yesterday_high
            downMove = yesterday_low - today_low

            if (upMove > downMove) & (upMove > 0):
                pos_DM = upMove

            else:
                pos_DM = 0

            if (downMove > upMove) & (downMove > 0):
                neg_DM = downMove

            else:
                neg_DM = 0

            main_df.loc[x] = [pos_DM, neg_DM]

            x += 1

        main_df['+DM'][0] = 'NAN'
        pos_DM_ma = main_df['+DM'].rolling(window=50, min_periods=0).mean()
        smooth_pos_DM = pos_DM_ma.rolling(window=50, min_periods=0).mean()
        neg_DM_ma = main_df['-DM'].rolling(window=50, min_periods=0).mean()
        smooth_neg_DM = neg_DM_ma.rolling(window=50, min_periods=0).mean()

        df = df[1:]
        true_range = df['High'] - df['Low']
        true_range = round(true_range, 0)

        TR_ma = true_range.rolling(window=50, min_periods=0).mean()
        TR_ma = pd.Series(TR_ma.values)
        TR_ma = TR_ma[1:]

        pos_DI = 100 * (smooth_pos_DM / TR_ma)
        neg_DI = 100 * (smooth_neg_DM / TR_ma)

        ADX = abs((pos_DI - neg_DI) / (pos_DI + neg_DI))
        ADX = ADX.rolling(window=10, min_periods=0).mean()
        ADX_full = 100 * ADX
        ADX_full = ADX_full.values

        # ADX = ADX_full[-1]
        # pos_DI = pos_DI[-1]
        # neg_DI = neg_DI[-1]

        ADX_full = pd.DataFrame(ADX_full)
        pos_DI = pd.DataFrame(pos_DI.values)
        neg_DI = pd.DataFrame(neg_DI.values)

        return ADX, pos_DI, neg_DI, ADX_full

    # Commodity Channel Index
    '''
    *High readings of 100 or above, for example, indicate the price is well 
    above the historic average and the trend has been strong to the upside.

    *Low readings below -100, for example, indicate the price is well below 
    the historic average and the trend has been strong to the downside.

    *Going from negative or near-zero readings to +100 can be used as a signal 
    to watch for an emerging uptrend.

    *Going from positive or near-zero readings to -100 may indicate an emerging downtrend.
    '''
    '''
    Developed by Donald Lambert, the Commodity Channel Index​ (CCI) is a 
    momentum-based oscillator used to help determine when an investment 
    vehicle is reaching a condition of being overbought or oversold. It is 
    also used to assess price trend direction and strength. This information 
    allows traders to determine if they want to enter or exit a trade, refrain 
    from taking a trade, or add to an existing position. In this way, the 
    indicator can be used to provide trade signals when it acts in a certain way.

    *The CCI measures the difference between the current price and the 
    historical average price.

    *When the CCI is above zero it indicates the price is above the historic 
    average. When CCI is below zero, the price is below the hisitoric average.

    *High readings of 100 or above, for example, indicate the price is well 
    above the historic average and the trend has been strong to the upside.

    *Low readings below -100, for example, indicate the price is well below 
    the historic average and the trend has been strong to the downside.

    *Going from negative or near-zero readings to +100 can be used as a signal 
    to watch for an emerging uptrend.

    *Going from positive or near-zero readings to -100 may indicate an emerging downtrend.

    *CCI is an unbounded indicator meaning it can go higher or lower indefinitely. 
    For this reason, overbought and oversold levels are typically determined for 
    each individual asset by looking at historical extreme CCI levels where the 
    price reversed from.
    '''

    @staticmethod
    def Comm_CI(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        i = 0
        main_df = pd.DataFrame(columns=['Typical Price'])
        while i < len(df):
            high = df['High'][i]
            low = df['Low'][i]
            close = df['Close'][i]

            main_df.loc[i] = (high + low + close) / 3
            i += 1

        main_df['20ma'] = main_df['Typical Price'].rolling(window=20, min_periods=0).mean()
        main_df['SMA'] = main_df['20ma'].rolling(window=20, min_periods=0).mean()
        main_df['20trix'] = main_df['SMA'].rolling(window=20, min_periods=0).mean()

        count = 0
        for value in main_df['Typical Price']:
            count += int(value)

        main_df['MD'] = count / len(df)
        main_df['CCI'] = (main_df['Typical Price'] - main_df['20trix']) / (.015 * main_df['MD'])
        CCI_full = pd.DataFrame(main_df['CCI'].values)
        CCI = main_df['CCI'].values[-1]

        return CCI, main_df['Typical Price'][len(main_df) - 1], CCI_full

    # Detrended Price Oscillator
    '''
    if DPO_yesterday < 0 and DPO_today >= 0:
        buy()
    elif DPO_yesterday > 0:
        if DPO_today < DPO_2daysago:
            sell()
        else:
            pass
    '''
    '''
    A detrended price oscillator is an oscillator that strips out price 
    trends in an effort to estimate the length of price cycles from peak 
    to peak or trough to trough. Unlike other oscillators, such as the 
    stochastic or moving average convergence divergence (MACD), the DPO 
    is not a momentum indicator. It highlights peaks and troughs in price, 
    which are used to estimate buy and sell points in line with the historical cycle.

    *The DPO is used for measuring the distance between peaks and troughs 
    in the price/indicator.

    *If troughs have historically been about two months apart, that may 
    help a trader make future decisions as they can locate the most recent 
    trough and determine that the next one may occur in about two months.

    *Traders can use the estimated future peaks as selling opportunities 
    or the estimated future troughs as buying opportunities.

    *The indicator is typically set to look back over 20 to 30 periods.
    '''

    @staticmethod
    def DPO(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        n = 50
        x = int(n / 2 + 1)
        df['nma'] = df['Adj Close'].rolling(window=x, min_periods=0).mean()
        DPO = (df['Adj Close'] - df['nma'])
        DPO = DPO.values

        today = DPO[-1]
        yesterday = DPO[-2]
        two_days = DPO[-3]

        DPO = pd.DataFrame(DPO)

        return today, yesterday, two_days, DPO

    # Moving Average Convergence/Divergence
    '''
    Moving Average Convergence Divergence (MACD) is a trend-following 
    momentum indicator that shows the relationship between two moving 
    averages of a security’s price. The MACD is calculated by subtracting 
    the 26-period Exponential Moving Average (EMA) from the 12-period EMA.


    The result of that calculation is the MACD line. A nine-day EMA of the 
    MACD called the "signal line," is then plotted on top of the MACD line, 
    which can function as a trigger for buy and sell signals. Traders may buy 
    the security when the MACD crosses above its signal line and sell
    - or short - the security when the MACD crosses below the signal line. 
    Moving Average Convergence Divergence (MACD) indicators can be interpreted 
    in several ways, but the more common methods are crossovers, divergences, 
    and rapid rises/falls.

    *MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.

    *MACD triggers technical signals when it crosses above (to buy) or below (to sell) 
    its signal line.

    *The speed of crossovers is also taken as a signal of a market is overbought or 
    oversold.

    *MACD helps investors understand whether the bullish or bearish movement in the 
    price is strengthening or weakening.
    '''

    @staticmethod
    def MACD(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        df['9ma'] = df['Adj Close'].rolling(window=9, min_periods=0).mean()
        df['12ma'] = df['Adj Close'].rolling(window=12, min_periods=0).mean()
        df['26ma'] = df['Adj Close'].rolling(window=26, min_periods=0).mean()

        AVG1 = df['9ma']
        AVG2 = df['12ma']
        AVG3 = df['26ma']

        close = df['Adj Close']

        MACD1 = AVG1 - close
        MACD2 = AVG2 - close
        MACD3 = AVG3 - close

        MACD1 = pd.DataFrame(MACD1.values)
        MACD2 = pd.DataFrame(MACD2.values)
        MACD3 = pd.DataFrame(MACD3.values)

        # MACD1 = MACD1.values[-1]
        # MACD2 = MACD2.values[-1]
        # MACD3 = MACD3.values[-1]

        return MACD1, MACD2, MACD3

    # Know Sure Thing Oscillator
    '''
    if KST >= high * 0.382:
        buy()
    if KST <= low * 0.382:
        sell()
    '''
    '''
    Know Sure Thing, or KST, is a momentum oscillator developed 
    by Martin Pring to make rate-of-change readings easier for 
    traders to interpret. In a 1992 Stocks and Commodities article, 
    Mr. Pring referred to the indicator as "Summed Rate of Change (KST)," 
    but the KST term stuck with technical analysts. The indicator is 
    relatively common among technical analysts preferring momentum 
    oscillators to make decisions.

    The Know Sure Thing indicator can be used in the same manner as many 
    other momentum oscillators, such as the well-known relative strength 
    index (RSI). Trading signals are generated when the KST crosses over 
    the signal line, but traders may also look for convergence and divergence 
    with the price, overbought or oversold conditions, or crossovers of the center line.

    Many traders combine the KST indicator with other forms of technical 
    analysis to maximize their odds of success. For example, traders may 
    look at other non-momentum indicators, chart patterns, or candlestick 
    patterns to help in their decision-making.
    '''

    @staticmethod
    def KST_Osc(stock):
        AVG1 = 10
        AVG2 = 10
        AVG3 = 10
        AVG4 = 15

        W1 = 1
        W2 = 2
        W3 = 3
        W4 = 4

        X1 = 10
        X2 = 15
        X3 = 20
        X4 = 30

        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        main_df = pd.DataFrame(columns=['Roc1', 'Roc2', 'Roc3', 'Roc4'])
        x = 0
        i = 1
        while i < (len(df) - 30):
            price = df['Adj Close']
            ROC1 = (price[-i] / price[-X1] - 1) * 100
            ROC2 = (price[-i] / price[-X2] - 1) * 100
            ROC3 = (price[-i] / price[-X3] - 1) * 100
            ROC4 = (price[-i] / price[-X4] - 1) * 100
            main_df.loc[x] = [ROC1, ROC2, ROC3, ROC4]
            x += 1
            i += 1
            X1 += 1
            X2 += 1
            X3 += 1
            X4 += 1

        ROC1ma = main_df['Roc1'].rolling(window=AVG1, min_periods=0).mean()
        ROC2ma = main_df['Roc2'].rolling(window=AVG2, min_periods=0).mean()
        ROC3ma = main_df['Roc3'].rolling(window=AVG3, min_periods=0).mean()
        ROC4ma = main_df['Roc4'].rolling(window=AVG4, min_periods=0).mean()
        KST = ROC1ma * W1 + ROC2ma * W2 + ROC3ma * W3 + ROC4ma * W4
        KST_full = pd.DataFrame(KST.values)
        KST_last = KST.values[-1]

        KST.values.sort()
        high = KST[len(KST) - 1]
        low = KST[0]

        return high, low, KST_last, KST_full

    # Mass Index
    '''
        if yesterday > 25:
            if yesterday - today > 0:
                if today <= 25:
                    sell()
                else:
                    pass
            else:
                pass
        elif yesterday < 25:
            if today - yesterday > 0:
                if today >= 24:
                    buy()
                else:
                    pass
            else:
                pass
    '''
    '''
    Dorsey hypothesized that, when the figure jumps above 27 – creating 
    a “bulge” – and then drops below 26.5, the stock is ready to change 
    course. An index of 27 represents a rather volatile stock, so some 
    traders set a lower baseline when determining the presence of a price bulge.
    '''
    '''
    Mass index is a form of technical analysis that examines 
    the range between high and low stock prices over a period 
    of time. Mass index, developed by Donald Dorsey in the early 
    1990s, suggests that a reversal of the current trend will likely 
    take place when the range widens beyond a certain point and then 
    contracts. 

    By analyzing the narrowing and widening of trading ranges, mass 
    index identifies potential reversals based on market patterns 
    that aren’t often considered by technical analysts largely focused 
    on singular price and volume movements. However, since the patterns 
    do not provide insight into the direction of the reversals, 
    technical analysts should combine the indicator’s readings with 
    directional indicators like the AD Line that specialize in 
    predicting those types of things.

    Dorsey hypothesized that, when the figure jumps above 27 – creating 
    a “bulge” – and then drops below 26.5, the stock is ready to change 
    course. An index of 27 represents a rather volatile stock, so some 
    traders set a lower baseline when determining the presence of a price bulge.

    While you can use a lot of other technical indicators, such as standard 
    deviation, to measure volatility, the reversal bulge function of the of 
    the mass index can offer you a unique perspective about the market 
    condition. You can also use mass index to trade trend continuations.

    The mass index indicator can be a great tool for short-term trading, 
    if a trader takes the time to change the sensitivity or periods 
    according to the historical volatility of the particular stock they are studying.
    '''

    @staticmethod
    def MI(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        df['Diff'] = df['High'] - df['Low']
        df['Diff_ma'] = df['Diff'].rolling(window=9, min_periods=0).mean()
        df['EMA'] = df['Diff_ma'].rolling(window=9, min_periods=0).mean()
        df['Sum'] = df['Diff_ma'] / df['EMA']
        Mass_Index = df['Sum'].rolling(window=25, min_periods=0).sum()
        Mass_Index = Mass_Index.values

        today = Mass_Index[-1]
        yesterday = Mass_Index[-2]

        Mass_Index = pd.DataFrame(Mass_Index)

        return today, yesterday, Mass_Index

    # Triple Exponential Average
    '''
    if yesterday < 0:
        if today - yesterday > 0:
            if today >= 0:
                buy()
            else:
                pass
        else:
            pass

    elif yesterday > 0:
        if yesterday - today > 0:
            if today <= 0:
                sell()
            else:
                pass
        else:
            pass      
    '''
    '''
    The triple exponential average (TRIX) indicator is an oscillator 
    used to identify oversold and overbought markets, and it can 
    also be used as a momentum indicator. Like many oscillators, TRIX 
    oscillates around a zero line. When it is used as an oscillator, a 
    positive value indicates an overbought market while a negative value 
    indicates an oversold market. When TRIX is used as a momentum indicator, 
    a positive value suggests momentum is increasing while a negative 
    value suggests momentum is decreasing. Many analysts believe that when 
    the TRIX crosses above the zero line it gives a buy signal, and when it 
    closes below the zero line, it gives a sell signal. Also, divergences 
    between price and TRIX can indicate significant turning points in the market.


    TRIX calculates a triple exponential moving average of the log of the price 
    input over the period of time specified by the length input for the current 
    bar. The current bar's value is subtracted by the previous bar's value. This 
    prevents cycles that are shorter than the period defined by length input from 
    being considered by the indicator.
    '''

    @staticmethod
    def Trix(stock):
        n = 10
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        df['ma1'] = df['Adj Close'].rolling(window=n, min_periods=0).mean()
        df['ma2'] = df['ma1'].rolling(window=n, min_periods=0).mean()
        df['ma3'] = df['ma2'].rolling(window=n, min_periods=0).mean()
        triple_EMA = df['ma3']
        triple_EMA = pd.DataFrame(triple_EMA.values)
        main_df = pd.DataFrame(columns=['Trix'])
        x = 0
        while x < len(df) - 1:
            trix = ((triple_EMA[0][x + 1] - triple_EMA[0][x]) / triple_EMA[0][x]) * 100
            main_df.loc[x] = [trix]
            x += 1

        trix = main_df['Trix'].values
        today = trix[-1]
        yesterday = trix[-2]

        trix = pd.DataFrame(trix)

        return today, yesterday, trix

    # Vortex Indicator
    '''
    if VMup_yesterday < VMdown_yesterday:
        if VMup_today > VMdown_today:
            buy()
        else:
            pass
    elif VMup_yesterday > VMdown_yesterday:
        if VMup_today < VMdown_today:
            sell()
        else:
            pass
    '''
    '''
    A vortex indicator (VI) is an indicator composed of two lines 
    - an uptrend line (VI+) and a downtrend line (VI-). These 
    lines are typically colored green and red respectively. A vortex 
    indicator is used to spot trend reversals and confirm current trends.

    The vortex indicator was first developed by Etienne Botes and 
    Douglas Siepman who introduced the concept in the January 2010 edition 
    of “Technical Analysis of Stocks & Commodities.” The vortex indicator 
    is based on two trendlines: VI+ and VI-.

    The vortex indicator is commonly used in conjunction with other reversal 
    trend patterns to help support a reversal signal. It is integrated into 
    most technical analysis software programs. VI+ and VI- are typically 
    graphed independently below a candlestick chart. The chart below provides 
    an example with lines that indicate changing trend signals on a candlestick chart.
    '''

    @staticmethod
    def VI(stock):
        n = 21
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        true_range = df['High'] - df['Low']
        df['True Range'] = round(true_range, 0)
        main_df = pd.DataFrame(columns=['+VM', '-VM'])

        a = 0
        b = 1
        while b < len(df):
            pos_VM = int(df['High'][a] - df['Low'][b])
            neg_VM = int(df['Low'][a] - df['High'][b])
            main_df.loc[a] = [pos_VM, -neg_VM]
            a += 1
            b += 1

        TR = df['True Range'][1:]
        TR = pd.Series(TR).values
        TR = pd.Series(TR)
        TR_sum = TR.rolling(window=n, min_periods=0).sum()
        pos_VM_sum = main_df['+VM'].rolling(window=n, min_periods=0).sum()
        neg_VM_sum = main_df['-VM'].rolling(window=n, min_periods=0).sum()
        VMn_up = pos_VM_sum / TR_sum
        VMn_up = VMn_up.values
        VMn_down = neg_VM_sum / TR_sum
        VMn_down = VMn_down.values

        VMup_today = VMn_up[-1]
        VMdown_today = VMn_down[-1]
        VMup_yesterday = VMn_up[-2]
        VMdown_yesterday = VMn_down[-2]

        VMn_up = pd.DataFrame(VMn_up)
        VMn_down = pd.DataFrame(VMn_down)

        return VMup_today, VMdown_today, VMup_yesterday, VMdown_yesterday, VMn_up, VMn_down


class Momentum:
    # Money Flow Index
    '''
    Below 20 = Oversold
    Above 80 = Overbought
    '''
    '''
    The Money Flow Index (MFI) is a technical oscillator that uses price 
    and volume for identifying overbought or oversold conditions in an 
    asset. It can also be used to spot divergences which warn of a trend 
    change in price. The oscillator moves between 0 and 100.


    Unlike conventional oscillators such as the Relative Strength Index (RSI), 
    the Money Flow Index incorporates both price and volume data, as opposed 
    to just price. For this reason, some analysts call MFI the volume-weighted RSI.

    *The indicator is typically calculated using 14 periods of data.

    *An MFI reading above 80 is considered overbought and an MFI reading
    below 20 is considered oversold.

    *Overbought and oversold doesn't necessarily mean the price will reverse, 
    only that the price (factoring for volume) is near the high or low of its 
    recent price range.

    *The creators of the index, Gene Quong and Avrum Soudack, recommended using 
    90 and 10 as overbought and oversold levels. These levels are rarely reached, 
    but when they are it often means the price could be due for a direction change.

    *A divergence between the indicator and price is noteworthy. For example, 
    if the indicator is rising while the price is falling or flat, the price could 
    start rising.
    '''

    @staticmethod
    def MFI(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        high = df['High']
        low = df['Low']
        close = df['Adj Close']
        typical_price = (high + low + close) / 3
        money_flow = typical_price * df['Volume']
        money_flow = pd.Series(money_flow.values)
        typical_price = pd.Series(typical_price.values)
        pos_MF = 0
        neg_MF = 0
        i = len(typical_price) - 1
        x = len(typical_price) - 2
        a = 0
        while i > 0:
            if typical_price[i] > typical_price[x]:
                pos_MF += money_flow

            elif typical_price[i] < typical_price[x]:
                neg_MF += money_flow

            else:
                pass
            i -= 1
            x -= 1

        x = 0
        money_ratio = pd.DataFrame(columns=['money ratio'])
        for value in pos_MF:
            money_ratio.loc[x] = value / neg_MF[x]
            x += 1

        MFI_full = 100 - (100 / (1 + money_ratio))
        MFI = str(MFI_full.values[-1])
        MFI = MFI.replace("[", "").replace("]", "")

        MFI_full = pd.DataFrame(MFI_full.values)

        return MFI, MFI_full

    # Relative Strength Index
    '''
    Below 30 = Oversold
    Above 70 = Overbought
    '''
    '''
    The relative strength index (RSI) is a momentum indicator that 
    measures the magnitude of recent price changes to evaluate overbought 
    or oversold conditions in the price of a stock or other asset. 
    The RSI is displayed as an oscillator (a line graph that moves between 
    two extremes) and can have a reading from 0 to 100. The indicator 
    was originally developed by J. Welles Wilder Jr. and introduced in 
    his seminal 1978 book, New Concepts in Technical Trading Systems.


    Traditional interpretation and usage of the RSI are that values of 70 
    or above indicate that a security is becoming overbought or overvalued 
    and may be primed for a trend reversal or corrective pullback in price. 
    An RSI reading of 30 or below indicates an oversold or undervalued condition.

    *The RSI is a popular momentum oscillator developed in 1978.


    *The RSI compares bullish and bearish price momentum plotted against the 
    graph of an asset's price.

    *Signals are considered overbought when the indicator is above 70% and oversold 
    when the indicator is below 30%.
    '''

    @staticmethod
    def RSI(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')
        close = df['Adj Close']
        x = 0
        main_df = pd.DataFrame(columns=['Up', 'Down'])
        while x < len(df) - 1:
            if close[x] > close[x + 1]:
                up = close[x] - close[x + 1]
                down = 0

            elif close[x] < close[x + 1]:
                up = 0
                down = close[x + 1] - close[x]

            else:
                up = 0
                down = 0

            main_df.loc[x] = [up, down]
            x += 1

        main_df['Up MA'] = main_df['Up'].rolling(window=14, min_periods=0).mean()
        main_df['Down MA'] = main_df['Down'].rolling(window=14, min_periods=0).mean()
        main_df['RSI'] = 100 - (100 / (1 + main_df['Up MA'] / main_df['Down MA']))
        RSI_full = main_df['RSI'].rolling(window=14, min_periods=0).mean()

        RSI = RSI_full.values[-1]

        RSI_full = pd.DataFrame(RSI_full.values)

        return RSI, RSI_full

    # Stochastic Oscillator
    '''
    Below 20 = Oversold
    Above 80 = Overbought
    '''
    '''
    A stochastic oscillator is a momentum indicator comparing 
    a particular closing price of a security to a range of its 
    prices over a certain period of time. The sensitivity of the 
    oscillator to market movements is reducible by adjusting that 
    time period or by taking a moving average of the result. It is 
    used to generate overbought and oversold trading signals, utilizing 
    a 0-100 bounded range of values.

    The stochastic oscillator is range-bound, meaning it is always 
    between 0 and 100. This makes it a useful indicator of overbought 
    and oversold conditions. Traditionally, readings over 80 are considered 
    in the overbought range, and readings under 20 are considered oversold. 
    However, these are not always indicative of impending reversal; 
    very strong trends can maintain overbought or oversold conditions for an 
    extended period. Instead, traders should look to changes in the 
    stochastic oscillator for clues about future trend shifts.

    *A stochastic oscillator is a popular technical indicator for generating 
    overbought and oversold signals.

    *It was developed in the 1950s and is still in wide use to this day.

    *Stochastic oscillators are sensitive to momentum rather than absolute price.
    '''

    @staticmethod
    def Stoch_Osc(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        i = len(df) - 1
        y = 0
        main_df = pd.DataFrame(columns=['K'])
        while i > len(df) - 4:
            x = i - 14
            price = df['Adj Close'][i]
            h14 = df['High'][x:i]
            l14 = df['Low'][x:i]

            values = []
            for h in h14:
                values.append(h)

            values.sort()
            h14 = values[-1]

            values = []
            for l in l14:
                values.append(l)

            values.sort()
            l14 = values[0]
            K = (price - l14) / (h14 - l14) * 100
            main_df.loc[y] = [K]
            i -= 1
            y += 1
        K = main_df['K'][0]
        D = main_df['K'].rolling(window=3, min_periods=0).mean()

        return K

    # True Strength
    '''
    if TSI_yesterday < 20 and TSI_today > 20:
        buy()
    elif TSI_yesterday > -20 and TSI_today < -20:
        sell()
    '''
    '''
    The true strength index is a technical momentum oscillator. 
    The indicator may be useful for determining overbought and 
    oversold conditions, indicating potential trend direction 
    changes via centerline or signal line crossovers, and warning 
    of trend weakness through divergence.

    The indicator is primarily used to identify overbought and 
    oversold conditions in an asset's price, spot divergence, 
    identify trend direction and changes via the centerline, and 
    highlight short-term price momentum with signal line crossovers.

    Since the TSI is based on price movements, oversold and overbought 
    levels will vary by the asset being traded. Some stocks may 
    reach +30 and -30 before tending to see price reversals, while 
    another stock may reverse near +20 and -20.

    *The TSI fluctuates between positive and negative territory. 
    Positive territory means the bulls are more in control of the asset. 
    Negative territory means the bears are more in control.

    *When the indicator divergences with price, that may signal the 
    price trend is weakening and may reverse.

    *A signal line can be applied to the TSI indicator. When the TSI 
    crosses above the signal line it can be used as a buy signal, and 
    when it crosses below, a sell signal. Such crossovers occur frequently 
    so use with caution.

    *Overbought and oversold levels will vary by the asset being traded.
    '''

    @staticmethod
    def TSI(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        i = len(df) - 1
        main_df = pd.DataFrame(columns=['M'])
        main_df2 = pd.DataFrame(columns=['M'])
        x = 0
        while i > 0:
            c0 = df['Adj Close'][i]
            c1 = df['Adj Close'][i - 1]
            m = c0 - c1
            main_df.loc[x] = m
            main_df2.loc[x] = abs(m)
            i -= 1
            x += 1
        main_df['Mma'] = main_df['M'].rolling(window=25, min_periods=0).mean()
        smoothMma = main_df['Mma'].rolling(window=13, min_periods=0).mean()
        main_df2['Mma'] = main_df2['M'].rolling(window=25, min_periods=0).mean()
        smoothMma2 = main_df2['Mma'].rolling(window=13, min_periods=0).mean()
        TSI = 100 * (smoothMma / smoothMma2)
        TSI = TSI.values

        today = TSI[-1]
        yesterday = TSI[-2]

        TSI = pd.DataFrame(TSI)

        return today, yesterday, TSI

    # Ultimate Oscilator
    '''
    Below 30 = Oversold
    Above 70 = Overbought
    '''
    '''
    The Ultimate Oscillator is a technical indicator that was developed 
    by Larry Williams in 1976 to measure the price momentum of an asset 
    across multiple timeframes. By using the weighted average of three 
    different timeframes the indicator has less volatility and fewer trade 
    signals compared to other oscillators that rely on a single timeframe. 
    Buy and sell signals are generated following divergences. The Ultimately 
    Oscillator generates fewer divergence signals than other oscillators due 
    to its multi-timeframe construction.

    The Ultimate Oscillator is a range-bound indicator with a value that 
    fluctuates between 0 and 100. Similar to the Relative Strength Index (RSI), 
    levels below 30 are deemed to be oversold, and levels above 70 are deemed 
    to be overbought. Trading signals are generated when the price moves in 
    the opposite direction as the indicator, and are based on a three-step method.



    *The indicator uses three timeframes in its calculation: seven, 14, and 28 periods.

    *The shorter timeframe has the most weight in the calculation, while the 
    longer timeframe has the least weight.

    *Buy signals occur when there is bullish divergence, the divergence low is 
    below 30 on the indicator, and the oscillator then rises above the divergence high.

    *A sell signal occurs when there is bearish divergence, the divergence 
    high is above 70, and the oscillator then falls below the divergence low.
    '''

    @staticmethod
    def Ult_Osc(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        thl_df = pd.DataFrame(columns=['BP', 'TH', 'TL'])
        x = 0
        i = len(df) - 1
        while i > 0:
            high = df['Low'][i]
            low = df['Low'][i]
            close = df['Adj Close'][i - 1]
            if low < close:
                TL = low
            elif close < low:
                TL = close
            else:
                TL = close

            BP = close - TL
            if high > close:
                TH = high
            elif close > high:
                TH = close
            else:
                TH = close

            thl_df.loc[x] = [BP, TH, TL]
            x += 1
            i -= 1

        thl_df['TR'] = thl_df['TH'] - thl_df['TL']
        thl_df['BP'][:7]
        BP_sum = 0
        for value in thl_df['BP'][:7]:
            BP_sum += value

        BP_sum2 = 0
        for value in thl_df['BP'][:14]:
            BP_sum2 += value

        BP_sum3 = 0
        for value in thl_df['BP'][:28]:
            BP_sum3 += value

        TR_sum = 0
        for value in thl_df['TR'][:7]:
            TR_sum += value

        TR_sum2 = 0
        for value in thl_df['TR'][:14]:
            TR_sum2 += value

        TR_sum3 = 0
        for value in thl_df['TR'][:28]:
            TR_sum3 += value

        sumBP7 = BP_sum
        sumBP14 = BP_sum2
        sumBP28 = BP_sum3

        sumTR7 = TR_sum
        sumTR14 = TR_sum2
        sumTR28 = TR_sum3

        avg7 = sumBP7 / sumTR7
        avg14 = sumBP14 / sumTR14
        avg28 = sumBP28 / sumTR28

        UltOsc = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)

        return UltOsc

    # Williams %R
    '''
        below -80 = oversold
        *conditions for buying oversold*
            %R reaches -100%.
            Five trading days pass since -100% was last reached
            %R fall below -95% or -85%.

        above -20 = overbought
        *conditions for selling overbought*
            %R reaches 0%.
            Five trading days pass since 0% was last reached
            %R rise above -5% or -15%.
    '''
    '''
    Williams %R, also known as the Williams Percent Range, is a type of 
    momentum indicator that moves between 0 and -100 and measures 
    overbought and oversold levels. The Williams %R may be used to find 
    entry and exit points in the market. The indicator is very similar 
    to the Stochastic oscillator and is used in the same way. It was 
    developed by Larry Williams and it compares a stock’s closing price to 
    the high-low range over a specific period, typically 14 days or periods.

    *Williams %R moves between zero and -100.

    *A reading above -20 is overbought.

    *A reading below -80 is oversold.

    *An overbought or oversold reading doesn't mean the price will reverse. 
    Overbought simply means the price is near the highs of its recent range, 
    and oversold means the price is in the lower end of its recent range.

    *Can be used to generate trade signals when the price and the indicator 
    move out of overbought or oversold territory.
    '''

    @staticmethod
    def R(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        n = 10
        i = len(df) - 1
        close = df['Adj Close'][i]
        highs = []
        lows = []
        while n > 0:
            high = df['High'][i - n]
            low = df['Low'][i - n]
            highs.append(high)
            lows.append(low)
            n -= 1

        highs = np.sort(highs)
        lows = np.sort(lows)
        highest = highs[-1]
        lowest = lows[0]
        R = (highest - close) / (highest - lowest) * -100

        return R


class Volume:
    # Accumulation/Distribution Index
    '''
    *A rising A/D line helps confirm a rising price trend.

    *A falling A/D line helps confirm a price downtrend.

    *If the price is rising but A/D is falling, it signals underlying weakness
    and a potential decline in price.
    '''
    '''
    Accumulation/distribution is a cumulative indicator that uses 
    volume and price to assess whether a stock is being accumulated 
    or distributed. The accumulation/distribution measure seeks to 
    identify divergences between the stock price and volume flow. 
    This provides insight into how strong a trend is. If the price 
    is rising but the indicator is falling this indicates that buying 
    or accumulation volume may not be enough to support the price rise 
    and a price decline could be forthcoming.

    *The accumulation/distribution line gauges supply and demand by looking 
    at where the price closed within the period's range, and then multiplying 
    that by volume.

    *The A/D indicator is cumulative, meaning one period's value is added or 
    subtracted from the last.

    *A rising A/D line helps confirm a rising price trend.

    *A falling A/D line helps confirm a price downtrend.

    *If the price is rising but A/D is falling, it signals underlying weakness 
    and a potential decline in price.
    '''

    @staticmethod
    def AccDistIndex(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        high = df['High']
        low = df['Low']
        close = df['Adj Close']
        volume = df['Volume']
        CLV = ((close - low) - (high - close)) / (high - low)
        accdist = volume * CLV
        main_df = pd.DataFrame(columns=['AccDist'])
        x = 0
        while x < len(accdist) - 1:
            AD_prev = accdist[x]
            AD = accdist[x + 1]

            AD = AD + AD_prev
            main_df.loc[x] = [AD]
            x += 1

        accdist = main_df['AccDist'].values
        today = accdist[-1]
        yesterday = accdist[-2]

        if today > yesterday:
            direction = "rise"
        elif today < yesterday:
            direction = "decline"

        accdist = pd.DataFrame(accdist)

        return today, yesterday, accdist

    # Negative Volume Index
    '''
    The Negative Volume Index is a technical indication line that 
    integrates volume and price to graphically show how price movements 
    are affected from down volume days.

    The Negative Volume Index (NVI) can be used in conjunction with the 
    Positive Volume Index (PVI). Both indexes were first developed by 
    Paul Dysart in the 1930s and gained popularity in the 1970s after 
    spotlighted in Norman Fosback’s book titled "Stock Market Logic."

    The Positive and Negative Volume Indexes are trendlines that can help 
    an investor to follow how a security’s price is changing with affects 
    from volume. Positive and Negative Volume Index trendlines are typically 
    available through most advanced technical charting software programs such 
    as MetaStock and EquityFeedWorkstation. Trendlines are usually added below 
    a candlestick pattern similar to the visualization of volume bar charts. 
    Theory around the Positive and Negative Volume Indexes suggests. Negative 
    Volume Index trendlines can potentially be the best trendline for following 
    mainstream, smart money movements typically characterized by institutional 
    investors. Positive Volume Index trendlines are usually more broadly 
    associated with high volume market trending affects which are known to be 
    more heavily influenced by both smart money and noise traders.
    '''

    @staticmethod
    def VolI(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        main_df = pd.DataFrame(columns=['+VI', '-VI'])
        x = 0
        while x < len(df) - 1:
            close_t = df['Adj Close'][x + 1]
            close_y = df['Adj Close'][x]
            volume_t = df['Volume'][x + 1]
            volume_y = df['Volume'][x]

            NVI = 0
            if volume_t < volume_y:
                NVI += (close_t - close_y) / close_y
            else:
                NVI += 0

            PVI = 0
            if volume_t > volume_y:
                PVI += (close_t - close_y) / close_y
            else:
                PVI += 0

            main_df.loc[x] = [PVI, NVI]
            x += 1

        pos_full = main_df['+VI'].values
        neg_full = main_df['-VI'].values
        pos_VI = pos_full[-1]
        neg_VI = neg_full[-1]

        pos_full = pd.DataFrame(pos_full)
        neg_full = pd.DataFrame(neg_full)

        return pos_VI, neg_VI, pos_full, neg_full

    # On-Balance Volume
    '''
    if (today - yesterday) > (today * 0.001):
        sell_all()
    '''
    '''
    On-balance volume (OBV) is a technical trading momentum indicator 
    that uses volume flow to predict changes in stock price. Joseph Granville 
    first developed the OBV metric in the 1963 book Granville's New Key to 
    Stock Market Profits.

    Granville believed that volume was the key force behind markets and designed 
    OBV to project when major moves in the markets would occur based on volume 
    changes. In his book, he described the predictions generated by OBV as 
    "a spring being wound tightly." He believed that when volume increases sharply 
    without a significant change in the stock's price, the price will eventually 
    jump upward or fall downward.

    *On-balance volume (OBV) is a technical indicator of momentum, using volume 
    changes to make price predictions.

    *OBV shows crowd sentiment that can predict a bullish or bearish outcome.

    *Comparing relative action between price bars and OBV generates more actionable 
    signals than the green or red volume histograms commonly found at the bottom 
    of price charts. 
    '''

    @staticmethod
    def OBV(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        OBV = 0
        main_df = pd.DataFrame(columns=['OBV'])
        x = 0
        while x < len(df) - 1:
            close_t = df['Adj Close'][x + 1]
            close_y = df['Adj Close'][x]
            volume = df['Volume'][x + 1]

            if close_t > close_y:
                OBV += volume
            elif close_t < close_y:
                OBV - + volume
            else:
                OBV += 0
            main_df.loc[x] = [OBV]
            x += 1

        OBV_full = main_df['OBV'].values
        today = OBV_full[-1]
        yesterday = OBV_full[-4]

        OBV_full = pd.DataFrame(OBV_full)

        return today, yesterday, OBV_full

    # Volume Price Trend
    '''
    if (today - yesterday) < (-today * 0.1):
        sell_all()
    '''
    '''
    The volume price trend (VPT) indicator helps determine a security’s 
    price direction and strength of price change. The indicator consists 
    of a cumulative volume line that adds or subtracts a multiple of the 
    percentage change in a share price’s trend and current volume, 
    depending upon the security’s upward or downward movements.

    The volume price trend indicator is used to determine the balance 
    between a security’s demand and supply. The percentage change in the 
    share price trend shows the relative supply or demand of a particular 
    security, while volume indicates the force behind the trend. The VPT 
    indicator is similar to the on-balance volume (OBV) indicator in that 
    it measures cumulative volume and provides traders with information 
    about a security’s money flow. Most charting software packages have the 
    VPT indicator included.
    '''

    @staticmethod
    def VPT(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        main_df = pd.DataFrame(columns=['VPT'])
        VPT = 0
        x = 0
        while x < len(df) - 1:
            close_t = df['Adj Close'][x + 1]
            close_y = df['Adj Close'][x]
            volume = df['Volume'][x + 1]
            VPT += volume * ((close_t - close_y) / close_y)
            main_df.loc[x] = [VPT]
            x += 1

        vpt = main_df['VPT'].values
        today = vpt[-1]
        yesterday = vpt[-3]

        vpt = pd.DataFrame(vpt)

        return today, yesterday, vpt


class Volatility:
    # Average True Range
    '''
    if ATR_today - ATR 25 days ago > ATR_today * .5:
        sell_all()
    '''
    '''
    The average true range (ATR) is a technical analysis indicator that 
    measures market volatility by decomposing the entire range of an 
    asset price for that period. Specifically, ATR is a measure of volatility 
    introduced by market technician J. Welles Wilder Jr. in his book, 
    "New Concepts in Technical Trading Systems."

    The true range indicator is taken as the greatest of the following: 
    current high less the current low; the absolute value of the current 
    high less the previous close; and the absolute value of the current 
    low less the previous close. The average true range is then a moving 
    average, generally using 14 days, of the true ranges.

    *Average true range (ATR) is a technical indicator measuring market volatility.

    *It is typically derived from the 14-day moving average of a series of true 
    range indicators.

    *It was originally developed for use in commodities markets but has since 
    been applied to all types of securities.
    '''

    @staticmethod
    def ATR(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        TR_df = pd.DataFrame(columns=['TR'])
        x = 0
        while x < len(df) - 1:
            high = df['High'][x]
            low = df['Low'][x]
            close = df['Adj Close'][x + 1]
            val1 = high - low
            val2 = abs(high - close)
            val3 = abs(low - close)
            val_list = np.sort([val1, val2, val3])
            TR = val_list[2]
            TR_df.loc[x] = [TR]
            x += 1

        ATR = TR_df['TR'].rolling(window=14, min_periods=0).mean()
        ATR = ATR.values

        ATR1 = ATR[-1]
        ATR_25 = ATR[-3]

        ATR = pd.DataFrame(ATR)

        return ATR1, ATR_25, ATR

    # Keltner Channel
    '''
    A Keltner Channel is a volatility based technical indicator composed 
    of three separate lines. The middle line is an exponential moving 
    average (EMA) of the price. Additional lines are placed above and below 
    the EMA. The upper band is typically set two times the Average True Range 
    (ATR) above the EMA, and lower band is typically set two times the ATR 
    below the EMA. The bands expand and contract as volatility (measured by ATR) 
    expands and contracts.

    Since most price action will be encompassed within the upper and lower 
    bands (the channel), moves outside the channel can signal trend changes 
    or an acceleration of the trend. The direction of the channel, such as 
    up, down, or sideways, can also aid in identifying the trend direction 
    of the asset.

    *The EMA of a Keltner Channel is typically 20 periods, although this can be 
    adjusted if desired.

    *The upper and lower bands are typically set two times the ATR above and 
    below the EMA, although the multiplier can also be adjusted based on 
    personal preference. A larger multiplier will result in a wider channel.

    *Price reaching the upper band is bullish, while reaching the lower band 
    is bearish. Reaching a band may indicate a continued trend in that direction.

    *The angle of the channel also aids in identifying the trend direction. When 
    the channel is angled upwards, the price is rising. When the channel is angled 
    downward the price is falling. If the channel is moving sideways, the price has 
    been as well.

    *The price may also oscillate between the upper and lower bands. When this 
    happens, the upper band is viewed as resistance and the lower band is support.
    '''

    @staticmethod
    def KC(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        high = df['High']
        low = df['Low']
        close = df['Adj Close']
        df['TP'] = (high + low + close) / 3
        KC_full = df['TP'].rolling(window=10, min_periods=0).mean()

        KC = KC_full.values[-1]

        KC_full = pd.DataFrame(KC_full.values)

        return KC, KC_full

    # Standard Deviation
    '''
    *A volatile stock has a high standard deviation, while the deviation of a 
    stable blue-chip stock is usually rather low.
    '''
    '''
    The standard deviation is a statistic that measures the dispersion 
    of a dataset relative to its mean and is calculated as the square 
    root of the variance. It is calculated as the square root of variance 
    by determining the variation between each data point relative to the 
    mean. If the data points are further from the mean, there is a higher 
    deviation within the data set; thus, the more spread out the data, the 
    higher the standard deviation.

    Standard deviation is a statistical measurement in finance that, when 
    applied to the annual rate of return of an investment, sheds light on 
    the historical volatility of that investment. The greater the standard 
    deviation of securities, the greater the variance between each price and 
    the mean, which shows a larger price range. For example, a volatile stock 
    has a high standard deviation, while the deviation of a stable blue-chip 
    stock is usually rather low.

    *Standard deviation measures the dispersion of a dataset relative to its mean.

    *A volatile stock has a high standard deviation, while the deviation of a 
    stable blue-chip stock is usually rather low.

    *As a downside, it calculates all uncertainty as risk, even when it’s in 
    the investor's favor—such as above average returns.
    '''

    @staticmethod
    def Std_Dev(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        STD = df['Adj Close'].rolling(window=50).std()

        # STD = STD.values[-1]
        STD = pd.DataFrame(STD.values)

        return STD


class Breadth:
    # McClellan Oscilator
    '''
    *The McClellan Oscillator formula can be applied to any stock exchange or
    group of stocks.

    *A reading above zero helps confirm a rise in the index, while readings
    below zero confirm a decline in the index.

    *When the index is rising but the oscillator is falling, that warns that
    the index could start declining too. When the index is falling and the
    oscillator is rising, that indicates the index could start rising soon.
    This is called divergence.

    *A significant change, such as moving 100 points or more, from a negative
    reading to a positive reading is called a breadth thrust. It may indicate
    a strong reversal from downtrend to uptrend is underway on the stock exchange.
    '''
    '''
    The McClellan Oscillator is a market breadth indicator that is 
    based on the difference between the number of advancing and declining 
    issues on a stock exchange, such as the New York Stock Exchange (NYSE) 
    or NASDAQ. The indicator is compared to stock market indexes related 
    to the exchange.

    The indicator is used to show strong shifts in sentiment in the indexes, 
    called breadth thrusts. It also helps in analyzing the strength of an 
    index trend via divergence or confirmation.

    *The McClellan Oscillator formula can be applied to any stock exchange or 
    group of stocks.

    *A reading above zero helps confirm a rise in the index, while readings 
    below zero confirm a decline in the index.

    *When the index is rising but the oscillator is falling, that warns that 
    the index could start declining too. When the index is falling and the 
    oscillator is rising, that indicates the index could start rising soon. 
    This is called divergence.

    *A significant change, such as moving 100 points or more, from a negative 
    reading to a positive reading is called a breadth thrust. It may indicate 
    a strong reversal from downtrend to uptrend is underway on the stock exchange.
    '''

    @staticmethod
    def McClellan_Osc(stock):
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock), index_col='Date')

        df['19ma'] = df['Adj Close'].rolling(window=19, min_periods=0).mean()
        df['39ma'] = df['Adj Close'].rolling(window=39, min_periods=0).mean()
        df['50ma'] = df['Adj Close'].rolling(window=50, min_periods=0).mean()
        df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
        df['200ma'] = df['Adj Close'].rolling(window=200, min_periods=0).mean()
        df.dropna(inplace=True)
        open_val = df['Open']
        close_val = df['Close']
        change = round(close_val - open_val, 2)
        pct_change = round((close_val - open_val) / open_val * 100, 2)
        df['Change'] = change
        df['% Change'] = pct_change

        mcclellan = df['19ma'] - df['39ma']

        # mcclellan = mcclellan.values[-1]

        mcclellan = pd.DataFrame(mcclellan.values)

        return mcclellan

    # Advance/Decline Ratio
    @staticmethod
    def ad_ratio(stock):
        advance = 0
        decline = 0
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        values = []
        for value in df['Adj Close'].tail(2):
            values.append(value)
        diff = values[1] - values[0]
        if diff > 0:
            advance += 1
        elif diff < 0:
            decline += 1

        ad_ratio = advance / decline
        nineteen_EMA = (advance - decline) * 0.01 + calculations(folder, stock)
        thirtynine_EMA = (advance - decline) * 0.05 + calculations(folder, stock)

        return ad_ratio.tail(1), nineteen_EMA.tail(1), thirtynine_EMA.tail(1)

    # Advance/Decline Volume
    @staticmethod
    def ad_volume(stock):
        advance = 0
        decline = 0
        df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
        values = []
        for value in df['Volume'].tail(2):
            values.append(value)
        diff = values[1] - values[0]
        if diff > 0:
            advance += 1
        elif diff < 0:
            decline += 1

        ad_volume = advance / decline

        return ad_volume


class Sentiment:
    @staticmethod
    def google_sentiment(stock):
        subjectivity = 0
        sentiment = 0
        url = 'https://www.google.com/search?q={0}&source=lnms&tbm=nws'.format(stock)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headline_results = soup.find_all('div', class_="BNeawe vvjwJb AP7Wnd")
        SUM = 0
        for result in headline_results:
            result = re.sub("<.*?>", "", str(result))
            blob = TextBlob(result)
            sentiment += blob.sentiment.polarity / len(headline_results)
            subjectivity += blob.sentiment.subjectivity / len(headline_results)
            true_value = (1 - subjectivity) * sentiment
            SUM += true_value
        sentiment = SUM / len(headline_results)
        return sentiment

    @staticmethod
    def yahoo_sentiment(stock):
        subjectivity = 0
        sentiment = 0
        SUM = 0
        url = 'https://www.finance.yahoo.com/quote/{0}?p={0}&.tsrc=fin-srch'.format(stock)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headline_results = soup.find_all('h3', class_='Mb(5px)')
        for result in headline_results:
            result = result.get_text()
            blob = TextBlob(result)
            sentiment += blob.sentiment.polarity / len(headline_results)
            subjectivity += blob.sentiment.subjectivity / len(headline_results)
            true_value = (1 - subjectivity) * sentiment
            SUM += true_value

        sentiment = SUM / len(headline_results)
        return sentiment

    @staticmethod
    def bing_sentiment(stock):
        subjectivity = 0
        sentiment = 0
        SUM = 0
        url = 'https://www.bing.com/news/search?q={}&FORM=HDRSC6'.format(stock)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headline_results = soup.find_all('a', class_='title')
        for result in headline_results:
            result = result.get_text()
            blob = TextBlob(result)
            sentiment += blob.sentiment.polarity / len(headline_results)
            subjectivity += blob.sentiment.subjectivity / len(headline_results)
            true_value = (1 - subjectivity) * sentiment
            SUM += true_value
        sentiment = SUM / len(headline_results)
        return sentiment

    @staticmethod
    def reuters_sentiment(stock):
        subjectivity = 0
        sentiment = 0
        SUM = 0
        url = 'https://www.reuters.com/companies/{}.O'.format(stock)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headline_results = soup.find_all('div', class_='item')
        for result in headline_results:
            result = result.find('a').get_text()
            blob = TextBlob(result)
            sentiment += blob.sentiment.polarity / len(headline_results)
            subjectivity += blob.sentiment.subjectivity / len(headline_results)
            true_value = (1 - subjectivity) * sentiment
            SUM += true_value
        sentiment = SUM / len(headline_results)
        return sentiment

    @staticmethod
    def forbes_sentiment(stock):
        subjectivity = 0
        sentiment = 0
        SUM = 0
        url = 'https://www.forbes.com/search/?q={}'.format(stock)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headline_results = soup.find_all('a', class_='stream-item__title')
        for result in headline_results:
            result = result.get_text()
            blob = TextBlob(result)
            sentiment += blob.sentiment.polarity / len(headline_results)
            subjectivity += blob.sentiment.subjectivity / len(headline_results)
            true_value = (1 - subjectivity) * sentiment
            SUM += true_value
        sentiment = SUM / len(headline_results)
        return sentiment

    @staticmethod
    def BI_sentiment(stock):
        subjectivity = 0
        sentiment = 0
        SUM = 0
        url = 'https://www.businessinsider.com/s?q={}'.format(stock)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headline_results = soup.find_all('div', class_='search-result')
        for result in headline_results:
            result = result.find('h3').get_text()
            blob = TextBlob(result)
            sentiment += blob.sentiment.polarity / len(headline_results)
            subjectivity += blob.sentiment.subjectivity / len(headline_results)
            true_value = (1 - subjectivity) * sentiment
            SUM += true_value
        sentiment = SUM / len(headline_results)
        return sentiment


def sp500Index():
    url = 'https://www.marketwatch.com/investing/index/spx'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    li = soup.find_all('li', class_="kv__item")

    values = {}
    i = 0
    for value in li:
        value = value.find_all('span')
        values[i] = value[0].get_text()
        i += 1
    open = values[0]
    day_range = values[1]
    fiftytwo_week_range = values[2]

    value_df = pd.DataFrame({
        'Open': [open],
        'Day Range': [day_range],
        '52 Week Range': [fiftytwo_week_range]
    })

    return value_df


# CBOE Volatility Index
def CVI():
    url = 'https://www.marketwatch.com/investing/index/vix'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    li = soup.find_all('li')

    values = {}
    i = 0
    for value in li:
        value = value.find_all('span')
        values[i] = value[0].get_text()
        i += 1
    open = values[0]
    day_range = values[1]
    fiftytwo_week_range = values[2]

    value_df = pd.DataFrame({
        'Open': [open],
        'Day Range': [day_range],
        '52 Week Range': [fiftytwo_week_range]
    })

    return value_df


# Misc
def current_price(stock):
    url = 'https://finance.yahoo.com/quote/' + stock + '?p=' + stock
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    data = soup.find_all('span', attrs={"data-reactid": "14"})

    current_price = data[0]
    current_price = current_price.get_text()
    current_price = current_price.replace(',', '').replace('"', '')

    return current_price


def adj_closes(stocks):
    for stock in stocks:
        df = pd.read_csv('sp500_joined_closes.csv')
        df.set_index('Date', inplace=True)
        stock_df = df[stock]
        print(stock)
        print(stock_df.head())


def hourly_prices(stocks):
    eastern = timezone('US/Eastern')
    eastern_time = dt.datetime.now(eastern)
    eastern_time = eastern_time.strftime('%H:%M:%S')
    time_split = eastern_time.split(":")
    date = dt.datetime.now()
    date = date.strftime("%Y-%m-%d")
    days = {}
    a = 0
    b = 0

    while True:
        hour_check = str(time_split[0])
        min_check = int(time_split[1])
        day = {}

        if int(hour_check[0]) == 0:
            if int(hour_check[1]) >= 9:
                if min_check > 30:
                    hour = {}
                    c = 0

                    for stock in stocks:
                        url = 'https://finance.yahoo.com/quote/' + stock + '?p=' + stock
                        response = requests.get(url)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        data = soup.find_all('span', attrs={"data-reactid": "14"})
                        current_price = data[0]
                        current_price = current_price.get_text()
                        hour[c] = {stock: current_price}
                        c += 1

                    day[b] = {now: hour}
                    b += 1
                    hour = {}
                    time.sleep(3600)

                else:
                    print("closed")
                    time.sleep(300)

        elif int(hour_check) >= 10:
            if int(hour_check) < 16:
                hour = {}
                c = 0

                for stock in stocks:
                    url = 'https://finance.yahoo.com/quote/' + stock + '?p=' + stock
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    data = soup.find_all('span', attrs={"data-reactid": "14"})
                    current_price = data[0]
                    current_price = current_price.get_text()
                    hour[c] = {stock: current_price}
                    c += 1
                day[b] = {now: hour}
                b += 1
                hour = {}

                if int(time_split[0]) == 15:
                    days[a] = {date: day}
                    a += 1
                    b = 0
                    time.sleep(57600)

                else:
                    time.sleep(3600)

            else:
                print("closed")
                time.sleep(3600)

        else:
            print("closed")
            time.sleep(3600)


def ticker(stocks):
    while True:
        for stock in stocks:
            url = 'https://finance.yahoo.com/quote/' + stock + '?p=' + stock
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            data = soup.find_all('span', attrs={"data-reactid": "14"})
            current_price = data[0]
            current_price = current_price.get_text()
            print(stock + ": " + current_price)
        time.sleep(10)


def compile_closes(stocks):
    main_df = pd.DataFrame()
    for stock in stocks:
        try:
            df = pd.read_csv('watch_dfs/{}.csv'.format(stock))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
        except:
            pass

    main_df.to_csv('watching_joined_closes.csv')


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        ticker = ticker.replace(".", "")
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def daily_ohlc(stock):
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()

    try:
        df = web.DataReader(stock, 'yahoo', start, end)
        df.to_csv('stock_dfs/{}.csv'.format(stock))

    except:
        pass


def SP_daily_ohlc(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        try:
            df = web.DataReader(ticker, 'yahoo', start, end)
            df = df.tail(2)
            print(ticker)
            df.to_csv('stock_dfs/{}.csv'.format(ticker), mode='a', header=False)

        except:
            pass


def portfolio(stock):
    df = pd.read_csv('Portfolio/{}.csv'.format(stock))
    total_investment = 0
    for value in df['Total']:
        total_investment += value
    total_holdings = 0

    for value in df['Qty']:
        total_holdings += value
    current_price(stock)
    loss_gain = (total_holdings * float(current_price(stock))) - total_investment

    return loss_gain


def buy(Qty, stock):
    date = dt.datetime.now()
    date = date.strftime('%Y-%m-%d')
    price = float(current_price(stock))
    cost = Qty * price

    main_df = pd.DataFrame({
        'Date': [date],
        'Qty': [Qty],
        'Price': [price],
        'Total': [cost]
    })

    df = pd.read_csv('Portfolio/Account.csv')
    action = "buy " + str(Qty) + " " + stock
    new_amount = df['Account'][len(df) - 1] - cost
    new_amount = pd.DataFrame({
        "Date": [date],
        "Action": [action],
        "Account": [new_amount]
    })

    new_amount.to_csv("Portfolio/Account.csv", mode='a', header=False)
    main_df.to_csv('Portfolio/{}.csv'.format(stock), mode='a', header=False)


def sell(Qty, stock):
    df = pd.read_csv('Portfolio/{}.csv'.format(stock))
    qty = df['Qty']
    if sum(qty) > Qty:
        date = dt.datetime.now()
        date = date.strftime('%Y-%m-%d')
        price = float(current_price(stock))
        cost = Qty * price

        main_df = pd.DataFrame({
            'Date': [date],
            'Qty': [-Qty],
            'Price': [price],
            'Total': [-cost]
        })

        df = pd.read_csv('Portfolio/Account.csv')
        action = "sell " + str(Qty) + " " + stock
        new_amount = df['Account'][len(df) - 1] + cost
        new_amount = pd.DataFrame({
            "Date": [date],
            "Action": [action],
            "Account": [new_amount]
        })

        new_amount.to_csv("Portfolio/Account.csv", mode='a', header=False)
        main_df.to_csv('Portfolio/{}.csv'.format(stock), mode='a', header=False)

    else:
        pass


def sell_all(stock):
    df = pd.read_csv('Portfolio/{}.csv'.format(stock))
    Qty = sum(df['Qty'])
    if Qty > 0:
        date = dt.datetime.now()
        date = date.strftime('%Y-%m-%d')
        price = float(current_price(stock))
        cost = Qty * price

        main_df = pd.DataFrame({
            'Date': [date],
            'Qty': [-Qty],
            'Price': [price],
            'Total': [-cost]
        })

        df = pd.read_csv('Portfolio/Account.csv')
        action = "sell all " + str(Qty) + " " + stock
        new_amount = df['Account'][len(df) - 1] + cost

        new_amount = pd.DataFrame({
            "Date": [date],
            "Action": [action],
            "Account": [new_amount]
        })

        new_amount.to_csv("Portfolio/Account.csv", mode='a', header=False)
        main_df.to_csv('Portfolio/{}.csv'.format(stock), mode='a', header=False)

    else:
        pass


def seasonal_buy(Qty, stock):
    date = dt.datetime.now()
    date = date.strftime('%Y-%m-%d')
    price = float(current_price(stock))
    cost = Qty * price

    main_df = pd.DataFrame({
        'Date': [date],
        'Qty': [Qty],
        'Price': [price],
        'Total': [cost]
    })

    df = pd.read_csv('Portfolio/Account.csv')
    action = stock + " season buy " + str(Qty)
    new_amount = df['Account'][len(df) - 1] - cost
    new_amount = pd.DataFrame({
        "Date": [date],
        "Action": [action],
        "Account": [new_amount]
    })

    new_amount.to_csv("Portfolio/Account.csv", mode='a', header=False)
    main_df.to_csv('Portfolio/{}.csv'.format(stock), mode='a', header=False)


def seasonal_sell(Qty, stock):
    df = pd.read_csv('Portfolio/{}.csv'.format(stock))
    qty = df['Qty']
    if sum(qty) > Qty:
        date = dt.datetime.now()
        date = date.strftime('%Y-%m-%d')
        price = float(current_price(stock))
        cost = Qty * price

        main_df = pd.DataFrame({
            'Date': [date],
            'Qty': [-Qty],
            'Price': [price],
            'Total': [-cost]
        })

        df = pd.read_csv('Portfolio/Account.csv')
        action = stock + " season sell " + str(Qty)
        new_amount = df['Account'][len(df) - 1] + cost
        new_amount = pd.DataFrame({
            "Date": [date],
            "Action": [action],
            "Account": [new_amount]
        })

        new_amount.to_csv("Portfolio/Account.csv", mode='a', header=False)
        main_df.to_csv('Portfolio/{}.csv'.format(stock), mode='a', header=False)

    else:
        pass


def score_switch(score):
    switcher = {
        -4: [0.0, 0.25],
        -3: [0.05, 0.2],
        -2: [0.1, 0.15],
        -1: [0.1, 0.1],
        0: [0.1, 0.1],
        1: [0.1, 0.1],
        2: [0.15, 0.1],
        3: [0.2, 0.05],
        4: [0.25, 0.0]
    }

    return switcher.get(score)


def linear_regression(stock):
    df = pd.read_csv('Calculations/{}.csv'.format(stock))
    df = df[1:].replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    predict = 'Close'

    for i in range(1, hm_days + 1):
        df['{}d'.format(i)] = ((df[predict].shift(-i) - df[predict]) / df[predict])
    df.fillna(0, inplace=True)

    def buy_sell_hold(*args):
        cols = [c for c in args]
        requirement = 0.005
        for col in cols:
            if col > requirement:
                return 1
            elif col < -requirement:
                return -1
            else:
                return 0

    df['target'] = list(map(buy_sell_hold, df['1d'], df['2d'], df['3d'], df['4d'], df['5d'], df['6d'], df['7d']))
    vals = df['target'].values
    str_vals = [str(i) for i in vals]
    print('Data spread: ', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[predict].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = np.array(df.drop([predict], 1))
    y = np.array(df[predict])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5, train_size=0.5,
                                                                                shuffle=False)

    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)

    accuracy = linear.score(X_test, y_test)
    accuracy = accuracy * 100
    # print(stock)
    # print("Accuracy: " + str(accuracy) + "%")
    # print("Coefficients: " + str(linear.coef_))
    # print("Intercept: " + str(linear.intercept_))

    predictions = linear.predict(X_test)

    for x in range(len(predictions)):
        if (predictions[x] - y_test[x - 1]) / y_test[x - 1] > 0.02:
            action = "Buy"
            # print("Buy")
        elif (predictions[x] - y_test[x - 1]) / y_test[x - 1] < -0.02:
            action = "Sell"
            # print("Sell")
        else:
            action = "Hold"
            # print("Hold")
        # print("Prediction:" + str(predictions[x]) + " Actual:" + str(y_test[x]))

    # print(action)
    # print("Buy:" + str(Buy) + " Sell:" + str(Sell) + " Hold:" + str(Hold))
    return accuracy, action
