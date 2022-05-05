import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as plt

all_symbols = ['TIF', 'GOOG', 'EXPE', 'NFLX', 'BABA', 'ZG', 'AMZN', 'BMY', 'TDG', 'NVDA', 'VRTX', 'ADBE', 'JPM', 'MSCI',
               'EQIX', 'LRCX', 'QCOM', 'AAPL', 'ANET', 'FISV', 'FTNT', 'SHOP']


def LinearRegression(stock):
    df = pd.read_csv('Calculations/{}.csv'.format(stock))
    df = df[1:].replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    predict = 'Close'
    '''
    for i in range(1, hm_days+1):
        df['{}d'.format(i)] = ((df[predict].shift(-i) - df[predict])/df[predict])
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
    '''

    X = np.array(df.drop([predict], 1))
    y = np.array(df[predict].shift(periods=1, fill_value=0))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.05, train_size=0.95,
                                                                                shuffle=False)

    linear = linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    accuracy = linear.score(X_test, y_test)
    accuracy = accuracy * 100
    predictions = linear.predict(X_test)
    df = pd.read_csv('stock_dfs/{}.csv'.format(stock))
    close = df['Adj Close'].tail(1)
    print("Close: " + str(close))
    for x in range(len(predictions)):
        print("Prediction: " + str(predictions[x]) + " Actual: " + str(y_test[x]))
    '''
    with open('linearmodel.pickle', 'wb') as f:
        pickle.dump(linear, f)

    pickle_in = open('linearmodel.pickle', 'rb')
    linear = pickle.load(pickle_in)
    accuracy = linear.score(X_test, y_test) * 100
    '''
    print("Accuracy: " + str(accuracy) + "%")
    linear = linear.fit(X, y)
    prediction = float(str(linear.predict([X[-1]])).replace("[", "").replace("]", ""))
    print(prediction)
    return prediction


'''
Accuracy = 0
for stock in all_symbols:
   LR = LinearRegression(stock)
   Accuracy += LR[0]
   Action = LR[1]
   print(Action + " " + stock)
print("Overall Accuracy: " + str(Accuracy/len(all_symbols)) + "%")
'''
LinearRegression('ADBE')




