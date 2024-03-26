import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
stocks=['1101.tw', '1102.tw', '1103.tw', '1104.tw', '1108.tw', '1109.tw', '1110.tw']#, '1101B.tw']#股票代碼
specific_weight=[0.1, 0.5, 0.2, 0.2, 0, 0, 0]#指定權重
expected_return_i=[]#平均回報
stocks_return=np.nan#個股回報(日)
for i in range(0, len(stocks)):#擷取資料
    if i==0:
        n=yf.download(stocks[i], start='2019-01-01', end='2020-01-01')
        stocks_return=round(n['Close'].diff()/n['Close'], 7)
        n5=len(n['Close'])
    else:
        n=yf.download(stocks[i], start='2019-01-01', end='2020-01-01')
        na=round(n['Close'].diff()/n['Close'], 7)
        stocks_return=pd.concat([stocks_return, na], axis=1)
stocks_return.columns=stocks#增加各股回報名稱
stocks_return=stocks_return.dropna()#去除第一位空值
for i in range(0, len(stocks)):#平均回報(日)
    na=stocks_return.iloc[:, i].mean()
    expected_return_i.append(na)
expected_return_i=pd.Series(expected_return_i)#平均回報(年)
covariance_table=stocks_return.cov()*(n5-1)#取變異數、共變異數(年)

#指定權重的投資組合回報與投資組合風險
weight_avg=np.array(specific_weight)#指定個股權重
portfolio_return_avg=sum(expected_return_i*weight_avg)#portfolio  return
portfolio_risk_avg=np.sqrt(reduce(np.dot, [weight_avg, covariance_table,  weight_avg.T]))#portfolio  risk

#模擬可行的投資組合在回報-風險圖上分布
portfolio_risk_i, portfolio_return_i=[],  []#用於記錄各點
N=100000#樣本數目
for _ in range(0, N):
    weight=np.random.rand(len(stocks))
    weight=weight/sum(weight)#使各股權重合=1
    return_i=sum(expected_return_i*weight)
    risk_i=np.sqrt(reduce(np.dot, [weight, covariance_table, weight.T]))
    portfolio_risk_i.append(risk_i)
    portfolio_return_i.append(return_i)
#繪圖
fig = plt.figure(figsize = (10,6))
fig.suptitle('random  simulation', fontsize=20, fontweight='bold')
pic=fig.add_subplot()
pic.plot(portfolio_risk_i, portfolio_return_i, 'o', color='b')
pic.plot(portfolio_risk_avg, portfolio_return_avg,  'o', color='r')
pic.set_title(f'N={N}', fontsize=20)
pic.grid()
fig.show()
print('指定權重報酬', portfolio_return_avg, '指定權重風險', portfolio_risk_avg)
#print('亂數點相關係數', np.corrcoef(portfolio_risk_i, portfolio_return_i)[0][1])
