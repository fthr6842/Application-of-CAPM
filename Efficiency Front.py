import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
stocks=['1101.tw', '1102.tw', '1103.tw', '1104.tw', '1108.tw', '1109.tw', '1110.tw']# '1101B.tw']#股票代碼
expected_return_i=[]#平均回報
stocks_return=np.nan#個股回報(日)
for i in range(0, len(stocks)):#擷取資料
    if i==0:
        n=yf.download(stocks[i], start='2012-01-01', end='2013-01-01')
        stocks_return=round(n['Close'].diff()/n['Close'], 7)
        n5=len(n['Close'])
    else:
        n=yf.download(stocks[i], start='2012-01-01', end='2013-01-01')
        na=round(n['Close'].diff()/n['Close'], 7)
        stocks_return=pd.concat([stocks_return, na], axis=1)
stocks_return.columns=stocks#增加各股回報名稱
stocks_return=stocks_return.dropna()#去除第一位空值
for i in range(0, len(stocks)):#平均回報(日)
    na=stocks_return.iloc[:, i].mean()
    expected_return_i.append(na)
expected_return_i=pd.Series(expected_return_i)#平均回報(年)
covariance_table=stocks_return.cov()*(n5-1)#取變異數、共變異數(年)

#相同權重下的投資組合回報與投資組合風險
weight_avg=np.array([(1/len(stocks))]*len(stocks))#個股權重相同
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
print(max(portfolio_risk_i), min(portfolio_risk_i))
for i in range(0, len(portfolio_risk_i)):
    portfolio_risk_i[i]=round(portfolio_risk_i[i], 2)
    portfolio_return_i[i]=round(portfolio_return_i[i], 7)
list_x, list_y=[], []
for i in range(0, len(portfolio_risk_i)):
    if portfolio_risk_i[i] not in list_x:
        list_x.append(portfolio_risk_i[i])
        list_y.append(portfolio_return_i[i])
    else:
        na=list_x.index(portfolio_risk_i[i])
        if list_y[na] >=  portfolio_return_i[i]:
            pass
        else:
            list_y[na]=portfolio_return_i[i]
df=pd.DataFrame({'x':list_x, 'y':list_y})
df=df.sort_values(by='x')
list_x, list_y=df.iloc[:, 0].to_list(), df.iloc[:, 1].to_list()
plt.plot(list_x, list_y, color='b')
plt.title('efficient frontier')
plt.xlabel('risk')
plt.ylabel('return')
plt.grid()
print('平均權重報酬', portfolio_return_avg, '平均權重風險', portfolio_risk_avg)
print('亂數點相關係數', np.corrcoef(portfolio_risk_i, portfolio_return_i)[0][1])
plt.show()
