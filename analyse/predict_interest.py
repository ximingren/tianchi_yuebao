import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

# 观察法
def draw_trend(timeSeries,column):
    f=plt.figure(facecolor='white')
    # rol_mean=timeSeries.rolling(window=size).mean()
    # rol_weighted_mean=pd.ewma(timeSeries,span=size)
    timeSeries=timeSeries.diff(3)
    timeSeries.plot(color='blue',label='original')
    # rol_mean.plot(color='red',label='rolling mean')
    # rol_weighted_mean.plot(color='black',label='weight mean')
    plt.legend(loc='best')
    plt.title(column)
    plt.show()

# 单位根检验
def analyse_stationary(data,column):
    #原数据，未差分的
    # stationiary = (data)
    # 差分后的
    global stationiary
    stationiary=(data.diff(1)).iloc[1:]
    result = adfuller(stationiary)
    print(column)
    print("ADF  Statistic:%f" % result[0])
    print("p-value : %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t %s: %.3f" % (key, value))
# Interest_0_N  和 Interest_1_W Interest_2_W Interest_1_M 这几个的p值小于0.05，说明原假设是假的，序列是平稳的。
# 而其他的则是不平稳的
# 一阶差分后的数据都是平稳的

# 绘制acf和pacf图
# lags是多少阶的意思
def draw_acf_pacf(data,column):
    plt.subplot(211)
    plot_acf(data,ax=plt.gca(),lags=40)
    plt.subplot(212)
    plot_pacf(data,ax=plt.gca(),lags=40)
    plt.savefig(r'../piture/acf_pacf/'+column)
#经过对acf和pacf图的分析，大多数的属性都是拖尾，一阶相关性，少数是有截尾的

def predict(data,column,first):
    global model
    model=ARMA(data,order=(2,2)).fit(disp=0)
    # forecast返回3个值的列表,第一个是预测的值，第二个是预测值的标准差，第三个是预测系数
    predict_sunspots=model.forecast(steps=32)[0]
    first=np.append(np.array([first]),predict_sunspots)
    result=pd.Series(first,index=pd.date_range(start=datetime.strptime('20140829','%Y%m%d'),periods=33),name=column).cumsum()
    return result
    # 绘制出预测值与实际值的线图
    # predict_sunspots=model.predict()
    # # 去除没有预测的
    # data=data[predict_sunspots.index]
    # fig,ax=plt.subplots(figsize=(12,8))
    # ax=data.plot(ax=ax)
    # predict_sunspots.plot(ax=ax,color='red')
    # plt.show()

    # QQ图是用来分析是否为正太分布的.
    # qqplot(model.resid,line='q',fit=True)
    # plt.show()
# 不存在自相关性


if __name__=='__main__':
    date_interest=pd.read_csv(r'../analyed_data/date_interest.csv',index_col='report_date',parse_dates=['report_date'])
    columns=date_interest.columns
    result=pd.DataFrame()
    for i in range(len(columns)):
        draw_trend(date_interest,column)
        # 进行一阶差分
        data=date_interest.diff(1)
        column=columns[i]
        result[column]=(predict(data.iloc[1:,i],column,date_interest.iloc[0,i]))
    result.to_csv(r'../analyed_data/predict_interest.csv')