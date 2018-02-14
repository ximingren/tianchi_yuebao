import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
# 观察法
def draw_trend(timeSeries,column):
    f=plt.figure(facecolor='white')
    # rol_mean=diff3_timeSeries.rolling(window=size).mean()
    # rol_weighted_mean=pd.ewma(diff3_timeSeries,span=size)
    timeSeries.plot(color='blue',label='original')
    diff1_timeSeries=timeSeries.diff(1)
    diff2_timeSeries=timeSeries.diff(2)
    diff3_timeSeries=timeSeries.diff(3)
    diff1_timeSeries.plot(color='red',label='diff_1')
    diff2_timeSeries.plot(color='green',label='diff_2')
    diff3_timeSeries.plot(color='yellow',label='diff_3')
    # rol_mean.plot(color='red',label='rolling mean')
    # rol_weighted_mean.plot(color='black',label='weight mean')
    plt.legend(loc='best')
    plt.title(column)
    plt.savefig(r'../piture/timeSeries/'+column)
    plt.show()
# 有时序图可以看出，一阶差分的数据就已经足够平稳了

# 单位根检验
def analyse_stationary(data,column):
    #原数据，未差分的
    # stationiary = (data)
    # 差分后的
    diff1_data=(data.diff(1)).iloc[1:]
    raw_result=adfuller(data)
    diff1_result = adfuller(diff1_data)
    print(column)
    print("ADF  Statistic:%f" % diff1_result[0])
    print("ADF  Statistic:%f" % raw_result[0])
    print(diff1_result[1])
    print(raw_result[1])
    print("Critical Values:")
    for key, value in diff1_result[4].items():
        print("\t %s: %.3f" % (key, value))
    for key, value in raw_result[4].items():
        print("\t %s: %.3f" % (key, value))
# Interest_0_N  和 Interest_1_W Interest_2_W Interest_1_M 这几个的p值小于0.05，说明原假设是假的，序列是平稳的。
# 而其他的则是不平稳的
# 一阶差分后的数据都是平稳的

# 绘制平稳序列的acf和pacf图
# lags是多少阶的意思
def draw_acf_pacf(data,column):
    plt.subplot(211)
    plot_acf(data,ax=plt.gca(),lags=40)
    plt.subplot(212)
    plot_pacf(data,ax=plt.gca(),lags=40)
    plt.show()
#经过对acf和pacf图的分析，大多数的属性都是拖尾，一阶相关性，少数是有截尾的

def func1(data,order):
    global model
    model = ARMA(data, order=order).fit(disp=0)
    predict_sunspots = model.forecast(steps=35)[0]
    return predict_sunspots


def func2(data,order):
    global model
    # 第一行的数据,差分后恢复需要这个
    first = data.iloc[0]
    data = data.diff(1).iloc[1:]
    model = ARMA(data, order=order).fit(disp=0)
    predict_sunspots = model.forecast(steps=35)[0]
    result = np.append(np.array([first]), predict_sunspots).cumsum()
    return result[1:]

# 以值为参数,预测可靠多
def predict(data,column):
    # python中没有switch,用字典代替
    # 首先进行模型拟合
    # 然后获得预测值,返回预测值
    # 如果是进行差分的则需要回滚
    orders={'mfd_daily_yield':[(7,3),func1],
               'Interest_O_N':[(1,5),func1],
               'Interest_1_W':[(5,4),func1],
               'Interest_2_W':[(8,2),func1],
               'Interest_1_M':[(3,4),func1],
               'Interest_3_M':[(3,6),func1],
               'mfd_7daily_yield':[(4,3),func2],
               'Interest_6_M':[(3,2),func2],
               'Interest_9_M':[(1,0),func2],
               'Interest_1_Y':[(1,0),func2],
               }
    order=orders[column][0]
    func=orders[column][1]
    result=func(data,order)
    for t in range(len(result)):
        result[t]=("%.3f" % result[t])
    return result

    # first = data.iloc[0]
    # data = data.diff(1).iloc[1:]
    # model = ARMA(data, order=(1,1)).fit(disp=0)
    # # predict_sunspots = model.forecast(steps=35)[0]
    # # result = np.append(np.array([first]), predict_sunspots).cumsum()
    # # for t in range(len(result)):
    # #     result[t]=("%.3f" % result[t])
    # #
    # # return result[1:]

    # forecast返回3个值的列表,第一个是预测的值，第二个是预测值的标准差，第三个是预测系数

    # 模型的残差
    # resid=model.resid

    # D-W检验，结果均接近2，表明不存在一阶自相关性
    # print(durbin_watson(model.resid.values))

    # Ljung-Box检验用来检验时间序列是否存在滞后相关的一种统计检验
    # 有点问题
    # r,q,p=sm.tsa.stattools.acf(resid.values.squeeze(),qstart=True)
    # data=np.c_[range(1,41),r[1:],q,p]
    # table=pd.DataFrame(data,columns['lag','AC','Q','Prob(>Q)'])
    # print(table.set_index('lag'))

    # 绘制出预测值与实际值的线图
    # predict_sunspots=model.predict()
    # # 去除没有预测的
    # data=data[predict_sunspots.index]
    # fig,ax=plt.subplots(figsize=(12,8))
    # ax=data.plot(ax=ax)
    # predict_sunspots.plot(ax=ax,color='red')
    # plt.title(column)
    # plt.show()

    # QQ图是用来分析是否为正太分布的.
    # qqplot(model.resid,line='q',fit=True)
    # plt.show()
# 不存在自相关性

def evaluate_arima_model(X,arima_order):
    # 初始化
    X=X.astype('float32')
    # 分割成
    train_size=int(len(X)*0.9)
    train,test=X[0:train_size],X[train_size:]
    # 训练过的历史记录
    history=[x for x in train]
    # predictions=list()
    # 将测试集的一个一个的加入都训练集中
        # 拟合模型
    model=ARIMA(history,order=arima_order)
    model_fit=model.fit(disp=0)
    aic=model_fit.aic
    # yhat=model_fit.forecast()[0]
    # predictions.append(yhat)
    # history.append(test[i])
    # mse=mean_squared_error(test,predictions)
    # rmse= np.sqrt(mse)
    return aic

def evaluate_models(dataset,p_values,d_values,q_values,column):
    # 转化格式，初始化
    dataset=dataset.astype('float32')
    best_score,best_cfg=float('inf'),None
    # 进行网格搜索
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order=(p,d,q)
                try:
                    # 评估模型，返回的是mean_squared_error
                    aic=evaluate_arima_model(dataset,order)
                    if aic<best_score:
                        best_score,best_cfg=aic,order
                        print(column+"ARIMA%s MSE=%f"%(order,aic))
                except:
                    print('error')
                    continue
    print('Best'+column+"ARIMA%s MSE=%.3f" % (best_cfg, best_score))

if __name__=='__main__':
    # 读取数据
    date_interest=pd.read_csv(r'../analyed_data/date_interest.csv',index_col='report_date',parse_dates=['report_date'])
    # 获取列名
    columns=date_interest.columns
    # 创建结果
    result=pd.DataFrame()
    # 忽略警告
    warnings.filterwarnings("ignore")
    # evaluate_models(date_interest.iloc[:,2].values,range(0,10),range(0,2),range(0,10),'Interest_0_N')
    for i in range(len(columns)):
        column=columns[i]
        # 预测数值
        result[column]=predict(date_interest.iloc[:,i],column)
    # 设置索引
    result.index=pd.date_range(start=datetime.strptime('20140829', '%Y%m%d'), periods=35)
    result.to_csv(r'../analyed_data/fourth_predict_interest.csv')