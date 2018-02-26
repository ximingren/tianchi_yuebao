from datetime import datetime
from math import sqrt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense, concatenate
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 随机种子，进行随机数操作的时候，为了着一个过程的可重复性，设置一个随机种子，让每次随机结果相同。
np.random.seed(7)


# look_back就是预测下一步所需要的time steps
def create_dataset(dataset,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:(i+look_back)]
        # 特征是长度为look_back的序列
        dataX.append(a)
        # 结果
        dataY.append(dataset[i+look_back])
    return np.array(dataX),np.array(dataY)
# def series_to_supervised(data, n_in=1, n_out=1,dropnan=True,columns=columns):
#     n_vars=1 if type(data) is list else data.shape[1]
#     df= pd.DataFrame(data)
#     cols,names=list(),list()
#     for i in range(n_in,0,-1):
#         cols.append(df.shift(i))
#         names +=[('%s(t-%d)' %(columns[j],i)) for j in range(n_vars)]
#     for i in range(0,n_out):
#         cols.append(df.shift(-i))
#         names +=[('%s(t)'%(columns[j])) for j in range(n_vars)]
#     agg= pd.concat(cols, axis=1)
#     agg.columns=names
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

date_interest=pd.read_csv(r'../analyed_data/date_interest.csv',index_col='report_date',parse_dates=['report_date'])
columns=date_interest.columns
result=[]
# 循环十次，有十个特征

for fea_var in range(10):
    # 第一个需要预测的特征
    values=date_interest.values[:,fea_var]
    values= values[:, np.newaxis]
    values=values.astype('float32')
    scaler=MinMaxScaler(feature_range=(0,1))
    # 将时间序列问题转化为监督学习问题
    scaled=scaler.fit_transform(values)
    # 初始化loob_back
    look_back=31
    data_x,data_y=create_dataset(scaled,look_back)
    count=len(data_x)
    # 分割成训练集和测试集
    train_X,train_y=data_x[:int(0.8*count)],data_y[:int(0.8*count)]
    test_X,test_y=data_x[int(0.8*count):],data_y[int(0.8*count):]
    # # 将数据重构成LSTM与其的3D格式，即[样本，时间步长，特征]
    train_X=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    model=Sequential()
    # # 定义具有50个神经元的LSTM和用于预测污染的输出层中的一个神经元。
    model.add(LSTM(50,input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dense(1))
    # # optimizer指定其优化器adam，loss指定误差规则，即平均绝对误差。
    model.compile(loss='mae',optimizer='adam')
    # # epochs是训练的次数,validation_data是验证的数据，batch_size是Number of samples per gradient update.
    history=model.fit(train_X,train_y,epochs=50,batch_size=72,validation_data=(test_X,test_y),verbose=2,shuffle=False)
    # # 根据train_x,test_X来预测
    # train_yhat=model.predict(train_X)
    # train_yhat=scaler.inverse_transform(train_yhat)
    # test_yhat=model.predict(test_X)
    # test_yhat=scaler.inverse_transform(test_yhat)
    pre_data=data_x[-1]
    pre_data=pre_data[np.newaxis]
    pre_data=pre_data.reshape((pre_data.shape[0],1,pre_data.shape[1]))
    var_result=[]
    for i in range(30):
        predict=model.predict(pre_data)
        pre_data=np.delete(pre_data,0)
        pre_data=np.append(pre_data,predict)
        pre_data = pre_data[np.newaxis]
        pre_data = pre_data.reshape((pre_data.shape[0], 1, pre_data.shape[1]))
        predict=scaler.inverse_transform(predict)
        var_result.append(predict)
    result.append(pd.Series(np.concatenate(var_result)[:,0]))
result=pd.concat(result,axis=1)
result.columns=columns
result.index=pd.date_range(start=datetime.strptime('20140901', '%Y%m%d'), periods=30)
result.to_csv(r'../analyed_data/sixth_predict_interest.csv')
# # 把数据复原为【样本，特征】的格式
# test_X=test_X.reshape(test_X.shape[0],test_X.shape[2])
# inv_yhat=pd.np.concatenate((yhat,test_X[:,1:]),axis=1)
# inv_yhat=scaler.inverse_transform(inv_yhat)
# # # 预测的值
# inv_yhat=inv_yhat[:,0]
# inv_y=scaler.inverse_transform(test_X)
# inv_y=inv_y[:,-1]
# rmse=sqrt(mean_squared_error(test_y,test_yhat))
# print(rmse)