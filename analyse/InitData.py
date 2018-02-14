import pandas as pd
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.shell import spark

"""
    该类是用来初始化数据的
    balanceData属性就是已经有特征向量的训练数据
    predict_interest就是通过时间序列预测出来的利率数据
"""
class initData(object):

    def createBalanceVector(self):
        # 读取训练集
        self.rawdata=pd.read_csv(r'../analyed_data/data_user_balance.csv',index_col='report_date',parse_dates=['report_date'])
        # 获取要删除的列名
        # self.drop_name = list(self.rawdata.columns[2:-10])
        # # 将没用到的列删除
        # self.drop_data = self.rawdata.drop(columns=self.drop_name)

        self.data = spark.createDataFrame(self.rawdata)
        # 用VectorAssembler来将多个列合并成一个列
        # vecAssembler = VectorAssembler(
        #     inputCols=['mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W', 'Interest_2_W',
        #                'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y'],
        #     outputCol='features')
        vecAssembler = VectorAssembler(
            inputCols=['Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y'],
            outputCol='features')
        self.balanceData=vecAssembler.transform(self.data)
        # 利用PCA降维
        # pca=PCA(k=4,inputCol='features',outputCol='pca_features')
        # 对数据进行转化
        # balancePCA=pca.fit(self.FeatureData)
        # self.balanceData=balancePCA.transform(self.FeatureData)

    def createInterestVector(self):
        self.predict_interest=pd.read_csv(r'../analyed_data/fourth_predict_interest.csv',parse_dates=['report_date'])
        self.predict_interest = spark.createDataFrame(self.predict_interest)
        # vecAssembler = VectorAssembler(
        #     inputCols=['mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W', 'Interest_2_W',
        #                'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y'],
        #     outputCol='features')
        vecAssembler = VectorAssembler(
            inputCols=[ 'Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y'],
            outputCol='features')
        self.predict_interest = vecAssembler.transform(self.predict_interest)
        # pca = PCA(k=4, inputCol='features', outputCol='pca_features')
        # 对数据进行转化
        # InterestPCA= pca.fit(self.FeatureData)
