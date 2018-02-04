import pandas as pd
from pyspark import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.regression import AFTSurvivalRegression, DecisionTreeRegressor
from pyspark.ml.linalg import Vectors
from pyspark.shell import sc, spark
from analyse.InitData import initData
"""
    决策树模型
"""
# 拟合模型
def train(train_data,param):
    dt=DecisionTreeRegressor(labelCol='tBalance',maxDepth=param)
    model=dt.fit(train_data)
    return model


# 评估模型
def evaluateModel(model,data):
    predict_data=model.transform(data)
    evaluator = RegressionEvaluator(labelCol='tBalance')
    return evaluator.evaluate(predict_data)

if __name__ =='__main__':
    dataObj=initData()
    dataObj.createBalanceVector()
    train_data,test_data=dataObj.balanceData.randomSplit([0.8,0.2])
    params=[2,5,10,15,20,25,30]
    models=[train(train_data,param) for param in params]
    with open(r'../evaluate/DecisionTreemodel.txt','a') as f:
        f.write(str([evaluateModel(model,test_data) for model in models]))
        f.write(str(models[1].numFeatures)+'numFeatures:\n')
        f.write(str(models[1].numNodes)+'numNodes:\n')
        f.write(str(models[1].depth)+'depth:\n')
    # predict_interest=read_data()
    # model.transform(predict_interest).select('prediction').toPandas().to_csv(r'../analyed_data/result2  .csv',index=False)
