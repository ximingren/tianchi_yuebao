from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from analyse.InitData import initData
"""
    GBT(梯度下降树模型)
    主要可调参数有：maxBins,maxDepth,maxIter
"""

def train(trainData,param):
    gbt=GBTRegressor(labelCol='tBalance',maxIter=param)
    model=gbt.fit(trainData)
    return model

def evaluateModel(model,data):
    predict_data=model.transform(data)
    evaluator = RegressionEvaluator(labelCol='tBalance')
    return evaluator.evaluate(predict_data)

if __name__=='__main__':
    dataObj=initData()
    dataObj.createBalanceVector()
    trainData,testData=dataObj.balanceData.randomSplit([0.8,0.2])
    params = [ 25, 30,35,40,45,50]
    models = [train(trainData, param) for param in params]
    with open(r'../evaluate/GBTmodel.txt', 'a') as f:
        f.write('Evaluating the maxIter:')
        f.write(str([evaluateModel(model, testData) for model in models]))
