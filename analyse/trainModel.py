import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, GeneralizedLinearRegression, IsotonicRegression, \
    DecisionTreeRegressor
from analyse.InitData import initData
"""
    GBT(梯度下降树模型)
    主要可调参数有：maxBins,maxDepth,maxIter
    GLM(广义线性回归模型)
    主要可调参数有：Family,link,maxIter
    DecisionTree(决策树模型)
    主要可调参数有：maxDepth
    RandomForestRegressor(随机森林模型)
    主要参数有numTrees,maxDepth,maxBins
    IsotonicRegression(保序回归模型)
    
"""


def train(algorithm,trainData,labelCol):
    gbt=algorithm(labelCol=labelCol,maxDepth=15,maxIter=30,featuresCol='features')
    model=gbt.fit(trainData)
    return model


def evaluateModel(model,data,labelCol):
    predict_data=model.transform(data)
    evaluator = RegressionEvaluator(labelCol=labelCol)
    return evaluator.evaluate(predict_data)


def join_result(result_tBalance,result_yBalance):
    result_tBalance = result_tBalance.select('report_date', 'prediction').toPandas()
    result_yBalance = result_yBalance.select('prediction').toPandas()
    result_tBalance.columns = ['date', 'tBalance']
    result_yBalance.columns = ['yBalance']
    result = result_tBalance.join(result_yBalance)
    result['date'] = result['date'].dt.strftime("%Y%m%d")
    return result


def scanBest():
    pass
    # 筛选出最佳参数
    # params=[2,5,10,15,20,25,30]
    # models=[train(algorithm,trainData,'total_redeem_amt',param) for param in params]
    # with open(r'../evaluate/GBTmodel','a') as f:
    #     f.write(str([evaluateModel(model,trainData,'total_redeem_amt') for model in models]))

    # 筛选出最佳模型
    # params=['GBTRegressor','RandomForestRegressor','GeneralizedLinearRegression','IsotonicRegreiion','DecisionTreeRegressor']
    # models=[train(algorithms[param],trainData,'total_purchase_amt') for param in params]
    # with open(r'../evaluate/GBTmodel', 'a') as f:
    #     f.write(str([evaluateModel(model,trainData,'total_purchase_amt') for model in models]))
    # 结果显示,GBT拟合效果较好

if __name__=='__main__':

    # 初始化数据
    dataObj=initData()
    # 形成训练集和测试集
    dataObj.createBalanceVector()
    # 形成预测利息的数据集
    dataObj.createInterestVector()
    trainData,testData=dataObj.balanceData.randomSplit([0.8,0.2])

    algorithms = {'GBTRegressor': GBTRegressor,
                  'RandomForestRegressor': RandomForestRegressor,
                  'GeneralizedLinearRegression': GeneralizedLinearRegression,
                  'IsotonicRegreiion': IsotonicRegression,
                  'DecisionTreeRegressor': DecisionTreeRegressor
                  }
    algorithm = algorithms['GBTRegressor']
    # 预测申购总量
    model=train(algorithm,dataObj.balanceData,'total_purchase_amt')
    result_tBalance=model.transform(dataObj.predict_interest)
    # 预测赎回总量
    model=train(algorithm,dataObj.balanceData,'total_redeem_amt')
    result_yBalance=model.transform(dataObj.predict_interest)
    # 将申购和赎回联合
    result=join_result(result_tBalance,result_yBalance)
    result.to_csv(r'../result/tc_comp_predict_table.csv',index=False)