import pandas as pd
import datetime
result1=pd.read_csv(r'../analyed_data/result.csv')
result2=pd.read_csv(r'../analyed_data/result2.csv')
date=pd.read_csv(r'../analyed_data/predict_interest.csv')
date.insert(column='tBalance',value=result1,loc=1)
date.insert(column='yBalance',value=result2,loc=2)
data=date.iloc[:,:3]
for key,value in data.iteritems():
    value=datetime.datetime.strptime(value, "%Y-%m-%d")
print(date)


