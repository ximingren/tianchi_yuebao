import pandas as pd
import matplotlib.pyplot as plt
# with open('./a','r') as f:
#     for t in f.readlines():
#         if t.split():
#             print(t.strip())

pred1=pd.read_csv(r'../analyed_data/third_predict_interest.csv',index_col='report_date')
pred2=pd.read_csv(r'../analyed_data/predict_interest.csv',index_col='report_date')
columns1=pred1.columns
columns2=pred2.columns
for column in columns1:
    pred1[column].plot()
    pred2[column].plot()
    plt.title(column)
    plt.show()
