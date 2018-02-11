import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np


pd.set_option('display.width',2000)

# 绘制相关系数的条形图
def draw_corr(name):
    pit = corr[name]
    x = np.arange(len(pit.index))
    t = plt.figure(1, (25, 7))
    plt.bar(x, pit.values)
    plt.title('corr_'+name)
    plt.ylabel('corr')
    plt.xticks([])
    for t in range(len(pit.index)):
        plt.annotate(pit.index[t], xytext=(-0.5 + t, pit.values[t]), xy=(0, 0.9))
    plt.savefig(r'../piture/corr/'+name)
    plt.show()

if __name__=='__main__':

    data=pd.read_csv(r'../analyed_data/data_user_balance.csv',index_col='report_date')
    # 相关系数
    corr=data.corr()
    # 消费总量相关系数
    for name in data.columns:
        draw_corr(name)
