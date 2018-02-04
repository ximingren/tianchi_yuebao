# coding:utf-8
import numpy as np
import pandas as pd


def read_data():
    global user_balance
    global user_profile
    global mfd_bank_shibor
    global mfd_yuebao
    user_balance=pd.read_csv(r"../data/user_balance_table.csv",parse_dates=['report_date'])
    user_profile=pd.read_csv(r"../data/user_profile_table.csv")
    mfd_bank_shibor=pd.read_csv(r"../data/mfd_bank_shibor.csv",parse_dates=['mfd_date'])
    mfd_yuebao=pd.read_csv(r"../data/mfd_day_share_interest.csv",parse_dates=['mfd_date'])


def init_data():
    global user_data
    global both_interest
    user_data = pd.merge(user_balance, user_profile, on='user_id')
    both_interest=pd.merge(mfd_yuebao,mfd_bank_shibor,on='mfd_date')
    both_interest=both_interest.rename(columns={'mfd_date':'report_date'})


def get_date_balance_table():
    global user_data
    global both_interest
    both_interest.index=both_interest['report_date']
    both_interest=both_interest.iloc[:,1:]
    date_balance=user_balance.groupby('report_date').sum().iloc[:,1:]
    date_balance_interest=pd.concat([date_balance,both_interest],join='inner',axis=1)
    date_balance_interest.to_csv(r'../analyed_data/date_user_balance.csv')


def get_user_table():
    data=pd.merge(user_data,both_interest,on='report_date')
    data.sort_values('user_id').to_csv(r'../analyed_data/all_data.csv',index=False)

def get_user_table_multiIndex():
    data = pd.merge(user_data, both_interest, on='report_date')
    multi_data=data.set_index(['user_id','report_date'])


def get_date_interest_table():
    global both_interest
    both_interest.index=both_interest['report_date']
    both_interest=both_interest.iloc[:,1:]
    both_interest.to_csv(r'../analyed_data/date_interest.csv')

if __name__=='__main__':
    pd.set_option('display.width', 3000)
    read_data()
    init_data()
    get_date_interest_table()
