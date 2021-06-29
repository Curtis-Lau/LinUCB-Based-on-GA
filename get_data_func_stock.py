import pandas as pd
import math
import random
import copy

# pro = ts.pro_api(token='5e2da4e6322b3b893f1c10e43d001749300351741d736d247251d837')
def Utility(df_daily_rate, eta):      # 以夏普比率为例
    rf = 0.015
    # sharpe_ratio = (np.power(1+df_daily_rate.mean(), 250)-1-rf) / (df_daily_rate.std()*np.sqrt(250))
    sharpe_ratio = df_daily_rate.mean() - eta*df_daily_rate.var()
    return round(sharpe_ratio, 6)

def random_select(numbs_funds, candidates):
    all_index= list(range(0, len(candidates)))
    result = []
    while numbs_funds:
        index = math.floor(random.random() * len(all_index))    # 限定了index在0至len(candidates)之间
        result.append(all_index[index])
        del all_index[index]
        numbs_funds -= 1
    codes_sorted = [candidates[i] for i in result]
    return codes_sorted

def get_closedata(code, startdate, enddate):
    hs300_data = pd.read_csv('hs300_code_data_season.csv').set_index('trade_date')
    hs300_data.index = pd.to_datetime(hs300_data.index)
    closedata = hs300_data.loc[startdate:enddate, code]
    return closedata

def handle_data(data):
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    return data

def get_portfolios(s_time: 'YearMonthDay', e_time: 'YearMonthDay', numbs_portfolios, eta):
    hs300_code = pd.read_csv('hs300_code_season.csv').set_index('Unnamed: 0')
    code = list(hs300_code[e_time])
    code_new = []
    for i in code:
        code_new.append(i)
    closedata = get_closedata(code_new, startdate=s_time, enddate=e_time)
    closedata.sort_index(ascending=True, inplace=True)  # 颠倒数据
    na_sum = closedata.isna().sum().sort_values()
    del_stock = list(na_sum[na_sum > (len(closedata) * 0.1)].index)
    closedata.drop(del_stock, axis=1, inplace=True)
    closedata = handle_data(closedata)
    closedata_ret = closedata/closedata.shift(1) - 1
    df_utility = Utility(closedata_ret, eta)

    df_utility.sort_values(inplace=True)
    candidates = list(df_utility.iloc[-120:].index)
    # 产生portfolios
    portfolios_list = []
    for i in range(numbs_portfolios):
        portfolios_list.append(random_select(10, candidates))
    quitted_list = copy.deepcopy(candidates)
    port_codes = list(set([code for a in portfolios_list for code in a]))
    for c in port_codes:
        quitted_list.remove(c)
    return portfolios_list, quitted_list, candidates
