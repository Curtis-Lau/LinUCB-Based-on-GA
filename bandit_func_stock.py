import pandas as pd
import numpy as np
import random
import copy
from get_data_func_stock import get_closedata

def Standardize(utility):
    df = np.transpose(utility)
    df_standard = (df-df.min())/(df.max()-df.min())
    return np.transpose(df_standard)

def get_fund_nav(code_list, s_time, e_time):
    closedata = get_closedata(code_list, s_time, e_time)
    closedata.sort_index(ascending=True, inplace=True)
    dropped_code = list(closedata.isna().sum()[closedata.isna().sum()>len(closedata)*0.1].index)
    closedata = closedata.drop(dropped_code, axis=1)
    closedata.fillna(method='bfill', inplace=True)
    closedata.fillna(method='ffill', inplace=True)
    return closedata

def get_pre_day(date, n, date_all):
    the_date = pd.to_datetime(date) #指定当前日期格式
    the_date_index = date_all.index(the_date)
    pre_date = date_all[the_date_index-n]    #设置取多长的时间戳
    return pre_date

def utility_oneday(returns_df, starttime, endtime, code, eta):
    return_cluster = returns_df.loc[starttime:endtime, code]  # 获取所有基金时间范围内收益
    rf = 0.015
    # D = (np.power(1 + return_cluster.mean(), 250)-1-rf) / (return_cluster.std()*np.sqrt(250))
    D = return_cluster.mean()-eta*return_cluster.var()
    return D

def utility_series(timerange, returns_df, code, n, date_all, eta):
    eve_utility = pd.DataFrame(index=timerange, columns=code, dtype="float")
    for i in timerange:
        preday = get_pre_day(i, n, date_all)
        eve_utility.loc[i, :] = utility_oneday(returns_df, preday, i, code, eta)
    return Standardize(eve_utility)

def ridge_regression(xArr, yArr, lam=0.02):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse !")
        return
    weight = denom.I * (xMat.T*yMat)
    return weight

def upper_bound_probs(weight, D, x_new, beta=0.25):
    a_a = np.mat(np.dot(D.values.T, D.values)+np.eye(D.shape[1]))
    upper_bound_probs = np.dot(np.mat(x_new), weight) + beta*np.sqrt(np.dot(np.dot(x_new.values, a_a.I), x_new.values.T))
    return upper_bound_probs

def bandit_stock(delta, n, t, eta, weight_lst, cluster, HS300, date_all, quitted_codes, cand, beta, lam):
    w_list = copy.deepcopy(weight_lst)
    cluster_copy = copy.deepcopy(cluster)
    quitted_codes_copy = copy.deepcopy(quitted_codes)
    candidates = copy.deepcopy(cand)

    moneyPrt = 1000000
    moneyHS300 = 1000000
    backtesting = pd.DataFrame(columns=['date', 'Portfolio-capital', 'HS300-capital'])
    regret = 0
    timerange_n = date_all[t-n:t]
    timerange_m = date_all[t-n-delta:t-delta]
    bound_dict = {}

    pre_time = str(date_all[t-n-delta-n])[:10].replace('-', '')
    hold_time = str(date_all[t-1+delta])[:10].replace('-', '')

    code_list = list(set([code for portfolio in cluster_copy for code in portfolio]))
    nav_df_cluster = get_fund_nav(code_list, pre_time, hold_time)
    nav_df_quitted = get_fund_nav(quitted_codes_copy, pre_time, hold_time)
    nav_df_all = get_fund_nav(candidates, pre_time, hold_time)
    returns_df = (nav_df_all - nav_df_all.shift(1)) / nav_df_all.shift(1)

    # 将停牌的基金剔除出cluster_copy
    quitted_codes_used = list(nav_df_quitted.columns)
    for _ in code_list:
        if _ in quitted_codes_used:
            quitted_codes_used.remove(_)

    stopped_code = []
    if list(nav_df_cluster.columns) != code_list:
        for q in code_list:
            if q not in list(nav_df_cluster.columns):
                stopped_code.append(q)
        stopped_code = list(set(stopped_code))
        substitute_code_1 = np.array(quitted_codes_used)[
            random.sample(list(range(0, len(quitted_codes_used))), len(stopped_code))]
        for g in range(len(cluster_copy)):
            for p in range(len(stopped_code)):
                if stopped_code[p] in cluster_copy[g]:
                    idx_ = cluster_copy[g].index(stopped_code[p])
                    cluster_copy[g][idx_] = substitute_code_1[p]
        for cd1 in substitute_code_1:
            quitted_codes_used.remove(cd1)

    d_quitted = utility_series(timerange_m, returns_df, quitted_codes_used, n, date_all, eta)
    d_quitted_1 = d_quitted.isna().sum()
    nan_code_quitted = list(set(d_quitted_1[d_quitted_1 != 0].index.tolist()))

    a_quitted = utility_series(timerange_n, returns_df, quitted_codes_used, n, date_all, eta)
    a_quitted_1 = a_quitted.isna().sum()
    nan_code_quitted_2 = list(set(a_quitted_1[a_quitted_1 != 0].index.tolist()))
    nan_code_quitted.extend(nan_code_quitted_2)
    nan_code_quitted = list(set(nan_code_quitted))

    d_quitted_new = d_quitted.drop(nan_code_quitted, axis=1)
    a_quitted_new = a_quitted.drop(nan_code_quitted, axis=1)
    for cd2 in nan_code_quitted:
        quitted_codes_used.remove(cd2)

    for m in range(len(cluster_copy)):
        a = utility_series(timerange_n, returns_df, cluster_copy[m], n, date_all, eta)
        d = utility_series(timerange_m, returns_df, cluster_copy[m], n, date_all, eta)
        b1 = a.isna().sum()
        b2 = d.isna().sum()
        nan_code = b1[b1 != 0].index.tolist()
        nan_code.extend(b2[b2 != 0].index.tolist())
        nan_code = list(set(nan_code))
        substitute_code_3 = np.array(quitted_codes_used)[
            random.sample(list(range(0, len(quitted_codes_used))), len(nan_code))]
        new_a = a.drop(nan_code, axis=1)
        new_d = d.drop(nan_code, axis=1)

        for _code_ in range(len(nan_code)):
            idx = cluster_copy[m].index(nan_code[_code_])
            cluster_copy[m][idx] = substitute_code_3[_code_]
            w_list[m][idx] = 0.05
            new_a[substitute_code_3[_code_]] = a_quitted_new[substitute_code_3[_code_]]
            new_d[substitute_code_3[_code_]] = d_quitted_new[substitute_code_3[_code_]]
        for cd4 in substitute_code_3:
            quitted_codes_used.remove(cd4)
        copy_ = w_list[m].copy()
        for w_ in range(len(copy_)):
            if copy_[w_] != 0.05:
                w_list[m][w_] = copy_[w_] / (np.sum(copy_) - 0.05 * len(nan_code)) * (1 - 0.05 * len(nan_code))

        u = np.dot(new_a, w_list[m])
        w = ridge_regression(new_d, u, lam)  # 无常数项的岭回归

        f = 0
        m_list = []
        for i in w.tolist():
            if i[0] <= 0:
                f += 1
        if f >= len(w_list[m]) * 0.5:
            m_list.append(m)
            weight_list_m = [k[0] for k in w.tolist()]
            _max_ = np.max(weight_list_m)
            if _max_ <= 0:
                _min_ = 0.05
            else:
                _min_ = np.min([s for s in weight_list_m if s > 0]) + 0.000001
                if _min_ > 0.050001 or _min_ < 0.03:
                    _min_ = 0.05
            for o in range(len(weight_list_m)):
                if weight_list_m[o] <= 0:
                    weight_list_m[o] = _min_
            weight_list_m_copy = weight_list_m.copy()
            for o_ in range(len(weight_list_m_copy)):
                if weight_list_m_copy[o_] != _min_:
                    weight_list_m[o_] = weight_list_m_copy[o_]/(np.sum(weight_list_m_copy)-f*_min_)*(1-f*_min_)
            w = np.mat(weight_list_m).T
        else:
            # 将系数标准化
            w[w < 0] = 0  # 将出现的权重为负，强制设置为0
        w_list[m] = w / np.sum(w)  # 对权重进行标准化
        x = utility_series(timerange_n, returns_df, cluster_copy[m], n, date_all, eta).mean()
        bound_dict[m] = upper_bound_probs(w, new_d, x, beta)[0, 0]

    key_name = max(bound_dict, key=bound_dict.get)
    print('  选择的臂:', key_name)

    # 得到所选臂的实际收益
    backtesting = backtesting.append(
        [{'date': timerange_n[-1], 'Portfolio-capital': moneyPrt, 'HS300-capital': moneyHS300}], ignore_index=True)
    code = cluster_copy[key_name]
    amount = []
    # 获取最优portfolio的每个股票权重
    weight = w_list[key_name]
    # 将资金按权重分给对应股票
    div = moneyPrt * weight
    # 计算资金可购买的股票数量
    for j in range(len(code)):
        amount.append(int(div[j][0, 0] / nav_df_all[code[j]][timerange_n[-1]]))
    # 买股票后的剩余资金
    cash_funds = moneyPrt - (np.mat(amount) * (np.mat(nav_df_all.loc[timerange_n[-1], code]).T))[0, 0]

    # 可以购买指数的数量
    amount_index = int(moneyHS300 / HS300.loc[timerange_n[-1]])
    # 买指数后的剩余资金
    cash_index = moneyHS300 - (HS300.loc[timerange_n[-1]] * amount_index)

    holding_time = date_all[t:t + delta]
    for d in holding_time:
        moneyPrt = (cash_funds + (np.mat(amount) * (np.mat(nav_df_all.loc[d, code].values).T)))[0, 0]
        moneyHS300 = cash_index + (amount_index * HS300.loc[d])
        backtesting = backtesting.append([{'date': d, 'Portfolio-capital': moneyPrt, 'HS300-capital': moneyHS300[0]}], ignore_index=True)
        # 计算后悔度
        reward = []
        for j in range(len(cluster_copy)):
            w = w_list[j]
            reward.append((np.dot(w.T, returns_df.loc[d, cluster_copy[j]]))[0, 0])
        regret += np.max(reward) - reward[key_name]
    return regret, w_list, cluster_copy