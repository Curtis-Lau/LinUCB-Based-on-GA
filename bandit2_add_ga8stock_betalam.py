import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_data_func_stock import get_portfolios
from get_data_func_stock import get_closedata
import random
from bandit_func_stock import bandit_stock
from ga_func import translateDNA
from ga_func import crossover_and_mutation
from ga_func import get_fitness
from ga_func import select
import time
import copy

# pro = ts.pro_api(token='5e2da4e6322b3b893f1c10e43d001749300351741d736d247251d837')

s_date = '20160101'
e_date = '20210331'

HS300 = pd.read_csv('hs300_close_data.csv').set_index('trade_date')
HS300.index = pd.to_datetime(HS300.index)
HS300.sort_index(inplace=True)

date_all = list(HS300.index)

def Standardize(utility):
    df = np.transpose(utility)
    df_standard = (df-df.min())/(df.max()-df.min())
    return np.transpose(df_standard)

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

# 计算指定日期的前n天的时间戳，求时间序列效用时用到
def get_pre_day(date, n):
    the_date = pd.to_datetime(date) #指定当前日期格式
    the_date_index = date_all.index(the_date)
    pre_date = date_all[the_date_index-n]    #设置取多长的时间戳
    return pre_date

# 获取组合内股票的一个时间段的收益月均值，与收益月方差，定义厌恶系数
# returns_df收益率，starttime效用窗口初始天，endtime效用窗口末尾天，code组合内基金代码，eta为厌恶系数
# 目的是为了获取X
def utility_oneday(returns_df, starttime, endtime, code, eta):
    return_cluster = returns_df.loc[starttime:endtime, code]  # 获取所有基金时间范围内收益
    rf = 0.015
    # len_date = len(return_cluster)
    # D = (np.power(1 + return_cluster.mean(), 250) - 1 - rf) / (return_cluster.std() * np.sqrt(250))
    D = return_cluster.mean()-eta*return_cluster.var()  # 收益减方差
    return D
# 获取时间范围内每日的向前推delta时间窗口（包括工作日与非工作日）的收益率月均和方差
# 目的是为了获取D
def utility_series(timerange, returns_df, code, n, eta):
    eve_utility = pd.DataFrame(index=timerange, columns=code, dtype="float")
    for i in timerange:
        preday = get_pre_day(i, n)
        # nav_df_m = get_fund_nav(code, str(preday)[:10].replace('-', ''), str(i)[:10].replace('-', ''))
        # returns_df_m = (nav_df_m - nav_df_m.shift(1)) / nav_df_m.shift(1)
        eve_utility.loc[i, :] = utility_oneday(returns_df, preday, i, code, eta)
    return Standardize(eve_utility)

# 计算置信上界
def upper_bound_probs(weight, D, x_new, beta=1):
    a_a = np.mat(np.dot(D.values.T, D.values)+np.eye(D.shape[1]))
    upper_bound_probs = np.dot(np.mat(x_new), weight) + beta*np.sqrt(np.dot(np.dot(x_new.values, a_a.I), x_new.values.T)) #compute_delta(X) for i in range(N)]  #.mul()乘法a * b
    return upper_bound_probs

# 根据基金code获取基金数据
def get_fund_nav(code_list, s_time, e_time):
    closedata = get_closedata(code_list, s_time, e_time)
    closedata.sort_index(ascending=True, inplace=True)
    dropped_code = list(closedata.isna().sum()[closedata.isna().sum()>len(closedata)*0.1].index)
    closedata = closedata.drop(dropped_code, axis=1)
    closedata.fillna(method='bfill', inplace=True)
    closedata.fillna(method='ffill', inplace=True)
    return closedata

# GA params
# 迭代次数
N_GENERATIONS = 10
# beta取值范围(upper bound probs奖励参数)
X_BOUND = [1, 5]
# lam取值范围(rigde regression惩罚参数)
Y_BOUND = [0.01, 1]
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.8
DNA_SIZE = 6
POP_SIZE = 40

# 初始资金
moneyPrt = 1000000
moneyHS300 = 1000000
# backtesting放回测数据
backtesting = pd.DataFrame(columns=['date', 'Portfolio-capital', 'HS300-capital'])
cum_regret = pd.DataFrame(columns=['date', 'cum_regret'])
calc_cluster = {}
weight_list = [1] * 10
cluster_params_copy = [None, None, None]
cluster_copy = None
quitted_codes_copy = None
change_time_list = []

# bandit params
t = 244 + 1
# date_all[120]='2016-07-01'   date_all[59]='2016-04-01'   date_all[244]='2017-01-03'   date_all[488]='2018-01-02'
t_raw = t
# delta：portfolio投资天数
delta = 12
# n：当前日期之前的n天
n = 18
eta = 0.1
eta_times = 5
numround = 1218    # 488:'2018-01-02'    731:'2019-01-02'    975:'2020-01-02'    1217:'2020-12-31
regret = 0
count_times = 1
# 初始化beta，lam
beta = 1
lam = 0.01
beta_list = []
lam_list = []

# 每半年进行一次大调仓（非bandit调仓），根据s_time至e_time期间（60天）股票的夏普比率调仓
def get_cluster_date(time, n=60):
    date_all_df = pd.DataFrame({'date': date_all, 'numb': range(len(date_all))}).set_index('date')
    time_ = pd.to_datetime(time)
    if time_.month in [1, 2, 3]:
        idx = date_all_df.loc[str(time_.year)+str('-')+'01', 'numb'][0]
        new_time = date_all[idx]
    if time_.month in [4, 5, 6]:
        idx = date_all_df.loc[str(time_.year)+str('-')+'04', 'numb'][0]
        new_time = date_all[idx]
    if time_.month in [7, 8, 9]:
        idx = date_all_df.loc[str(time_.year)+str('-')+'07', 'numb'][0]
        new_time = date_all[idx]
    if time_.month in [10, 11, 12]:
        idx = date_all_df.loc[str(time_.year)+str('-')+'10', 'numb'][0]
        new_time = date_all[idx]
    c_time_idx = date_all.index(new_time)
    e_time = str(date_all[c_time_idx-1])[:10].replace('-', '')
    s_time = str(date_all[c_time_idx-1-n])[:10].replace('-', '')
    return [s_time, e_time]

start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
while True:
    print('-----------开始第{}轮计算-----------'.format(count_times))
    print('delta:', delta)
    print('    n:', n)
    print('  eta:', eta)
    print(' beta:', beta)
    print('  lam:', lam)
    # 设定m和n期限
    # timerange_n：当前日期的前n天。该区间用来算X
    timerange_n = date_all[t-n:t]
    # 该区间用来算D
    timerange_m = date_all[t-n-delta:t-delta]
    # 存放每个arm的upper bound probs
    bound_dict = {}

    s_time = str(date_all[t-n])[:10].replace('-', '')
    e_time = str(timerange_n[-2])[:10].replace('-', '')
    # change_time：bandit调仓日期
    change_time = str(timerange_n[-1])[:10].replace('-', '')
    # 因为要计算timerange_m所有时间前n天的夏普比率，所以pre_time是timerange_m最早那天往前n天
    pre_time = str(date_all[t-n-delta-n])[:10].replace('-', '')
    # 当前portfolio投资天数（delta天）最后一天
    hold_time = str(date_all[t-1+delta])[:10].replace('-', '')
    ## hold_time的作用：为了防止投资期间股票缺失引起的复杂调整，先提前删除那些股票。选取portfolios和bandit运算时，不考虑hold_time。
    print('bandit调仓:', change_time)
    change_time_list.append(str(timerange_n[-1])[:10])

    # 当cluster参数调整时，也是cluster发生变化的时候，即大调仓那天。
    cluster_params = get_cluster_date(change_time)
    print('cluster_params:', cluster_params)
    # 当时间点不是大调仓，冻结cluster
    if cluster_params_copy[1] != cluster_params[1]:
        # cluster是选出的portfolio组合，每个portfolio10只股票
        # quitted_codes是最好的100个股票中没有被cluster选中的股票，用于填补后期portfolio中被踢出的股票
        # candidates是这100只股票
        cluster, quitted_codes, candidates = get_portfolios(cluster_params[0], cluster_params[1], 10, eta)
        cluster_params_copy = copy.deepcopy(cluster_params)
        cluster_copy = copy.deepcopy(cluster)
        quitted_codes_copy = copy.deepcopy(quitted_codes)
        cluster_change_idx = date_all.index(pd.to_datetime(change_time))
        print('cluster_change_idx:', cluster_change_idx)

    print('开始计算:', 'code_list')
    code_list = list(set([code for portfolio in cluster_copy for code in portfolio]))
    nav_df_cluster = get_fund_nav(code_list, pre_time, hold_time)
    nav_df_quitted = get_fund_nav(quitted_codes_copy, pre_time, hold_time)
    nav_df_all = get_fund_nav(candidates, pre_time, hold_time)

    # 将空缺值多的股票剔除出cluster_copy、quitter_codes_copy，因为get_fund_nav函数会对原始数据进行清洗，会剔除空缺值较多的股票
    print('开始计算:', 'quitted_codes_used')
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
        # 随机从quitted_codes_used中选取一只基金，填补空缺位置
        substitute_code_1 = np.array(quitted_codes_used)[random.sample(list(range(0, len(quitted_codes_used))), len(stopped_code))]
        print('替用基金1:', substitute_code_1)
        for g in range(len(cluster_copy)):
            for p in range(len(stopped_code)):
                if stopped_code[p] in cluster_copy[g]:
                    idx_ = cluster_copy[g].index(stopped_code[p])
                    cluster_copy[g][idx_] = substitute_code_1[p]
        # 将抽出的基金从quitted_codes_used中删除
        for cd1 in substitute_code_1:
            quitted_codes_used.remove(cd1)

    returns_df = (nav_df_all - nav_df_all.shift(1)) / nav_df_all.shift(1)

    print('开始计算:', 'quitted_x_d_u')
    # 提前计算好候补股票的各种数据，如：X，D，a(a是用来算U的)
    # x_quitted = utility_oneday(returns_df, timerange_n[0], timerange_n[-1], quitted_codes_used)
    x_quitted = utility_series(timerange_n, returns_df, quitted_codes_used, n, eta).mean()

    d_quitted = utility_series(timerange_m, returns_df, quitted_codes_used, n, eta)
    d_quitted_1 = d_quitted.isna().sum()
    nan_code_quitted = list(set(d_quitted_1[d_quitted_1 != 0].index.tolist()))

    # a_quitted = utility_series(timerange_n, returns_df, quitted_codes_used, n)
    a_quitted = returns_df.loc[timerange_n[0]:timerange_n[-1], quitted_codes_used]
    a_quitted_1 = a_quitted.isna().sum()
    nan_code_quitted_2 = list(set(a_quitted_1[a_quitted_1 != 0].index.tolist()))

    nan_code_quitted.extend(nan_code_quitted_2)
    nan_code_quitted = list(set(nan_code_quitted))

    x_quitted_new = x_quitted.drop(nan_code_quitted)
    d_quitted_new = d_quitted.drop(nan_code_quitted, axis=1)
    a_quitted_new = a_quitted.drop(nan_code_quitted, axis=1)
    for cd2 in nan_code_quitted:
        quitted_codes_used.remove(cd2)

    print('开始运行:', 'bandit')
    print('t:', t)
    print('-'*20)

    # 将这些数据copy，目的是为了给ga用
    weight_list_ga = copy.deepcopy(weight_list)
    cluster_ga = copy.deepcopy(cluster_copy)
    quitted_codes_ga = copy.deepcopy(quitted_codes_copy)
    candidates_ga = copy.deepcopy(candidates)

    if t == cluster_change_idx+1:
        print('*-*-*-*大调仓期*-*-*-*')
        print('-'*20)
        key_times = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for m in range(len(cluster_copy)):
            print('开始循环第{}个portfolio'.format(m))
            # x = utility_oneday(returns_df, timerange_n[0], timerange_n[-1], cluster_copy[m])
            x = utility_series(timerange_n, returns_df, cluster_copy[m], n, eta).mean()
            # x = returns_df.loc[timerange_n[0]:timerange_n[-1], cluster_copy[m]]
            d = utility_series(timerange_m, returns_df, cluster_copy[m], n, eta)

            d1 = d.isna().sum()
            nan_code_1 = list(set(x[x.isna()].index))
            nan_code = list(set(d1[d1 != 0].index.tolist()))
            nan_code.extend(nan_code_1)
            print('      计算:', 'd')
            new_d = d.drop(nan_code, axis=1)
            print('      计算:', 'x')
            new_x = x.drop(nan_code)

            substitute_code_2 = np.array(quitted_codes_used)[random.sample(list(range(0, len(quitted_codes_used))), len(nan_code))]
            print(' 替用基金2:', substitute_code_2)
            for idx1 in range(len(nan_code)):
                idx2 = cluster_copy[m].index(nan_code[idx1])
                cluster_copy[m][idx2] = substitute_code_2[idx1]
                new_d[substitute_code_2[idx1]] = d_quitted_new[substitute_code_2[idx1]]
                new_x[substitute_code_2[idx1]] = x_quitted_new[substitute_code_2[idx1]]
            for cd3 in substitute_code_2:
                quitted_codes_used.remove(cd3)

            print('      更新:', 'w')
            w = np.mat([1 / len(cluster_copy[m])] * len(cluster_copy[m])).T
            weight_list[m] = w

            print('      更新:', '臂')
            bound_dict[m] = upper_bound_probs(w, new_d, new_x, beta)[0, 0]
            print('结束循环第{}个portfolio'.format(m))
            print('-'*20)
    else:
        print('*-*-*-*非大调仓*-*-*-*')
        print('-' * 20)
        for m in range(len(cluster_copy)):
            print('开始循环第{}个portfolio'.format(m))
            a = utility_series(timerange_n, returns_df, cluster_copy[m], n, eta)
            # Ca = returns_df.loc[timerange_n[0]:timerange_n[-1], cluster_copy[m]]
            d = utility_series(timerange_m, returns_df, cluster_copy[m], n, eta)
            b1 = a.isna().sum()
            b2 = d.isna().sum()
            nan_code = b2[b2 != 0].index.tolist()
            nan_code.extend(b2[b2 != 0].index.tolist())
            nan_code = list(set(nan_code))
            substitute_code_3 = np.array(quitted_codes_used)[random.sample(list(range(0, len(quitted_codes_used))), len(nan_code))]
            print(' 替用基金3:', substitute_code_3)
            new_a = a.drop(nan_code, axis=1)
            new_d = d.drop(nan_code, axis=1)

            idx_list = []
            for _code_ in range(len(nan_code)):
                idx = cluster_copy[m].index(nan_code[_code_])
                cluster_copy[m][idx] = substitute_code_3[_code_]
                # 给予替换的基金较低的权重
                weight_list[m][idx] = 0.05
                new_a[substitute_code_3[_code_]] = a_quitted_new[substitute_code_3[_code_]]
                # new_d[substitute_code_3[_code_]] = d_quitted_new[substitute_code_3[_code_]]
            for cd4 in substitute_code_3:
                quitted_codes_used.remove(cd4)
            # 将未替换的基金按原来的权重重新分配权重
            copy_ = weight_list[m].copy()
            for w_ in range(len(copy_)):
                if copy_[w_] != 0.05:
                    weight_list[m][w_] = copy_[w_] / (np.sum(copy_)-0.05*len(nan_code)) * (1-0.05*len(nan_code))

            u = np.dot(new_a, weight_list[m])
            print('      更新:', 'w')
            w = ridge_regression(new_d, u, lam)  #无常数项的岭回归

            # 处理某个组合中权重小于0的数量过多的情形：
            # 如果全部为负，等权处理；如果最小值大于0.05或小于0.03，赋权0.05；其他情况按最小值赋权
            f = 0
            m_list = []
            for i in w.tolist():
                if i[0] <= 0:
                    f += 1
            # 如果权重列表中小于0的数大于等于5个，进行人为赋值。否则，默认将小于0的权重赋值为0
            if f >= len(weight_list[m]) * 0.5:
                m_list.append(m)
                weight_list_m = [k[0] for k in w.tolist()]
                _max_ = np.max(weight_list_m)
                # 如果权重全部小于0，等权处理
                if _max_ <= 0:
                    _min_ = 0.1
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
                        weight_list_m[o_] = weight_list_m_copy[o_] / (np.sum(weight_list_m_copy)-f*_min_) * (1-f*_min_)
                w = np.mat(weight_list_m).T
            else:
                w[w<0] = 0    #将出现的权重为负，强制设置为0
            weight_list[m] = w/np.sum(w)   #对权重进行标准化
            print('      计算:', 'x')
            # x = utility_oneday(returns_df, timerange_n[0], timerange_n[-1], cluster_copy[m])
            x = utility_series(timerange_n, returns_df, cluster_copy[m], n, eta).mean()
            # x = returns_df.loc[timerange_n[0]:timerange_n[-1], cluster_copy[m]].mean()
            print('      更新:', '臂')
            bound_dict[m] = upper_bound_probs(w, new_d, x, beta)[0, 0]
            print('结束循环第{}个portfolio'.format(m))
            print('-'*20)

    print('计算权重:', weight_list)
    print('计算臂值:', bound_dict)
    key_name = max(bound_dict, key=bound_dict.get)
    print('选择的臂:', key_name)
    # print('选择的权重:', weight_list[key_name].reshape((2, 5)))
    print('选择的组合:', cluster_copy[key_name])
    key_times[str(key_name)] += 1
    print('臂选择次数:', key_times)

    # 得到所选臂的实际收益
    backtesting = backtesting.append(
        [{'date': timerange_n[-1], 'Portfolio-capital': moneyPrt, 'HS300-capital': moneyHS300}], ignore_index=True)
    code = cluster_copy[key_name]
    amount = []
    # 获取最优portfolio的每个股票权重
    weight = weight_list[key_name]
    # 将资金按权重分给对应股票
    div = moneyPrt * weight
    print('div:', div)
    # 计算资金可购买的股票数量
    for j in range(len(code)):
        amount.append(int(div[j][0, 0] / nav_df_all[code[j]][timerange_n[-1]]))
    # 买股票后的剩余资金
    cash_funds = moneyPrt - (np.mat(amount) * (np.mat(nav_df_all.loc[timerange_n[-1], code]).T))[0, 0]
    # 可以购买指数的数量
    amount_index = int(moneyHS300 / HS300.loc[timerange_n[-1]])
    # 买指数后的剩余资金
    cash_index = moneyHS300 - (HS300.loc[timerange_n[-1]] * amount_index)

    print('开始计算:', 'holding time')
    # 计算持有portfolio delta天的每天收益
    holding_time = date_all[t:t+delta]
    for d in holding_time:
        moneyPrt = (cash_funds + (np.mat(amount) * (np.mat(nav_df_all.loc[d, code].values).T)))[0, 0]
        print('moneyPt:', moneyPrt)
        moneyHS300 = cash_index + (amount_index * HS300.loc[d])
        print('moneyHS:', moneyHS300[0])
        print('-'*15)
        backtesting = backtesting.append([{'date': d, 'Portfolio-capital': moneyPrt, 'HS300-capital': moneyHS300[0]}],
                                         ignore_index=True)

        # 计算后悔度
        reward = []
        for j in range(len(cluster_copy)):
            weight = weight_list[j]
            reward.append((np.dot(weight.T, returns_df.loc[d, cluster_copy[j]]))[0, 0])
        regret += np.max(reward) - reward[key_name]
        cum_regret = cum_regret.append([{'date': d, 'cum_regret': regret}], ignore_index=True)

    if backtesting['Portfolio-capital'].isna().sum() != 0:
        break

    print('开始计算:', 'holding_later')
    # 计算投资期最后一天的总资本
    holding_later = date_all[t+delta-1]
    moneyPrt = cash_funds + (np.mat(amount) * (np.mat(nav_df_all.loc[holding_later, code]).T))[0, 0]
    moneyHS300 = (cash_index + (amount_index * HS300.loc[holding_later]))[0]

    # GA:大调仓时跳过ga，其他时间正常运行
    if len(set([i[0, 0] for a in weight_list for i in a])) != 1:
        print('-' * 20)
        print('*******遗传算法*******')
        print('-' * 20)
        pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
        cluster_list = []
        for _ in range(N_GENERATIONS):
            print('开始更新第{}代'.format(_+1))
            x, y = translateDNA(pop, DNA_SIZE, X_BOUND, Y_BOUND)
            print('beta:', x)
            print(' lam:', y)
            print('  -'*4)
            unique_pop_dict = {}
            regret2_dict = {}
            weight_list_dict = {}
            cluster_dict = {}
            for c in range(POP_SIZE):
                print('  计算第{}个pop: {}'.format(c+1, (x[c], y[c])))
                regret2 = bandit_stock(delta, n, t, eta, weight_list_ga, cluster_ga, HS300, date_all, quitted_codes_ga, candidates_ga, x[c], y[c])
                cluster_list.append(cluster_copy)

                print('  -'*4)
                if ((x[c], y[c])) not in regret2_dict.keys():
                    regret2_dict[(x[c], y[c])] = round(regret2[0], 6)
                    unique_pop_dict[(x[c], y[c])] = pop[c]
                    weight_list_dict[(x[c], y[c])] = regret2[1]
                    cluster_dict[(x[c], y[c])] = regret2[2]
            print('regret2:', regret2_dict)
            fitness = get_fitness([-i for i in list(regret2_dict.values())])
            superb_params = min(regret2_dict, key=regret2_dict.get)
            print('最优(beta,lam):', superb_params)
            print('最小的regret:', regret2_dict[superb_params])
            # 保留最好的一个pop，放入下一代populations
            pop_best = unique_pop_dict[superb_params]
            print('最优的pop:', pop_best)
            pop_remain = select(np.array(list(unique_pop_dict.values())), POP_SIZE, fitness)
            print('开始crossover和mutation')
            pop_remain = np.array(crossover_and_mutation(pop_remain, DNA_SIZE, CROSSOVER_RATE, MUTATION_RATE))
            pop = np.insert(pop_remain, 0, pop_best, axis=0)
            print('-' * 15)

        t = t + delta

        # 将beta、lam、weight、cluster替换为ga给出最优beta、lam、weight、cluster
        beta = superb_params[0]
        lam = superb_params[1]
        beta_list.append(beta)
        lam_list.append(lam)

        weight_list = weight_list_dict[superb_params]
        cluster_copy = cluster_dict[superb_params]

        print('下一回测期的参数:', (beta, lam))
        print('下一回测期的权重:', weight_list)
    else:
        t = t+delta

    print('-----------结束第{}轮计算-----------'.format(count_times))
    count_times += 1
    print('*-' * 60)

    # 循环结束条件 numround选1218对应2020-12-31
    if t >= numround - delta:
        break

backtesting_copy = copy.deepcopy(backtesting)
backtesting_copy.set_index('date', inplace=True)
backtesting_copy.drop_duplicates(inplace=True)
backtesting_ret = backtesting_copy / backtesting_copy.iloc[0] - 1
backtesting_ret.plot()
plt.title('{}~{}'.format(delta, n))
plt.savefig('循环2图/{}~{}~{}-{}.jpg'.format(delta, n, eta, eta_times))
plt.show()
backtesting.to_excel('循环2图/{}~{}~{}-{}.xlsx'.format(delta, n, eta, eta_times))
end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('beta范围:', [np.min(beta_list), np.max(beta_list)])
print(' lam范围:', [np.min(lam_list), np.max(lam_list)])
print('bandit循环开始和结束时间:', (start_time, end_time))

def count_times(list):
    dict = {}
    for i in list:
        if i not in dict.keys():
            dict[i] = 0
        dict[i] += 1
    df = pd.DataFrame(dict, index=['times'])
    return np.transpose(df).sort_index()