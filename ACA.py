import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 蚁群算法解决电网经济调度问题
# 读取数据
address = r"C:\Users\bendaye\Desktop\课程大作业（1）_IEEE30节点_日前经济调度数据.xls"
address2 = r"C:\Users\bendaye\Desktop\损耗参数.xlsx"
PD = np.array(pd.read_excel(address, usecols='B')[1:])
P_max = np.array(pd.read_excel(address, usecols='E')[1:7])
ai = np.array(pd.read_excel(address, usecols='G')[1:7])
bi = np.array(pd.read_excel(address, usecols='H')[1:7])
ci = np.array(pd.read_excel(address, usecols='I')[1:7])
Bij = np.array(pd.read_excel(address2,  usecols=[1, 2, 3, 4, 5, 6]))[1:7, :]
Bi0 = np.array(pd.read_excel(address2,  usecols=[1, 2, 3, 4, 5, 6]))[7, :]
B00 = 9.8573e-4

# 参数设置
alpha = 1 # 信息素重要程度因子
beta = 4 # 启发函数重要程度因子
rho = 0.1  # 挥发速度
G = 1000 # 最大迭代值
R1 = 0.5 # 罚函数系数
R2 = 1e+20  # 罚函数系数
# 存储设置
PGi = np.zeros(len(P_max))  # 记录每个机组的出力
e = [0, 0, 0, 0, 0, 0]  # 存储适应度矩阵
P_predict_total = np.zeros(24)  # 存储每一时刻的预测负荷

# 计算每台机组节点的适应度
for i in range(len(P_max)):
    e[i] = (ai[i] ** 2 + bi[i]**2 + ci[i])[0]
# 迭代开始
for T in range(24):
    gen = 0
    expenditure_min = []  # 存储每次迭代的最小花费
    pheromonetable = np.ones(len(P_max))  # 信息素矩阵
    protrans = np.zeros(len(PGi))  # 每次循环都更改当前的机组概率矩阵
    strategy = []  # 存储每次迭代的策略
    strategy_best = []  # 存储每次迭代最优策略
    ant_num = int(PD[T]) + 2  # 蚂蚁数量
    while gen < G:
        expenditure = []  # 存储每次迭代个策略的花费
        # 生成任务节点矩阵
        task_node = np.zeros((ant_num, len(P_max)))
        # 按照信息素浓度和启发函数计算选择每个机组的概率
        for j in range(len(P_max)):
            protrans[j] = pheromonetable[j] ** alpha *( (1 / e[j]) ** beta )
        # 累计概率，轮盘赌选择
        cumsumprobtrans = (protrans / sum(protrans))
        # 每只蚂蚁被分配到一个机组
        for i in range(ant_num):
            cumsumprobtrans1 = cumsumprobtrans - np.random.rand()*max(cumsumprobtrans)*1.1
            # 求出离随机数产生最近的索引值
            while max(cumsumprobtrans1) < 0:
                cumsumprobtrans1 = cumsumprobtrans - np.random.rand()*max(cumsumprobtrans)*1.1
            l = np.where(cumsumprobtrans1 > 0)[0][0]
            while np.sum(task_node[:, l]) >= P_max[l]:
                # 移除l位置的数
                cumsumprobtrans2 = cumsumprobtrans
                cumsumprobtrans2[l] = 0
                cumsumprobtrans1 = cumsumprobtrans2 - np.random.rand()
                while max(cumsumprobtrans1) < 0:
                    cumsumprobtrans1 = cumsumprobtrans - np.random.rand()*max(cumsumprobtrans)*1.1
                l = np.where(cumsumprobtrans1 > 0)[0][0]
            task_node[i, l] = 1
            # 计算每个任务节点的出力
        for m in range(len(PGi)):
            PGi[m] = sum(task_node[:, m])
        strategy.append(PGi)
        upIndex = np.argsort(PGi, axis=0)
        # 计算该策略的适应度
        PL = np.dot(np.dot(PGi.T, Bij), PGi) * 1e-4 + np.dot(Bi0.T, PGi) * 1e-2 + B00
        pay = (np.dot(ai.T, PGi ** 2) + np.dot(bi.T, PGi)) + sum(ci)
        # print('第', gen+1, '次迭代，该策略为：', PGi)
        expenditure.append(pay)
        # if abs(pay-min(expenditure)) < 100 and gen > 80:
        #     break
        # 更新信息素
        for n in range(len(upIndex)):
            pheromonetable[upIndex[n]] = (1 - rho) * pheromonetable[upIndex[n]] + PGi[upIndex[-(n+1)]] * (pay ** (pay/100))
        # print('第', gen+1, '次迭代，''信息素矩阵：', pheromonetable)
        gen = gen + 1
    expenditure_min.append(min(expenditure))
    strategy_best.append(strategy[expenditure.index(min(expenditure))])
    # 输出结果
    print('第', T+1, '时刻最优策略：')
    print("PD[T] = ", PD[T])
    print('最小花费：', min(expenditure_min))
    for i in range(len(P_max)):
        print('机组', i+1, '出力：', strategy_best[expenditure_min.index(min(expenditure_min))][i])
    P_predict_total[T] = sum(strategy_best[expenditure_min.index(min(expenditure_min))])

# # 画图
# print('预测负荷：', P_predict_total)
# plt.figure()
# plt.plot(P_predict_total, 'r', label='预测负荷')
# plt.plot(PD, 'b', label='需求负荷')
# # plt.plot(PDL, 'g', label='实际负荷')
# plt.xlabel('时间/h')
# plt.ylabel('负荷/MW')
# plt.legend()
# plt.show()