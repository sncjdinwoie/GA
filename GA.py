import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
NP = 360  # 种群数量
G = 200  # 迭代次数
N = len(P_max)  # 机组数量
R1 = 180000 # 罚函数系数
R2 = 9000000000  # 罚函数系数
R3 = 3000000000  # 罚函数系数
R4 = 1000 # 罚函数系数
L = 7  # 二进制编码长度

# 存储设置
best = np.zeros(G)  # 存储历代最优解
best_pop = np.zeros((G, N, L))  # 存储每一代的最优个体
expenditure = np.zeros((NP, 1))  # 存储每一代的适应度函数值
P_predict_total = np.zeros(24)  # 存储每一天的预测负荷
PDL = np.zeros(24)  # 存储每一天的损耗负荷与实际负荷的差值

# 将二进制向量转化为十进制数函数
def binary2decimal(vector):
     m = 0
     for i in range(len(vector)):
         m = vector[i] * 2 ** (L - i - 1) + m
     return m


# 计算适应度函数
def cal(vector, T):
    L1 = len(vector)
    m = np.zeros((L1, 1))
    for i in range(len(vector)):
        for j in range(len(vector[i])):
                m[i] = m[i] + vector[i][j] * 2 ** (L - j - 1)
    PL = np.dot(np.dot(m.T, Bij), m) + np.dot(Bi0.T, m) + B00
    pay = np.dot(ai.T, m**2) + np.dot(bi.T, m) + sum(ci) + R1 * ((sum(m)-PD[T] - PL) ** 2 + R2 * max(0, max(m[2:] -P_max[2:]) + R3 * max(0, PD[T] - sum(m))))
    return pay[0][0]

# 迭代开始
for T in range(24):
    # 随机生成初始种群
    f = np.zeros((NP, N, L))
    for i in range(NP):
        for j in range(N):
            f[i, j] = np.random.randint(0, 2, L)
            # if binary2decimal(f[i, j]) > P_max[j][0]:
            #     f[i, j] = np.zeros(L)
    gen = 0
    while gen < G:
        # 计算适应度
        for i in range(NP):
            expenditure[i] = cal(f[i], T)
        # 找出最优解
        best[gen] = min(expenditure)
        best_pop[gen] = f[np.argmin(expenditure)]
        # 选择
        expenditure_index = np.argsort(expenditure, axis=0)
        cache_pop = np.zeros((NP, N, L))  # 缓存每一代的种群
        for i in range(int(NP/4)):
            cache_pop[i] = f[expenditure_index[i]]
        # 交叉以及变异
        aa = int(NP/4)
        while aa < NP:
            # 选择两个父代
            if aa < NP-1:
                i1 = np.random.randint(0, aa)
                cache_pop[aa] = cache_pop[i1]
                i2 = np.random.randint(0, aa)
                cache_pop[aa + 1] = cache_pop[i2]
                for j in range(N):
                    # 选择交叉点
                    i3 = np.random.randint(0, L-1)
                    # 交叉操作
                    cache_pop[aa][j, i3:] = cache_pop[i2][j, i3:]
                    # if binary2decimal(cache_pop[aa][j]) > P_max[j][0]:
                    #     cache_pop[aa][j, i3:] = cache_pop[i1][j, i3:]
                    cache_pop[aa+1][j, i3:] = cache_pop[i1][j, i3:]
                    # if binary2decimal(cache_pop[aa+1][j]) > P_max[j][0]:
                    #     cache_pop[aa+1][j, i3:] = cache_pop[i2][j, i3:]
                aa = aa + 2
            # 变异操作
            if aa < NP:
                # 选择变异个体
                i4 = np.random.randint(0, aa)
                cache_pop[aa] = cache_pop[i4]
                for j in range(N):
                    # 选择变异点
                    i5 = np.random.randint(0, L-1)
                    i6 = np.random.randint(0, L-1)
                    # 变异
                    cache_pop[aa][:, i5] = 1 - cache_pop[i4][:, i5]
                    # if binary2decimal(cache_pop[aa][j]) > P_max[j][0]:
                    #     cache_pop[aa][:, i5] = 1 - cache_pop[i4][:, i5]
                    cache_pop[aa][:, i6] = 1 - cache_pop[i4][:, i6]
                    # if binary2decimal(cache_pop[aa][j]) > P_max[j][0]:
                    #     cache_pop[aa][:, i6] = 1 - cache_pop[i4][:, i6]
                aa = aa + 1
        # 更新种群
        f = np.array(cache_pop).reshape(NP, N, L)
        gen = gen + 1
    # 输出结果
    P_predict = np.zeros(N)
    print('第', T+1, '时刻的最优解为：')
    print("PD[T] = ", PD[T])
    for i in range(N):
        P_predict[i] = binary2decimal(best_pop[np.argmin(best)][i])
        print('机组', i+1, '：', P_predict[i], "最大值：", P_max[i])
    P_predict_total[T] = sum(P_predict)
    PDL[T] = PD[T] + (np.dot(np.dot(P_predict.T, Bij), P_predict) + np.dot(Bi0.T, P_predict) + B00)
# 画图
print('预测负荷：', P_predict_total)
plt.figure()
plt.plot(P_predict_total, 'r', label='预测负荷')
plt.plot(PD, 'b', label='需求负荷')
# plt.plot(PDL, 'g', label='实际负荷')
plt.xlabel('时间/h')
plt.ylabel('负荷/MW')
plt.legend()
plt.show()
