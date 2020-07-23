import numpy as np
import re
import math
import copy
import matplotlib.pyplot as plt
import random
import time

# 退火算法
def TSP_tuihuo(T_start,T_end,L,init_list,Q):
    every_solution = []
    city_new = init_list
    city_old = []
    path_new = float("inf")
    path_old = float("inf")
    while T_start > T_end:
        for i in range(L):
            city_old = copy.copy(city_new)
            city_new = Get_new(city_new)
            path_old = path_length(city_old)
            path_new = path_length(city_new)
            df = path_new-path_old
            if df > 0:
                p = math.exp(-df / T_start)
                randnum = random.uniform(0, 1)
                if p > randnum:
                    city_old = copy.copy(city_new)
                else:
                    city_new = copy.copy(city_old)
            else:
                city_old = copy.copy(city_new)
        every_solution.append(path_old)
        T_start = T_start*Q
    return city_old, path_old, every_solution
# 构造新解
def Get_new(init_list):
    randp = random.uniform(0, 1)
    if randp <= 0.4:
        rand1 = random.randint(0, city_num - 1)
        rand2 = random.randint(0, city_num - 1)
        while rand1 == rand2:
            rand2 = random.randint(0, city_num - 1)
        init_list[rand1], init_list[rand2] = init_list[rand2], init_list[rand1]
    elif 0.4 < randp <= 0.7:  # 三交换
        while True:
            rand1 = random.randint(0, city_num - 1)
            rand2 = random.randint(0, city_num - 1)
            rand3 = random.randint(0, city_num - 1)
            if ((rand1 != rand2) & (rand1 != rand3) & (rand2 != rand3)):
                break

        if rand1 > rand2:
            rand1, rand2 = rand2, rand1
        if rand2 > rand3:
            rand2, rand3 = rand3, rand2
        if rand1 > rand2:
            rand1, rand2 = rand2, rand1

        tmplist = init_list[rand1:rand2].copy()
        init_list[rand1:rand3 - rand2 + 1 + rand1] = init_list[rand2:rand3 + 1].copy()
        init_list[rand3 - rand2 + 1 + rand1:rand3 + 1] = tmplist.copy()
    else:
        rand1 = random.randint(0, city_num - 1)
        rand2 = random.randint(0, city_num - 1)
        while rand1 == rand2:
            rand2 = random.randint(0, city_num - 1)
        if rand1 > rand2:
            rand1, rand2 = rand2, rand1
        for i in range(rand1, int((rand1 + rand2) / 2) + 1):  # 交换序列
            temp = init_list[i]
            init_list[i] = init_list[rand2 + rand1 - i]
            init_list[rand2 + rand1 - i] = temp
    """
    elif 0.7 < randp <=0.8:
        while True:
            rand1 = random.randint(0, city_num - 1)
            rand2 = random.randint(0, city_num - 1)
            rand3 = random.randint(0, city_num - 1)
            if ((rand1 != rand2) & (rand1 != rand3) & (rand2 != rand3)):
                break
        init_list[rand1], init_list[rand2], init_list[rand3] = init_list[rand2], init_list[rand3], init_list[rand1]
    """
    return init_list

def openmap(fdir = "./st70.tsp"):
    f = open(fdir)
    cities = {}
    l = 6
    for line in f:
        if l >= 1:
            l -= 1
            continue
        elif (line[0]+line[1]+line[2]) != "EOF":
            line = line.split(' ')
            line = list(filter(None, line))
            cities[float(line[0])] = [float(line[1]),float(line[2])]
        elif (line[0]+line[1]+line[2]) == "EOF":
            break
    f.close()
    print(cities)
    return cities

def draw(mymap, bestlist):
    cities = np.array(mymap)
    city_num = cities.shape[0]
    bestpath  = np.array(bestlist)
    Rangex = max([float(cities[i][0]) for i in range(city_num)])
    Rangey = max([float(cities[i][1]) for i in range(city_num)])
    plt.plot(cities[:, 0], cities[:, 1], 'r.', marker=u'$\cdot$')
    plt.xlim(0, Rangex)
    plt.ylim(0, Rangey)

    for i in range(city_num-1):
        m = int(bestpath[i])
        n = int(bestpath[i + 1])
        plt.plot([float(cities[m][0]), float(cities[n][0])],
                 [float(cities[m][1]), float(cities[n][1])], 'k')
    plt.plot([float(cities[int(bestpath[0])][0]), float(cities[int(n)][0])],
             [float(cities[int(bestpath[0])][1]), float(cities[int(n)][1])], 'b')
    plt.show()

def path_length(arr):  # 路径长度
    length = 0
    for i in range(city_num - 1):
        length += d[arr[i]][arr[i + 1]]
    length += d[arr[0]][arr[-1]]
    return length

def show_shoulian(every_solution):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    axes.plot(every_solution, 'k', marker=u'')
    axes.set_title('Average Length')
    axes.set_xlabel(u'iteration')
    plt.show()


cities_set = []
cities_tups = []
cities_dict = {}

distance = openmap()

T_start = 10000  # 初始温度
T_end = 1e-14  # 结束温度
Q = 0.985    # 退火系数
L = 500  # 迭代次数
city_num = len(distance)  # 城市数
#print(city_num)
init_list = [0] * city_num  # 初始城市序列

d = [[0 for col in range(city_num)] for row in range(city_num)]  # 距离矩阵
for i in range(city_num):
    for j in range(city_num):
        if i == j:
            d[i][j] = 0
        else:
            x = (float(distance[i+1][0]) - float(distance[j+1][0])) ** 2
            y = (float(distance[i+1][1]) - float(distance[j+1][1])) ** 2
            result = (x + y) ** 0.5
            d[i][j] = result
# print(d)


# 初始化一个解
for i in range(city_num):
    init_list[i] = i

start = time.time()
bestlist, bestsolution, every_solution = TSP_tuihuo(T_start, T_end, L, init_list, Q)
end = time.time()
print("运行时间: ", end - start)
print("最佳路径: ", bestlist)
print("最短距离: ", bestsolution)
#print(every_solution)
draw(tuple(distance.values()), bestlist)
show_shoulian(every_solution)