'''
Developer Name: Jam Cem
Developer's email: zsan63864@gmail.com
Developer Organization: Chongqing University of Technology(China)
Developer Date: 2024/5/26
Developer's bilibili account: UID2085885765
Note: This is the code of the genetic algorithm designed in my undergraduate thesis to solve the CVRP, it is strictly prohibited to copy and paste directly, can be appropriate reference to borrow, violators will be punished!
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

''' Defining Functions (定义函数) '''

# Initialize solutions (generate initial solutions)
# 初始化解（生成初始解）
def initial_solution(num):
    array = random.sample(range(1, num + 1), num)
    veh_num = num
    array.insert(num, 0)
    array.insert(0, 0)
    a_0 = list(range(1, num + 1, 1))
    route_0 = np.random.choice(a_0, veh_num - 1, replace = False) 
    for i in route_0:
        array.insert(i, 0)
    return array

# Split the initial solution (output a list consisting of random routes)
# 拆分初始解（输出随机路径组成的列表）
def disassemble(array):
    index = [i for i, val in enumerate(array) if val == 0]
    veh_set = []
    for i in range(len(index) - 1):
        veh_set.append(array[index[i] + 1: index[i + 1]])
    if array[0] != 0:
        veh_set.append(array[:index[0]])
    if array[-1] != 0:
        veh_set.append(array[index[-1] + 1:])
    return veh_set

# Calculate loads (output loads for each vehicle on a random route)
# 计算载重（输出随机路径上每辆车的载重量）
def veh_load(veh_set):
    veh_loads = []
    for i in veh_set:
        total_load = sum(need[j - 1] for j in i)
        veh_loads.append(total_load)
    return veh_loads

# Determination of load (each vehicle load meets weight constraints?)
# 判断载重（每辆车载重符合重量约束？）
def check_veh_loads(veh_loads):
    overloads = [i for i, load in enumerate(veh_loads, 1) if load > veh_max_limit]
    no_overloads = [i for i, load in enumerate(veh_loads, 1) if load <= veh_max_limit]
    flag = len(overloads) == 0
    return no_overloads, overloads, flag

# Adjustment of loads (adjustment of overloaded goods to vehicles that are not overloaded)
# 调整载重（将超载的货物调整到未超载的车辆上）
def adjust(overloads, no_overloads, veh_set, limit):
    veh_loads = veh_load(veh_set)
    for j in overloads:
        while veh_loads[j - 1] > limit:
            for i in no_overloads:
                if veh_set[j - 1] and need[veh_set[j - 1][0] - 1] <= limit - veh_loads[i - 1]:
                    veh_set[i - 1].append(veh_set[j - 1][0])
                    veh_set[j - 1].pop(0)
                    veh_loads = veh_load(veh_set)
    return veh_set

# Rationalization of loads (judgement and adjustment of loads)
# 合理化载重（判断和调整载重）
def rationalize(array):
    veh_set = disassemble(array)
    veh_loads = veh_load(veh_set)
    no_overloads, overloads, flag = check_veh_loads(veh_loads)
    veh_set = adjust(overloads, no_overloads, veh_set, veh_max_limit)
    return veh_set

# Merge (combining each random route after rationalization into a solution)
# 合并（将合理化后的每条随机路径合并为解）
def merge_array(veh_set):
    new_array = []
    for i in veh_set:
        i.append(0)
        new_array.extend(i)
    new_array.insert(0, 0)
    return new_array

# Delineate routes (add zeros before and after split solutions)
# 划分路径（在拆分的解前后添加0）
def add_0(veh_set):
    for i in veh_set:
        i.append(0)
        i.insert(0, 0)
    return veh_set

# Calculate fitness (calculate fitness for each route)
# 计算适应度（计算每条路径的适应度）
def get_fit(veh_set):
    load_sum = 0
    for i in veh_set: 
        for j in range(len(i) - 1):
            load_sum += matrix[i[j], i[j + 1]]
    return load_sum

 # Cyclic generation of initial solutions
 # 循环生成初始解
def generate_initial_solution():
    route_sum_set = []
    for i in range(array_num):
        array = initial_solution(num)
        veh_set = rationalize(array)
        new_array = merge_array(veh_set)
        array_set.append(new_array)
        veh_set = disassemble(new_array)
        veh_set = add_0(veh_set)
        route_sum = get_fit(veh_set)
        route_sum_set.append(route_sum)
    return route_sum_set

# Selection of populations
# 选择种群
def select(array_set, route_sum_set):
    new_array_set = []
    k = len(array_set) // 2
    route_sum_set = np.array(route_sum_set)
    array_index = route_sum_set.argsort()[0: k]
    for i in array_index:
        new_array_set.append(array_set[i])
    return new_array_set

# Crossover
# 交叉
def crossover(new_selectors, crossover_rate):
    selectors_cr = []
    while len(selectors_cr) < len(new_selectors):
        if random.random() < crossover_rate:
            parent1, parent2 = random.sample(new_selectors, 2)
            child1, child2 = pmx_crossover(parent1, parent2)
            selectors_cr.extend([child1, child2])
        else:
            selectors_cr.extend(random.sample(new_selectors, 2))
    return selectors_cr

# PMX
# 部分匹配交叉
def pmx_crossover(parent1, parent2):
    length = len(parent1)
    cross_point1, cross_point2 = sorted(random.sample(range(length), 2))
    child1, child2 = parent1[:], parent2[:]
    for i in range(cross_point1, cross_point2):
        value1, value2 = child1[i], child2[i]
        index1, index2 = child1.index(value2), child2.index(value1)
        child1[i], child1[index1] = child1[index1], child1[i]
        child2[i], child2[index2] = child2[index2], child2[i]
    return child1, child2

# Mutate
# 变异
def mutate(selectors_mu, mutation_rate):
    mutation_count = 0
    for selector in selectors_mu:
        if random.random() < mutation_rate:
            mutation_count += 1
            swap = random.sample(range(len(selector)), k=2)
            site1, site2 = swap
            selector[site1], selector[site2] = selector[site2], selector[site1]
    return selectors_mu

# Number of vehicles counted
# 统计车辆数
def count_vehicle_num():
    count = 0
    for i in veh_opt:
        if len(i) > 2:
            count += 1
    print(f'需要车辆数为：{count}辆')
    # print(f'Number of vehicles required: {count}')

# Output the distance traveled by vehicles on each route
# 输出每条路径车辆的行驶距离
def distance_per_vehicle(veh_set):
    j = 1
    for i, route in enumerate(veh_set, start=1):
        if len(route) > 2:
            distance = get_fit([route])
            print(f'车辆 {j} 的行驶距离为: {distance:.2f} 公里')
            # print(f'The distance traveled by vehicle {j} is: {distance:.2f} km')
            j += 1

# Convergence plot for number of iterations
# 迭代次数收敛图
def plt_convergence():
    Y = min_routes
    times = len(Y)
    X = list(range(1, times+1))
    plt.figure(figsize = (10, 6))
    plt.plot(X, Y, marker = 'X', ms = 1, label = '车辆行驶距离', color = 'red')  # label = 'Vehicle travel distance'
    plt.title('车辆最短路径迭代收敛图', fontsize=16)  # Vehicle shortest route iteration convergence plot  
    plt.xlabel('迭代次数', fontsize = 14) and plt.ylabel('行驶距离', fontsize = 14)  # xlabel = 'Number of iterations', ylabel = 'Distance traveled' 
    plt.legend(fontsize = 12)
    plt.grid(True)
    plt.show()

# Map of vehicle travel routes
# 车辆行驶路径图
def plt_vehicle_routes():
    colors = plt.cm.viridis(np.linspace(0, 1, len(veh_opt)))
    plt.figure(figsize = (10, 6))
    for i, route in enumerate(veh_opt):
        X = [local[index][0] for index in route]
        Y = [local[index][1] for index in route]        
        plt.plot(X, Y, marker = 'o', linestyle = '-', color = colors[i])
        plt.title('车辆最短路径规划图', fontsize = 16)  # title = 'Vehicle shortest route planning map'
        plt.grid(True)        
        for j in range(len(route)):
            index = route[j]
            x, y = local[index]
            plt.text(x, y, f'({x}, {y})', fontsize=8, ha='right')
    plt.xlabel('X', fontsize = 14) and plt.ylabel('Y', fontsize = 14)
    plt.show()

''' Parameter Settings (参数设置) '''

# Setting the demand at the customer point
# 设置客户点的需求量
need = [5, 6, 15, 7, 1, 12, 18, 8, 3, 13]
num = len(need)

# Vehicle capacity
# 车辆容量
veh_max_limit = 20

# Vehicle cost per kilometer traveled
# 车辆每公里行驶成本
veh_per_cost = 12

# Set site coordinates
# 设置站点坐标
local = [[0, 0], [-16, 12], [32, -2], [-22, 48], [-8, -36], [7, -25], [8, 45], [-20, -2], [-37, -42], [-20, -31], [31, 30]]
starting_point = [0, 0]

# Calculate the distance matrix
# 计算距离矩阵
matrix = []
matrix = distance.cdist(local, local, 'euclidean')

# Number of iterations
# 迭代的次数
times = 10000

# population size
# 种群数量
array_num = 50
array_set = []

# Crossover rate
# 交叉概率
crossover_rate = 0.7

# Mutate rate
# 变异概率
mutatu_rate = 0.02

''' Main Cycle (主循环) '''

min_routes = []
route_sum_set = generate_initial_solution() 
count = 0
for i in range(times):
    count += 1
    print(f'目前迭代次数为：{count}次')
    # print(f'The current number of iterations is: {count} times')
    
    # Adaptation calculations
    # 适应度计算
    route_sum_set = []
    for j in array_set:
        veh_set = disassemble(j)
        veh_set = add_0(veh_set)
        route_sum_set.append(get_fit(veh_set))

    # Selection of populations
    # 选择种群
    selectors = select(array_set, route_sum_set)

    # Clone populations
    # 克隆种群
    selectors_clone = selectors[:]

    # Output selects the worst individual of the population
    # 输出选择种群的最差个体
    route_sum_opt = []
    for j in selectors:
        veh_opt = disassemble(j)
        veh_opt = add_0(veh_opt)
        route_sum_opt.append(get_fit(veh_opt))
    bad_index = route_sum_opt.index(max(route_sum_opt))

    # Replicating the worst individual in a clonal population to reach the population size
    # 复制克隆种群中最差个体达到种群数量
    new_selectors = []
    if len(selectors_clone) < array_num:
        s = array_num - len(selectors_clone)
        for _ in range(s):
            new_selectors.append(selectors_clone[bad_index])

    # Crossover
    # 交叉
    crossoverer = crossover(new_selectors, crossover_rate)

    # Mutate
    # 变异
    mutatuer = mutate(crossoverer, mutatu_rate)

    # Rationalized crossovered and mutated solutions
    # 合理化交叉与变异后的解
    array_opt = []
    for i in mutatuer:
        veh_rationalize = rationalize(i)
        array_rationalize = merge_array(veh_rationalize)
        array_opt.append(array_rationalize)

    array_opt_1 = []
    for i in selectors:
        veh_rationalize = rationalize(i)
        array_rationalize = merge_array(veh_rationalize)
        array_opt_1.append(array_rationalize)
    selectors = array_opt_1

    # Rationalize and then merge new populations
    # 合理化后再合并新种群
    for i in array_opt:
        selectors.append(i)
    array_opt = selectors

    # Record the current route and continue iterating
    # 记录当前路径，继续进行迭代
    array_set = array_opt 
    
    # Calculate adaptation at the end of the process
    # 流程结束后计算适应度
    route_sum_opt = []
    for j in array_opt:
        veh_opt = disassemble(j)
        veh_opt = add_0(veh_opt)
        route_sum_opt.append(get_fit(veh_opt))

    # Output the current optimal data
    # 输出目前最优数据
    program = route_sum_opt.index(min(route_sum_opt))
    print(f'目前最短路径规划方案为：{array_set[program]}')
    # print(f'The current shortest route planning scheme is: {array_set[program]}')
    min_routes.append(min(route_sum_opt))
    print(f'目前最短路线为：{min(route_sum_opt):.2f}公里')
    # print(f'The current shortest route is: {min(route_sum_opt):.2f} km')

# Split each solution
# 拆分每个解
veh_opt = disassemble(array_opt[program])
veh_opt = add_0(veh_opt)
min_route_length = get_fit(veh_opt)
total_cost = veh_per_cost * min_route_length
print(f'最短路径规划方案为：{veh_opt}')
print(f'最短路径为：{min_route_length:.2f}公里')
print(f'最短路径成本为：{total_cost:.2f}元')
'''
print(f'The shortest route planning scheme is: {veh_opt}')
print(f'The shortest route is: {min_route_length:.2f} km')
print(f'The shortest route cost is: {total_cost:.2f} dollars')'''
count_vehicle_num()
distance_per_vehicle(veh_opt)

# Font setting (Chinese Song)
# 字体设置（中文宋体）
plt.rcParams.update({
    'font.sans-serif': 'SimSun',
    'axes.unicode_minus': False
})

# Plt
# 绘图
plt_convergence()
plt_vehicle_routes()