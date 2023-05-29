import numpy as np
import heapq
from random import choice


DNA_SIZE = 24
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 50
X_BOUND = [-3, 3]
Y_BOUND = [-3, 3]

# 10个学习者
# previous knowledge
goal =
PF = np.random.randint(1,5,10) # preferable format学习者对学习材料的偏好格式
tl = # 学习时间
KL = np.random.randint(1,4,10) # knowledge level学习者的知识水平
CL = np.random.randint(1,4,10) #  concentration level 学习者的集中水平



# 构造目标函数
def F(x, y):
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)

# 翻译DNA
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

# 计算适应度fitness

def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    return (pred - np.min(pred)) + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度

# 求精英数目

def Elite_num(Elite_percentage):
    Elite_num = int(Elite_percentage * POP_SIZE)
    return Elite_num

# 挑选精英个体

def Selection_Elitism(pop, Elite_percentage):
    fitness = get_fitness(pop)
    Pop_Elite = heapq.nlargest(Elite_num(Elite_percentage), fitness)
    return Pop_Elite

# 得到精英索引位置

def get_Elite_index(pop, Elite_percentage):
    Elite_index = list()
    for i in list(range(0,Elite_num(Elite_percentage))):
        a = get_fitness(pop).tolist()
        Elite_index.append(a.index(heapq.nlargest(Elite_num(Elite_percentage),a)[i]))
    return Elite_index

# 交叉

def crossover(pop, Elite_percentage, CROSSOVER_RATE=0.8):
    new_pop = []
    Elite_index = get_Elite_index(pop, Elite_percentage)
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[choice(Elite_index)]  # 再从精英群体中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points] = mother[cross_points]  # 孩子得到位于交叉点处的母亲的基因
        new_pop.append(child)
    return new_pop

# 变异

def mutation(pop, Elite_percentage, CROSSOVER_RATE=0.8, MUTATION_RATE=0.003):
    new_pop = crossover(pop, Elite_percentage, CROSSOVER_RATE)
    pop_child = []
    for child in new_pop:
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转
        pop_child.append(child)
    return pop_child

# 选择

def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]

# 展示结果

def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))

# 主函数

if __name__ == "__main__":

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        new_pop = crossover(pop, 0.1, CROSSOVER_RATE)
        pop = np.array(mutation(pop, 0.1, CROSSOVER_RATE, MUTATION_RATE))
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
