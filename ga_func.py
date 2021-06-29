import numpy as np

def translateDNA(pop, DNA_SIZE, X_BOUND, Y_BOUND):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 每行取奇数列表示x
    y_pop = pop[:, ::2]   # 每行取偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2**np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y = y_pop.dot(2**np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    x = np.round(x, 1)
    y = np.round(y, 2)
    return x, y

def crossover_and_mutation(pop, DNA_SIZE, CROSSOVER_RATE, MUTATION_RATE):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(len(pop))]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child, DNA_SIZE, MUTATION_RATE)  # 每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop

def mutation(child, DNA_SIZE, MUTATION_RATE=0.1):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

def get_fitness(pred):
    return (pred - np.min(pred)) + 1e-4

def select(pop, POP_SIZE, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(len(pop)), size=POP_SIZE-1, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]