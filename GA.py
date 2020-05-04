import numpy as np
import random
import math


class Problem:
    def __init__(self, model, image, true_label, eps, batch_size, least_index):
        self.model = model
        self.image = image
        self.true_label = true_label
        self.eps = eps
        self.batch_size = batch_size
        self.least_index = least_index

    def evaluate(self, x):
        predict = self.model.predict(
            np.clip(np.tile(self.image, (len(x), 1, 1, 1)) + np.reshape(x, (len(x),) + self.image.shape) * self.eps, 0.,
                    1.), batch_size=self.batch_size)
        p = np.copy(predict[:, self.true_label])
        r = np.copy(predict[:, self.least_index])
        predict[:, self.true_label] = 0
        q = np.max(predict, axis=-1)

        # Unconstrained optimization
        G = np.zeros(p.shape)
        return [p[:, np.newaxis], (-q)[:, np.newaxis], (p - q)[:, np.newaxis], (-r)[:, np.newaxis]], G[:, np.newaxis]


def select(pop, fitness, feasible, n_select):
    length = len(pop)
    S = np.ndarray((n_select * 2,), dtype=np.int)
    indexlist = []
    index = np.arange(n_select * 2)
    np.random.shuffle(index)
    indexlist.append(index)
    np.random.shuffle(index)
    indexlist.append(index)
    indexlist = np.concatenate(indexlist)
    for i in range(indexlist.shape[0] // 2):
        if (feasible[indexlist[2 * i] % length] > 0 or feasible[indexlist[2 * i + 1] % length] > 0):
            S[i] = indexlist[2 * i] % length if feasible[indexlist[2 * i] % length] < feasible[
                indexlist[2 * i + 1] % length] else indexlist[2 * i + 1] % length
        else:
            S[i] = indexlist[2 * i] % length if fitness[indexlist[2 * i] % length] < fitness[
                indexlist[2 * i + 1] % length] else indexlist[2 * i + 1] % length
    return np.reshape(S, (n_select, 2))


def cross(pop, parents):
    M = np.random.random((len(pop), pop.shape[1])) <= 0.5
    off_new = pop[parents[:, 0]].copy()
    off_new[M] = pop[parents[:, 1]][M].copy()
    return off_new


def mutate(pop):
    prob = 1.0 / pop.shape[1]
    do_mutation = np.random.rand(pop.shape[0], pop.shape[1]) < prob
    mutate_matrix = np.random.randint(1, 3, size=pop.shape)
    pop[do_mutation] = pop[do_mutation] + mutate_matrix[do_mutation]
    pop[pop == 2] = -1
    pop[pop == 3] = 0
    return pop


def mating(pop, fitness, feasible):
    n_select = len(pop)
    parents = select(pop, fitness, feasible, n_select)
    off = cross(pop, parents)
    off = mutate(off)
    return off


def survival(off, off_fitness, off_feasible, n_gen, flag=0):
    temp = off_fitness[flag]
    temp = np.argsort(temp[:, 0])
    return off[temp][:n_gen], [off_fitness[0][temp][:n_gen], off_fitness[1][temp][:n_gen],
                               off_fitness[2][temp][:n_gen], off_fitness[3][temp][:n_gen]], off_feasible[temp][:n_gen]


def GA(pop_size, generation, length, model, image, true_label, eps, batch_size, gradient=None, multi_fit=False):
    generation_save = np.zeros((10000,))

    pred = np.squeeze(model.predict(image[np.newaxis, :, :, :]))
    r = np.argmin(pred)
    a = pred[true_label]
    pred[true_label] = 0
    b = -np.max(pred)
    c = a - b

    problem = Problem(model, image, true_label, eps, batch_size, r)
    pop = np.random.randint(-1, 2, size=(pop_size, length))
    if (not (gradient is None)):
        pop[0] = np.reshape(np.sign(gradient), (length))
    max_eval = pop_size * generation
    eval_count = 0
    fitness, feasible = problem.evaluate(pop)
    eval_count += pop_size

    use_flag = 0
    flag = 0
    count = 0
    generation_save[count] = np.min(fitness[2])
    if (len(np.where(fitness[2] < 0)[0]) == 0):
        while (eval_count < max_eval):
            count += 1
            P = random.random()

            # use multiple fitness
            if (multi_fit == True):
                if (use_flag != 1):
                    if (eval_count > max_eval / 2 or b - np.min(fitness[1]) > 0.25):
                        use_flag = 1
                P_list = [0.33, 0.66, 1]
                if (use_flag == 1):
                    if (eval_count > max_eval * 0.9 or np.min(fitness[2]) <= 0.03):
                        P_list = [0.25, 0.5, 1]
                    if (P < P_list[0]):
                        flag = 0
                    elif (P < P_list[1]):
                        flag = 1
                    else:
                        flag = 2
                else:
                    flag = random.choice([0, 3])
            # use traditional GA
            else:
                flag = 2
            off = mating(pop, fitness[flag], feasible)
            off_fitness, off_feasible = problem.evaluate(off)
            eval_count += pop_size
            off = np.row_stack((pop, off))
            off_fitness[0] = np.row_stack((fitness[0], off_fitness[0]))
            off_fitness[1] = np.row_stack((fitness[1], off_fitness[1]))
            off_fitness[2] = np.row_stack((fitness[2], off_fitness[2]))
            off_fitness[3] = np.row_stack((fitness[3], off_fitness[3]))
            off_feasible = np.row_stack((feasible, off_feasible))
            pop, off_fitness, feasible = survival(off, off_fitness, off_feasible, len(pop), flag)
            generation_save[count] = np.min(fitness[2])
            fitness = off_fitness
            if (len(np.where(fitness[2] < 0)[0]) != 0):
                break
    if (len(np.where(fitness[2] < 0)[0]) != 0):
        return pop[np.where(fitness[2] < 0)[0][0]], eval_count, generation_save[:count + 1]
    else:
        return pop[0], eval_count, generation_save[:count + 1]
