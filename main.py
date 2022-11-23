import copy
from datetime import timedelta, date
from json import dump
from typing import Callable, List
import numpy as np
import struct
from random import random, uniform, randint
from time import time
from joblib import Parallel, delayed
from plotly.figure_factory import create_gantt
from tqdm import tqdm

population_size = 250
crossover_probability = 0.85
mutation_probability = 0.2
epochs = 250


def softmax(vector: np.ndarray) -> np.ndarray:
    z = vector - max(vector)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


def inv_chr(string: str, position: int) -> str:
    if int(string[position]) == 1:
        string = string[:position] + '0' + string[position + 1:]
    else:
        string = string[:position] + '1' + string[position + 1:]
    return string


def bin_to_dec(binary: str) -> int:
    ret = 0
    for i in range(len(binary)):
        ret += 2 ** i * int(binary[-i])
    return ret


class ExtendableList(list):
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if self.__len__() < key.start + 1:
                self.extend([0] * (key.start + 1 - self.__len__()))
            if self.__len__() < key.stop + 1:
                self.extend([0] * (key.stop - self.__len__()))
        else:
            while self.__len__() < key + 1:
                self.extend([0] * (key + 1 - self.__len__()))
        super().__setitem__(key, value)

    def __getitem__(self, item):
        if item >= self.__len__():
            return 0
        return super().__getitem__(item)

class ManufacturingTask:
    id = 1

    def __init__(self, resource_list: np.ndarray, costs_list: np.ndarray):
        if len(resource_list) != len(costs_list):
            raise Exception('Resources list and costs list must be the same length')
        self.resource_list = resource_list
        self.costs_list = costs_list
        self.total_cost = np.sum(costs_list)
        self.total_steps = len(self.costs_list)
        self.is_finished = False

        self.step = 0
        self.last_step_finished = 0
        self.next_idle_time = 0
        self.id = ManufacturingTask.id
        ManufacturingTask.id += 1

    def validate_args(self, res: int, t: int):
        if t < self.next_idle_time:
            raise Exception(
                f'Last step is not done yet. Passed time is {t}, and current process ends at time={self.next_idle_time}')

        if res != self.resource_list[self.step]:
            raise Exception(
                f'Wrong type of resource. Passed resource is {res} and next needed resource is {self.resource_list[self.step]}')

    def get_next_resource(self, res: int, t: int):
        self.validate_args(res, t)

        return self.resource_list[self.step]

    def get_value_state(self, res: int, t: int):
        self.validate_args(res, t)

        return np.array(
            [self.total_cost, self.total_steps - self.step, self.costs_list[self.step], t - self.next_idle_time,
             np.count_nonzero(self.resource_list[self.step:] == res)])

    def execute_next_step(self, res: int, t: int):
        self.validate_args(res, t)

        self.next_idle_time = t + self.costs_list[self.step]
        self.step += 1
        if self.step >= self.total_steps:
            self.is_finished = True
        return self.costs_list[self.step - 1]

    def is_ready(self, res: int, t: int) -> bool:
        if self.is_finished:
            return False
        return self.next_idle_time <= t and self.resource_list[self.step] == res


def priority_function(x: np.ndarray, coef: np.ndarray) -> int:
    return x @ coef.T


def create_manufacturing_queues(tasks: list, coef: np.ndarray) -> List[ExtendableList]:
    if len(tasks) == 0:
        raise Exception('There are no tasks')

    resources_needed = np.unique(np.concatenate([task.resource_list for task in tasks]))
    resources_queue_list = [ExtendableList() for _ in range(resources_needed[-1] + 1)]
    tasks_left = copy.deepcopy(tasks)
    t = 0
    while len(tasks_left):
        for resource in resources_needed:
            if len(resources_queue_list[resource]) > t:
                continue

            tasks_ready = [task for task in tasks_left if task.is_ready(resource, t)]
            if not len(tasks_ready):
                continue

            priorities = [priority_function(task.get_value_state(resource, t), coef) for task in tasks_ready]
            chosen_task: ManufacturingTask = tasks_ready[np.argmax(priorities)]
            execution_time = chosen_task.execute_next_step(resource, t)
            resources_queue_list[resource][t:t + execution_time] = [chosen_task.id for _ in range(0, execution_time)]
            if chosen_task.is_finished:
                tasks_left.remove(chosen_task)
        t += 1

    return resources_queue_list


def get_cost(coefficients: np.ndarray, tasks: list) -> int:
    return np.max([len(q) for q in create_manufacturing_queues(tasks, coefficients)])


def generate_genotype(n_features: int, feature_size: int = 32) -> str:
    return ''.join([np.random.choice(['0', '1']) for _ in range(n_features * feature_size)])


class UnitFloat:
    def __init__(self, genotype: str = None, n_features: int = 5, feature_size: int = 16,
                 cost_function: Callable[[np.ndarray, list], list] = get_cost):
        self.genotype = genotype
        self.feature_size = feature_size
        self._fitness = None
        self.cost_function = cost_function
        self.n_features = n_features
        if genotype is None:
            self.genotype = generate_genotype(n_features, feature_size)

    def fitness(self, tasks):
        if self._fitness is None:
            self._fitness = self.cost_function(self.phenotype, tasks)
        return self._fitness

    @property
    def phenotype(self) -> np.ndarray:
        return np.array([bin_to_dec(self.genotype[i * self.feature_size:i * self.feature_size + self.feature_size])
                         for i in range(self.n_features)], dtype=int)

    def cross(self, other, pivot: int = None):
        if len(self.genotype) != len(other.genotype):
            raise ValueError("Different genotype sizes")
        if pivot is None:
            pivot = int(uniform(1, len(self.genotype)))
        return (UnitFloat(self.genotype[:pivot] + other.genotype[pivot:], self.n_features, self.feature_size,
                          self.cost_function),
                UnitFloat(other.genotype[:pivot] + self.genotype[pivot:], self.n_features, self.feature_size,
                          self.cost_function))

    def mutate(self, pivot: int = None):
        if pivot is None:
            pivot = int(uniform(0, len(self.genotype)))
        self.genotype = inv_chr(self.genotype, pivot)
        return self

    def __str__(self):
        return str(self.phenotype)

    def __repr__(self):
        return str(self.phenotype)


def generate_gantt(tasks_lists: List[ExtendableList]) -> None:
    df, colors, tasks = [], [], []

    resource_idx = 0

    for resource_list in tasks_lists:
        if len(resource_list) == 0:
            continue
        resource_idx += 1
        idx = 0
        while idx < len(resource_list):
            while resource_list[idx] == 0 and idx < len(resource_list):
                idx += 1

            start_idx = idx
            while idx < len(resource_list) - 1 and resource_list[idx] == resource_list[idx + 1]:
                idx += 1
            idx += 1
            stop_inx = idx
            start = timedelta(seconds=start_idx)
            stop = timedelta(seconds=stop_inx)
            today = date.today()
            df.append({
                'Task': f'Resource {resource_idx}',
                'Start': f'{today} {start}',
                'Finish': f'{today} {stop}',
                'duration': f'{stop - start}',
                'Resource': f'Task-{resource_list[start_idx]}'
            })
            if resource_list[start_idx] not in tasks:
                tasks.append(resource_list[start_idx])
                colors.append(f'rgb({randint(0, 255)}, {randint(0, 255)}, {randint(0, 255)})')

    fig = create_gantt(
        df=df,
        index_col='Resource',
        colors=colors,
        group_tasks=True,
        show_colorbar=True,
        showgrid_x=True,
        showgrid_y=True,
        title='Genetic optimized production schedule',
    )

    fig.show()


def main():
    x = np.genfromtxt('GA_task.csv', delimiter=";", dtype=int)
    tasks = [ManufacturingTask(x[:, i], x[:, i + 1]) for i in range(0, x.shape[1], 2)]

    population = [UnitFloat() for _ in range(population_size)]
    last = time()
    for epoch in tqdm(range(epochs)):
        fitness = np.array(Parallel(n_jobs=6)(delayed(unit.fitness)(tasks) for unit in population))
        # fitness = -fitness
        fitness_probs = softmax(-fitness)

        new_population = []
        if epoch % (epochs // 10) == 0:
            # print(f'Epoch: {epoch}; Time since last update: {(time() - last):.2f}s')
            # print(f'Avg Fitness: {np.average(fitness)}')
            # print(f'Best Fitness: {min(fitness)}')
            # last = time()
            with open(f'epoch_{epoch}_coefs.csv', 'w') as f:
                best = np.argmin(fitness)
                data = {'coefs': str(population[best])}
                dump(data, f)

        for _ in range(population_size // 2):
            choices = np.random.choice(population, 2, p=fitness_probs, replace=False)
            parent1, parent2 = choices[0], choices[1]

            if random() < crossover_probability:
                child1, child2 = parent1.cross(parent2)
            else:
                child1, child2 = parent1, parent2

            if random() < mutation_probability:
                child1.mutate()
            if random() < mutation_probability:
                child2.mutate()
            new_population.append(child1)
            new_population.append(child2)

        population = new_population
        population = population[:population_size]

    fitness = np.array(Parallel(n_jobs=6)(delayed(unit.fitness)(tasks) for unit in population))
    best = np.argmin(fitness)
    print(f'Training finished; Time since last update: {time() - last}s')
    print(f'Best Fitness: {min(fitness)}')
    print(population[best].fitness(tasks))
    print(population[best].genotype)
    print(population[best].phenotype)
    final = create_manufacturing_queues(tasks, population[best].phenotype)
    generate_gantt(final)
    with open('GA_task_result.csv', 'w') as f:
        for q in final:
            for task in q:
                f.write(str(task) + ';')
            f.write('\n')

    with open(f'last_coefs.csv', 'w') as f:
        data = {'coefs': str(population[best])}
        dump(data, f)


if __name__ == '__main__':
    main()
