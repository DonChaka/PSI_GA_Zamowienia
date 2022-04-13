import copy
from typing import Callable
import numpy as np
import struct
from random import random, uniform
from time import time

population_size = 100
crossover_probability = 0.8
mutation_probability = 0.35
epochs = 100


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


class ExtendableList(list):
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if self.__len__() < key.start + 1:
                self.extend([None] * (key.start + 1 - self.__len__()))
            if self.__len__() < key.stop + 1:
                self.extend([None] * (key.stop - self.__len__()))
        else:
            while self.__len__() < key + 1:
                self.extend([None] * (key + 1 - self.__len__()))
        super().__setitem__(key, value)


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
            [self.total_cost, self.total_steps - self.step, self.costs_list[self.step], t - self.next_idle_time, np.count_nonzero(self.resource_list[self.step:] == res)])

    def execute_next_step(self, res: int, t: int):
        self.validate_args(res, t)

        self.next_idle_time = t + self.costs_list[self.step]
        self.step += 1
        if self.step >= self.total_steps:
            self.is_finished = True
        return self.costs_list[self.step-1]

    def is_ready(self, res: int, t: int) -> bool:
        if self.is_finished:
            return False
        return self.next_idle_time <= t and self.resource_list[self.step] == res


def priority_function(x: np.ndarray, coef: np.ndarray) -> int:
    return x @ coef.T


def create_manufacturing_queues(tasks: list, coef: np.ndarray) -> list:
    if len(tasks) == 0:
        raise Exception('There are no tasks')

    resources_needed = np.unique(np.concatenate([task.resource_list for task in tasks]))
    resources_queue_list = [ExtendableList() for _ in range(resources_needed[-1]+1)]
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


def generate_genotype(n_features: int) -> str:
    return ''.join([np.random.choice(['0', '1']) for _ in range(n_features * 32)])


class UnitFloat:
    def __init__(self, genotype: str = None, n_features: int = 5,
                 cost_function: Callable[[np.ndarray, list], list] = get_cost):
        self.genotype = genotype
        if genotype is None:
            self.genotype = generate_genotype(n_features)

        self.n_features = n_features
        self.cost_function = cost_function
        self._fitness = None

    def fitness(self, tasks: list):
        if self._fitness is None:
            self._fitness = self.cost_function(self.phenotype, tasks)
        return self._fitness

    @property
    def phenotype(self) -> np.ndarray:
        return np.array([bin_to_float(self.genotype[i * 32:i * 32 + 32]) for i in range(self.n_features)], dtype=np.float32)

    def cross(self, other, pivot: int = None):
        if len(self.genotype) != len(self.genotype):
            raise ValueError("Different genotype sizes")
        if pivot is None:
            pivot = int(uniform(1, len(self.genotype) - 1))
        return UnitFloat(self.genotype[:pivot] + other.genotype[pivot:]), UnitFloat(
            other.genotype[:pivot] + self.genotype[pivot:])

    def mutate(self, pivot: int = None):
        if pivot is None:
            pivot = int(uniform(0, self.n_features * 32))
        self.genotype = inv_chr(self.genotype, pivot)
        return self

    def __str__(self):
        return str(self.phenotype)

    def __repr__(self):
        return str(self.phenotype)


def main():
    x = np.genfromtxt('GA_task.csv', delimiter=";", dtype=int)
    tasks = [ManufacturingTask(x[:, i], x[:, i + 1]) for i in range(0, x.shape[1], 2)]

    population = [UnitFloat() for _ in range(population_size)]
    last = time()
    for epoch in range(epochs):
        fitness = np.array([unit.fitness(tasks) for unit in population])
        fitness_sum = np.sum(fitness)
        temp_probs = np.array([fitness_sum / f for f in fitness], dtype=np.float64)
        temp_sum = np.sum(temp_probs)
        fitness_probs = temp_probs / temp_sum

        new_population = []
        if epoch % (epochs // 10) == 0:
            print(f'Epoch: {epoch}; Time since last update: {time() - last}s')
            print(f'Avg Fitness: {fitness_sum / population_size}')
            print(f'Best Fitness: {min(fitness)}')
            last = time()
            # print(population)

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

    # print(population)
    # population.sort(key=lambda x: x.fitness(tasks))
    fitness = np.array([unit.fitness(tasks) for unit in population])
    best = np.argmin(fitness)
    print(f'Epoch: {epoch}; Time since last update: {time() - last}s')
    print(f'Best Fitness: {min(fitness)}')
    print(population[best].fitness(tasks))
    print(population[best].genotype)
    print(population[best].phenotype)
    final = create_manufacturing_queues(tasks, population[best].phenotype)
    with open('GA_task_result.csv', 'w') as f:
        for q in final:
            for task in q:
                f.write(str(task) + ';')
            f.write('\n')


if __name__ == '__main__':
    main()
