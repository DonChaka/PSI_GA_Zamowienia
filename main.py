import numpy as np
import struct


def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)


def bin_to_float(binary):
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


class ExtendableList(list):
    def __setitem__(self, key, value):
        print(type(key))
        if isinstance(key, slice):
            if self.__len__() < key.start + 1:
                self.extend([None] * (key.start + 1 - self.__len__()))
            if self.__len__() < key.stop + 1:
                self.extend([None] * (key.stop - self.__len__()))
        else:
            while self.__len__() < key + 1:
                self.extend([None] * (key + 1 - self.__len__()))
        super().__setitem__(key, value)


class Task:
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

    def validate_args(self, res: int, t: int):
        if t <= self.next_idle_time:
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

        return self.total_cost, self.total_steps - self.step, self.costs_list[self.step], t - self.next_idle_time

    def execute_next_step(self, res: int, t: int):
        self.validate_args(res, t)

        self.next_idle_time += self.costs_list[self.step]
        self.step += 1
        if self.step >= self.total_steps:
            self.is_finished = True
        return self.resource_list[self.step], self.costs_list[self.step]


def create_maufacturing_queues(tasks: list, alpha: float, beta: float, gamma: float, delta: float):
    if len(tasks) == 0:
        raise Exception('There are no tasks')

    resources_queue_list = []
    while True:
        tasks_left = [task for task in tasks if not task.is_finished]
        if not len(tasks_left):
            break

    return resources_queue_list
