import abc
import math
import random

import torch

from ..models.game import Movement


class BasePolicy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, state, movements: dict[int, Movement], steps_done: int = 0):
        pass


class EpsilonGreedy(BasePolicy):
    def __init__(self, net, eps_start=0.9, eps_end=0.05, eps_decay=200):
        self.net = net
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def __call__(self, state, movements: dict[int, Movement], steps_done: int = 0):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)

        score = 0
        keys = list(movements.keys())
        if sample > eps_threshold:
            with torch.no_grad():
                q = self.net(state)
                value, select = q[0, keys].max(0)
                select_idx = keys[select.item()]
                score = int(-math.log(1 / ((torch.clamp(value, -0.99, 0.99).item() + 1) / 2) - 1) * 600)
        else:
            select_idx = random.choice(keys)
        return select_idx, score


class Softmax(BasePolicy):
    def __init__(self, net, temperature=0.6):
        self.net = net
        self.temperature = temperature

    def __call__(self, state, movements: dict[int, Movement], steps_done: int = 0):
        with torch.no_grad():
            q = self.net(state)
            keys = list(movements.keys())
            log_prob = q[0, keys] / self.temperature
            select = torch.distributions.categorical.Categorical(logits=log_prob).sample()
            select_idx = keys[select.item()]
            value = q[0, select_idx]
            score = int(-math.log(1 / ((torch.clamp(value, -0.99, 0.99).item() + 1) / 2) - 1) * 600)
        return select_idx, score
