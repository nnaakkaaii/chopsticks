import torch

from .policy import BasePolicy


class SelectAction:
    def __init__(self, policy: BasePolicy, device=torch.device('cpu')):
        self.steps_done = 0
        self.policy = policy
        self.device = device

    def __call__(self, game, state):
        self.steps_done += 1

        movements = game.list_possible_movements()
        select, score = self.policy(state, movements, self.steps_done)

        return movements[select], torch.tensor([[select]], device=self.device, dtype=torch.long), score
