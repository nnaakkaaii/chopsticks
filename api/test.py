from pathlib import Path
from itertools import count

import gym
import torch

from .env.data import get_state_from_env
from .env.policy import Softmax
from .env.select_action import SelectAction
from .networks.mlp import MLP


def test(result_path: Path):
    device = torch.device("cpu")
    policy_net = MLP().to(device)
    env = gym.make("chopsticks-v0").unwrapped
    policy = Softmax(policy_net)
    select_action = SelectAction(policy, device=device)

    net_path = result_path / 'net_last.pth'
    state_dict = torch.load(net_path, map_location={'cuda:0': 'cpu'})
    policy_net.load_state_dict(state_dict)

    env.reset()
    state = get_state_from_env(env, device=device)
    env.game.display()

    for t in count():
        move, _, score = select_action(env.game, state)
        _, done, _ = env.step(move)
        env.game.display()

        if done:
            print(f'{t}手にて終了')
            print(f'{env.game.who_won}の勝利')
            break

        next_state = get_state_from_env(env, device=device)
        state = next_state


if __name__ == '__main__':
    test(Path('result'))

