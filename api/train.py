import os
from pathlib import Path
from itertools import count


import gym
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer, RMSprop

from .env.data import get_state_from_env
from .env.policy import EpsilonGreedy
from .env.replay_memory import ReplayMemory
from .env.select_action import SelectAction
from .env.transition import Transition
from .models.game import MOVEMENTS
from .networks.mlp import MLP

GAMMA = 0.7
OPTIMIZER_PER_EPISODES = 2
TARGET_UPDATE = 10
MAX_MOVES = 512


def optimize_model(memory: ReplayMemory,
                   policy_net: nn.Module,
                   target_net: nn.Module,
                   optimizer: Optimizer,
                   batch_size: int = 512,
                   device: torch.device = torch.device('cpu'),
                   ):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
        )
    non_final_next_states = torch.cat([
        s for s in batch.next_state if s is not None
        ])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_next_actions_list = []
    for next_actions in batch.next_actions:
        if next_actions is not None:
            next_actions = list(next_actions.keys())
            non_final_next_actions_list.append(next_actions + [next_actions[0]] * (len(MOVEMENTS) - len(next_actions)))
    non_final_next_actions = torch.tensor(non_final_next_actions_list,
                                          device=device,
                                          dtype=torch.long,
                                          )

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)

    target_q = target_net(non_final_next_states)
    next_state_values[non_final_mask] = target_q.gather(1, non_final_next_actions).max(1)[0].detach()

    expected_state_action_values = (-next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    print(f"loss = {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(result_dir: Path,
          batch_size: int = 512,
          num_episodes: int = 1000000,
          optimizer_per_episode: int = 100,
          memory_size: int = 10000,
          target_update: int = 10,
          ):
    os.makedirs(result_dir / 'train', exist_ok=True)
    net_path = result_dir / 'net_last.pth'

    memory = ReplayMemory(memory_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = MLP().to(device)
    target_net = MLP().to(device)
    optimizer = RMSprop(policy_net.parameters(), lr=1e-5)
    env = gym.make('chopsticks-v0').unwrapped
    policy = EpsilonGreedy(policy_net)
    select_action = SelectAction(policy, device=device)

    if net_path.is_file():
        state_dict = torch.load(net_path)
        policy_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)

    for episode in range(num_episodes):
        env.reset()
        state = get_state_from_env(env, device=device)

        for t in count():
            move, action, score = select_action(env.game, state)
            reward, done, _ = env.step(move)

            reward = torch.tensor([reward], device=device)

            if not done:
                next_state = get_state_from_env(env, device=device)
                next_actions = env.game.list_possible_movements()
            else:
                next_state = None
                next_actions = None

            memory.push(state, action, next_state, next_actions, reward)

            state = next_state

            if done:
                break

        if episode % optimizer_per_episode == optimizer_per_episode - 1:
            optimize_model(memory,
                           policy_net,
                           target_net,
                           optimizer,
                           batch_size,
                           device,
                           )

            if episode // optimizer_per_episode % target_update == 0:
                state_dict = policy_net.state_dict()
                target_net.load_state_dict(state_dict)
                torch.save(state_dict, net_path)

    env.close()


if __name__ == '__main__':
    train(Path('result'))
