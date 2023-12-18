import torch

from ..models.state import load_state_from_game, load_state_from_env


def get_state_from_env(env, device=torch.device('cpu')):
    features = load_state_from_env(env)
    state = torch.from_numpy(features[:1]).float().to(device)
    return state


def get_state_from_game(game, device=torch.device('cpu')):
    features = load_state_from_game(game)
    state = torch.from_numpy(features[:1]).float().to(device)
    return state
