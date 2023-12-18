import numpy as np

from .game import Game, Hand

NUM_STATES = 5


def load_state_from_env(env):
    return load_state_from_game(env.game)


def load_state_from_game(game: Game):
    arr = np.zeros((1, 5))
    arr[0, 0] = float(game.current_my_turn())
    arr[0, 1] = game.me.hands[Hand.LEFT] / 4
    arr[0, 2] = game.me.hands[Hand.RIGHT] / 4
    arr[0, 3] = game.enemy.hands[Hand.LEFT] / 4
    arr[0, 4] = game.enemy.hands[Hand.RIGHT] / 4
    return arr
