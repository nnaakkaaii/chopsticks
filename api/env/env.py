import gym
from gym import spaces

from ..models.game import Game, Player, MOVEMENTS
from ..models.state import NUM_STATES


class ChopsticksEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_STATES)
        self.observation_space = spaces.Discrete(len(MOVEMENTS))
        self.game = Game(Player('me'), Player('enemy'))

    def reset(self):
        self.game = Game(Player('me'), Player('enemy'))
        return self.game

    def step(self, movement):
        reward = 0
        info = {}

        continue_game = self.game.play_turn(movement)
        done = not continue_game
        if done:
            reward = 1 if self.game.who_won == 'me' else -1

        return reward, done, info

    def render(self, mode='human', close=False):
        self.game.display()
