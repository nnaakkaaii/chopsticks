from gym.envs.registration import register

register(
    id='chopsticks-v0',
    entry_point='api.env.env:ChopsticksEnv'
)
