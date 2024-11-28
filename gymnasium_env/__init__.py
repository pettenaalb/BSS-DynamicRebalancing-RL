from gymnasium.envs.registration import register

register(
    id="gymnasium_env/BostonCity-v0",
    entry_point="gymnasium_env.envs:BostonCity",
)
