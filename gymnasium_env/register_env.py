from gymnasium.envs.registration import register

register(
    id="gymnasium_env/BostonCity-v0",
    entry_point="gymnasium_env.envs:BostonCity",
    kwargs={'data_path': '../data/'},  # Additional arguments for the environment
)
