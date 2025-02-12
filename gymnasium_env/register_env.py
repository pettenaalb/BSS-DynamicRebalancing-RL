from gymnasium.envs.registration import register

register(
    id="gymnasium_env/FullyDynamicEnv-v0",
    entry_point="gymnasium_env.envs:FullyDynamicEnv",
    kwargs={'data_path': '../data/'},  # Additional arguments for the environment
)

register(
    id="gymnasium_env/StaticEnv-v0",
    entry_point="gymnasium_env.envs:StaticEnv",
    kwargs={'data_path': '../data/'},
)
