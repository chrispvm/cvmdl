import gym


def obs_action_shape_from_gym_env(env):
    # I don't think this works for all Env types. I think I'll need to make a separate version for
    # Box action spaces or whatever, and other types.
    assert isinstance(env, gym.Env)
    obs_shape = list(env.observation_space.shape)
    action_shape = [env.action_space.n]
    return obs_shape, action_shape
