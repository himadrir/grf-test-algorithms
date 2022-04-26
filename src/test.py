"""
SB3 discrete-action algorithms test
"""
from sb3_contrib import ARS, QRDQN, TRPO
from stable_baselines3 import A2C, DQN, PPO

import gfootball.env as football_env

academy_scenarios = [
    'academy_empty_goal_close',
    'academy_empty_goal',
    'academy_run_to_score',
    'academy_run_to_score_with_keeper',
    'academy_pass_and_shoot_with_keeper',
    'academy_run_pass_and_shoot_with_keeper',
    'academy_3_vs_1_with_keeper',
    'academy_corner',
    'academy_counterattack_easy',
    'academy_counterattack_hard',
    'academy_single_goal_versus_lazy',
]

timesteps = 100000

for i_scen in range(4):

    tensorboard_log = ("../sb3_test_logs/" + academy_scenarios[i_scen])

    for i_runs in range(5):

        env = football_env.create_environment(
            env_name=academy_scenarios[i_scen], 
            representation='simple115v2',
            render=False)

        # a2c = A2C("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        # a2c.learn(total_timesteps=timesteps)

        # ars = ARS("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        # ars.learn(total_timesteps=timesteps)
        #
        # trpo = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        # trpo.learn(total_timesteps=timesteps)
        #
        # dqn = DQN("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        # dqn.learn(total_timesteps=timesteps)
        #
        ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        ppo.learn(total_timesteps=timesteps)

        # qrdqn = QRDQN("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        # qrdqn.learn(total_timesteps=timesteps)

        env.close()

# steps = 0
# ep_rew = 0
# env.render()
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rew, done, info = env.step(action)
#     steps += 1
#     ep_rew += rew
#     if done:
#         print("Episode Steps: {} -- Episode Reward: {}".format(steps, ep_rew))
#         steps = 0
#         ep_rew = 0
#         obs = env.reset()
    
