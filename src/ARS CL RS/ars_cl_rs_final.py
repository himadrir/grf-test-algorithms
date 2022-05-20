'''
ARS with training on curriculum defined by first few academy problems, and "easy" policy transfer
'''


import gfootball.env as football_env
from sb3_contrib import ARS, QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import tensorboard
import os
import gym
import numpy as np
import math
import matplotlib.pyplot as plt

TOTAL_STEPS = 0
count = 0
old_action = None


def get_shaped_rewards(observation, n_steps, action):
    global old_action

    if old_action is None:
        old_action = action
    timestep = n_steps
    const = 0.03
    player_coordinates = []
    reward_shaped = 0
    obs = observation
    coordinates_left_team = obs[0: 22]
    ball_pos = obs[88: 91]
    ball_ownership = obs[94:97]
    active_player_map = obs[97:108]
    for i in range(len(coordinates_left_team) - 1):
        x = coordinates_left_team[i]
        y = coordinates_left_team[i + 1]
        player_coordinates.append((x, y))
        i += 2
    result = np.where(active_player_map == 1)

    active_player = player_coordinates[result[0][0]]

    #if old_action == action:
    #  reward_shaped = -4
    #  old_action = action
    if ball_ownership[0] == 0:
        dist_to_ball = math.sqrt((ball_pos[0] - active_player[0]) ** 2 + (ball_pos[1] - active_player[1]) ** 2)
        reward_shaped = -((2 * const) / (dist_to_ball + const)) * 10
    else:
        dist_to_ball = math.sqrt((ball_pos[0] - active_player[0]) ** 2 + (ball_pos[1] - active_player[1]) ** 2)
        dist_to_goal = math.sqrt((active_player[0] + 1) ** 2 + (active_player[1] - 0) ** 2)
        dist_to_goal = 2 - dist_to_goal
        # print(dist_to_goal)

        if action == 0 or action == 18:  # idle action, penalise agent for it
            reward_shaped += -20

        if action == 11 or action == 10 or action == 9 and dist_to_goal < 0.7:
            reward_shaped += 10

        if dist_to_goal > 0.9:
            reward_shaped += -7 * dist_to_goal
        else:
            reward_shaped += 1.8 / dist_to_goal

        if dist_to_goal < 0.5:
            if action == 17:  # dribbling inside the box
                reward_shaped += -200
            if action == 12:  # shooting
                reward_shaped += 80

    return reward_shaped


class RSWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.env = env

    def step(self, action):
        self.steps += 1
        next_state, reward, done, info = self.env.step(action)
        shaped_reward = get_shaped_rewards(next_state, reward, self.steps, action, done) + (reward * 1000)
        return next_state, shaped_reward, done, info

    def reset(self):
        self.steps = 0
        self.env.reset()
        return


academy_scenarios = [
    'academy_empty_goal',
    'academy_run_to_score',
    'academy_run_to_score_with_keeper',
    'academy_empty_goal_close',
    'academy_pass_and_shoot_with_keeper',
    'academy_run_pass_and_shoot_with_keeper',
    'academy_3_vs_1_with_keeper',
    'academy_corner',
    'academy_counterattack_easy',
    'academy_counterattack_hard',
    'academy_single_goal_versus_lazy',
]

total_timesteps = 300000
tensorboard_log = "./runs/"
model_name = "ars_cl_rs"

for i_scen in range(1, 6):

    print("=== New Training Scenario: {} ===".format(academy_scenarios[i_scen]))

    env = football_env.create_environment(
        env_name=academy_scenarios[i_scen],
        representation='simple115v2',
        render=False, logdir='../easy_ars_{}/'.format(academy_scenarios[i_scen]), write_goal_dumps=True)

    env = RSWrapper(env)
    print(count)
    # Check if model exists and load/create
    if os.path.isfile("../{}.zip".format(model_name)):
        print("Using stored model [{}]!".format(model_name))
        model = ARS.load("../{}.zip".format(model_name), env=env)
        reset_num = True
    else:
        model = ARS("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=1e-3)
        reset_num = False
    model.learn(total_timesteps=total_timesteps, tb_log_name=academy_scenarios[i_scen], reset_num_timesteps=reset_num)

    model.save("../{}".format(model_name))

    env.close()
