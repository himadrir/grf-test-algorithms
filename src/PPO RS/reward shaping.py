import gfootball.env as football_env
from PPO import Agent
from utils import plot_learning_curve
import math
import numpy as np

academy_scenario = 'academy_run_to_score_with_keeper'
figure_file = 'train_videos_ppo/result.png'


def get_reward_shaped_rewards(observation__, reward_, n_steps_):
    old_timestep = 0
    old_pos = None
    timestep = n_steps_
    const = 0.003
    reward_shaped = 0
    obs = observation__
    coordinates_left_team = obs[0: 22]
    ball_pos = obs[88: 91]
    ball_ownership = obs[94:97]
    active_player_map = obs[97:108]
    player_coordinates = [(coordinates_left_team[0], coordinates_left_team[1]),
                          (coordinates_left_team[2], coordinates_left_team[3])]
    result = np.where(active_player_map == 1)
    # print('\ncoordinates_left_team: ', player_coordinates[result[0][0]])
    # print('\nball position        : ', ball_pos)
    # print('\nball ownership       : ', ball_ownership)
    # print('\nactive player map    : ', result[0][0])
    active_player = player_coordinates[result[0][0]]

    if old_pos is None:
        old_pos = active_player

    if ball_ownership[0] == 0:
        dist_to_ball = math.sqrt((ball_pos[0] - active_player[0]) ** 2 + (ball_pos[1] - active_player[1]) ** 2)
        reward_shaped = ((0.02 * const) / (dist_to_ball + const)) - 0.01
    else:
        if timestep - old_timestep > 0:
            old_timestep = timestep
            if active_player[0] - old_pos[0] == 0 and active_player[1] - old_pos[
                1] == 0:  # check if position changed?
                reward_shaped = -0.08 * (timestep - old_timestep)
                old_pos = active_player

            else:
                dist_to_goal = math.sqrt((active_player[0] - 1) ** 2 + (
                        active_player[1] - 0.044) ** 2)  # check distance to goal from an activated player with ball
                reward_shaped = ((0.08 * const) / (dist_to_goal + const)) - 0.02

    # print('\ndistance to ball     : ', dist_to_ball)
    # print('\ndistance to goal     : ', dist_to_goal)
    # print('\nreward               : ', reward_)

    return reward_shaped


if __name__ == '__main__':
    env = football_env.create_environment(
        env_name=academy_scenario,
        representation='simple115v2',
        render=False, logdir='..gfootball/src/train_videos_ppo/', write_full_episode_dumps=False, write_video=True,
        write_goal_dumps=True)
    N = 20
    batch_size = 10
    n_epochs = 4
    alpha = 0.0003
    # print(env.observation_space.shape)
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 100000

    figure_file = 'plots/result.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # print('\n', observation_, '\n')
            n_steps += 1
            rs_rewards = get_reward_shaped_rewards(observation_, reward, n_steps)
            tot_reward = rs_rewards + reward
            score += tot_reward
            agent.remember(observation, action, prob, val, tot_reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.6f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

# obs = env.reset()
#
# # print(obs)
#
# epoch = 10000
#
# state = []
# actions = []
# values = []
# rewards = []
# const = 0.0009
#
# for i in range(epoch):
#     obs, reward, done, info = env.step(env.action_space.sample())
#     print('\nBall Position   [x, y, z]: \n', obs[88:91])
#     print('\nPlayer Position    [x, y]: \n', obs[0:2])
#     # print('\nBall controlled by       : \n', obs[94:97])
#
#     x = obs[88]
#     y = obs[89]
#     z = obs[90]
#
#     x_player = obs[0]
#     y_player = obs[1]
#
#     dist_to_ball = math.sqrt((x - x_player) ** 2 + (y - y_player) ** 2)
#     dist_to_goal = math.sqrt((x - 1) ** 2 + (y - 0.044) ** 2)
#
#     print('\nDistance of active player from ball: \n', dist_to_ball)
#     print('\nDistance of active player to goal  : \n', dist_to_goal)
#
#     reward_shaped = ((2 * const) / (const + dist_to_goal)) - 1
#     print('\nReward Shaped reward :\n', reward_shaped)
#     if done:
#         env.reset()
#
# env.close()
# obs = env.reset()
# ppo = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
# ppo.learn(total_timesteps=timesteps)
# obs = env.reset()
# # print(obs.shape, '\n')
# # print(obs)
# steps = 0
# ep_rew = 0
# env.render()
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rew, done, info = env.step(action)
#     print(obs)
#     steps += 1
#     ep_rew += rew
#     if done:
#         print("Episode Steps: {} -- Episode Reward: {}".format(steps, ep_rew))
#         steps = 0
#         ep_rew = 0
#         obs = env.reset()
