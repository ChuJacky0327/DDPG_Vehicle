import gym
import numpy as np
from ddpg import Agent
from utils_all_data import plot_learning_curve
from enviroment_csv_all_data import single_env

if __name__ == '__main__':
    #env = gym.make('Pendulum-v0')
    env = single_env()
    #agent = Agent(input_dims=env.observation_space.shape,
    #        n_actions=env.action_space.shape[0])
    agent = Agent(input_dims=env.observation_space,
            n_actions=env.action_space)
    n_games = 1000

    figure_file = 'plots/ddpg.png'

    best_score = env.reward_origin
    score_history = []
    load_checkpoint = False
    best_Episode_score = 0
    best_data_rate = []
    best_Transmission_delay = []
    best_power_consumption = []

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        nextstep_count = 1
        data_rate_list = []
        Transmission_delay_list = []
        power_consumption_list = []
        while not done:
            nextstep_count = nextstep_count + 1
            action = agent.choose_action(observation)
            observation_, reward, done, info, all_data_rate,all_Transmission_delay, all_power_consumption = env.step(action,nextstep_count)
            score += reward
            data_rate_list.append(all_data_rate)
            Transmission_delay_list.append(all_Transmission_delay)
            power_consumption_list.append(all_power_consumption)
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        
        if score > best_Episode_score:
            best_Episode_score = score
            print("best_Episode_score:",best_Episode_score)
            best_data_rate = data_rate_list
            best_Transmission_delay = Transmission_delay_list
            best_power_consumption = power_consumption_list

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file, best_data_rate, best_Transmission_delay, best_power_consumption)