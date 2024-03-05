import gym
import numpy as np
from ddpg import Agent
from utils import plot_learning_curve
from enviroment_csv import single_env

if __name__ == '__main__':
    #env = gym.make('Pendulum-v0')
    env = single_env()
    #agent = Agent(input_dims=env.observation_space.shape,
    #        n_actions=env.action_space.shape[0])
    agent = Agent(input_dims=env.observation_space,
            n_actions=env.action_space)
    n_games = 5000

    figure_file = 'plots/ddpg.png'

    best_score = env.reward_origin
    score_history = []
    load_checkpoint = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        nextstep_count = 1
        while not done:
            nextstep_count = nextstep_count + 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action,nextstep_count)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)