import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from environment import MultiStockEnv
from agent import DQNAgent
import argparse
import pickle

filepath='.csv'

def get_data(filepath):
    data = pd.read_csv(filepath)
    return data.values

df = get_data(filepath)

def get_scaler(env):

    states=[]
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
        state = next_state

        return info['cur_val']


if __name__ == '__main__':

    models_folder = 'rl_models'
    rewards_folder = 'rl_rewards'
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='train or test')
    args = parser.parse_args()

    data = get_data(filepath)
    n_timesteps, n_stocks = data.shape
    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    portfolio = []

    if args.mode == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load()

        env = MultiStockEnv(test_data, initial_investment)
        agent.epsilon = 0.01
        agent.load(f'{models_folder}/dqn.h5')

    for e in range(num_episodes):
        val = play_one_episode(agent, env, args.mode)
        print(f'episode: {e+1}/{num_episodes} \nValue: {val:.2f}')
        portfolio.append(val)

    if args.mode == 'train':
        agent.save(f'{models_folder}/dqn.h5')

        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio)


