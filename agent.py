from replay_buffer import ReplayBuffer
import numpy as np
from tensorflow.keras import layers, models

class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return  np.argmax(act_values[0])

    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return

        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        target = rewards + self.gamma * np.amax(self.model.predict(next_states) , axis=1)
        target[done] = rewards[done]

        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target
        self.model.train_on_batch(states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):

    i = layers.Input(shape=(input_dim,))
    x = i
    for _ in range(n_hidden_layers):
        x = layers.Dense(hidden_dim, activation='relu')(x)
    o = layers.Dense(n_action)(x)

    model = models.Model(i,o)
    model.compile(loss='mse', optimizer='adam')
    return model
