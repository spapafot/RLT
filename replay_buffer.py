import numpy as np


class ReplayBuffer():
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buff = np.zeros([size, obs_dim])
        self.obs2_buff = np.zeros([size,obs_dim])
        self.acts_buff = np.zeros(size)
        self.rews_buff = np.zeros(size)
        self.done_buff = np.zeros(size)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buff[self.ptr] = obs
        self.obs2_buff[self.ptr] = next_obs
        self.acts_buff[self.ptr] = act
        self.rews_buff[self.ptr] = rew
        self.done_buff[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buff[idxs],
                    s2=self.obs2_buff[idxs],
                    a=self.acts_buff[idxs],
                    r=self.rews_buff[idxs],
                    d=self.done_buff[idxs])
