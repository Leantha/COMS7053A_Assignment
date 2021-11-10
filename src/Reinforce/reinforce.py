import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])

import numpy as np
import gym
from nle import nethack
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import matplotlib.pyplot as plt
import random
import pickle

device = torch.device("cpu")

from minihack import MiniHackNavigation
from gym.envs import registration

class MiniHackNewTask(MiniHackNavigation):
    def init(self, args, kwargs,des_file):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        super().init(args, kwargs, des_file=des_file,)

registration.register(
    id="MiniHack-NewTask-v0",
    entry_point="path.to.file:MiniHackNewTask", # use the actual the path
)

ACTIONS = [
    nethack.CompassCardinalDirection.N,
    nethack.CompassCardinalDirection.E,
    nethack.CompassCardinalDirection.S,
    nethack.CompassCardinalDirection.W,
    nethack.Command.FIRE,
    nethack.Command.KICK,
    nethack.Command.OPEN,
    nethack.Command.PICKUP,
    nethack.Command.WIELD,
    nethack.Command.SEARCH,
    nethack.Command.ZAP
]

STATS_INDICES = {
    'x_coordinate': 0,
    'y_coordinate': 1,
    'score': 9,
    'health_points': 10,
    'health_points_max': 11,
    'hunger_level': 18,
}


def crop_glyphs(glyphs, x, y, size=7):
    x_max = 79
    y_max = 21

    x_start = x - size
    x_end = x + size

    if x_start < 0:
        x_end = x_end + (-1 * x_start)
        x_start = 0

    if x_end > x_max:
        x_start = x_start - (x_end - x_max)
        x_end = x_max

    y_start = y - size
    y_end = y + size

    if y_start < 0:
        y_end = y_end + (-1 * y_start)
        y_start = 0

    if y_end > y_max:
        y_start = y_start - (y_end - y_max)
        y_end = y_max

    y_range = np.arange(y_start, (y_end), 1)
    x_range = np.arange(x_start, (x_end), 1)
    window_glyphs = []
    for row in y_range:
        for col in x_range:
            window_glyphs.append(glyphs[row][col])
    # return crop.flatten()
    crop = np.asarray(window_glyphs)
    # crop = window_glyphs.copy()
    return crop


def transform_observation(observation):
    """Process the state into the model input shape
    of ([glyphs, stats], )"""
    observed_glyphs = observation['glyphs']

    stat_x_coord = observation['blstats'][STATS_INDICES['x_coordinate']]
    stat_y_coord = observation['blstats'][STATS_INDICES['y_coordinate']]
    stat_health = float(observation['blstats'][STATS_INDICES['health_points']]) - float(
        observation['blstats'][STATS_INDICES['health_points_max']] / 2)
    stat_hunger = observation['blstats'][STATS_INDICES['hunger_level']]

    observed_chars = observation['chars']
    cropped_chars = crop_glyphs(observed_chars, stat_x_coord, stat_y_coord)
    # chars_mean = np.mean(cropped_chars)
    # chars_std = np.std(cropped_chars)
    # print('MEAN:', chars_mean)
    # print('STD:', chars_std)
    # norm_chars = (cropped_chars - chars_mean)/chars_std
    chars_min = np.min(cropped_chars)
    chars_max = np.max(cropped_chars)
    chars_range = chars_max - chars_min
    norm_chars = (cropped_chars - chars_min) / chars_range
    return norm_chars


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h1_size=128, h2_size=64, a_size=2):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(s_size, h1_size),
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.ReLU(),
                                    nn.Linear(h1_size, h2_size),
                                    nn.Dropout(p=0.5, inplace=True),
                                    nn.ReLU(),
                                    nn.Linear(h2_size, a_size)).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return tnf.softmax(self.policy(x), dim=0)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    returns = sum([(gamma ** i) * reward for i, reward in enumerate(rewards)])
    return returns


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.seed(seed)

    scores = []
    losses = []
    for episode in range(number_episodes):

        rewards = []
        log_probs = []
        state = transform_observation(env.reset())
        done = False
        # for step in range(max_episode_length):
        while not done:
            action_probs = policy_model.forward(torch.from_numpy(state).float().to(device))
            # action_probs_copy = torch.empty_like(action_probs).copy_(action_probs)
            # action_probs_copy[9:17] = 0
            # action_probs_copy[19] = 0
            # action_probs_copy[0] = 0
            # print(action_probs_copy)
            action_sampler = torch.distributions.Categorical(action_probs)
            action = action_sampler.sample()
            # print('#########ACTION########: ', action.item())
            # print('#########Message#######:', state.shape)
            log_probs.append(action_sampler.log_prob(action))
            new_state, reward, done, info = env.step(action.item())
            if reward == 0:
                reward = -0.001
            else:
                reward = reward
            rewards.append(reward)
            state = transform_observation(new_state)
            # env.render()
            # if done:
            #     break

        #with torch.no_grad():

        scores.append(sum(rewards))
        returns = compute_returns(rewards, gamma)
        loss = -1 * np.sum(np.array(log_probs) * returns)
        losses.append(loss)
        policy_model.optimizer.zero_grad()
        #loss.backward()
        nn.utils.clip_grad_norm_(policy_model.parameters(), 5.0)
        policy_model.optimizer.step()

        window = 25
        if verbose and episode % window == 0 and episode != 0:
            print("Episode " + str(episode) + "/" + str(number_episodes) +
                  " Score: " + str(np.mean(scores[episode - window: episode])) +
                  ' Losses: ' + str(sum(losses[episode - window:episode])))


    policy = policy_model.parameters()
    return policy_model, scores, losses

def run_reinforce():
    env = gym.make("MiniHack-Quest-Hard-v0", actions=ACTIONS)  # actions = ACTIONS
    print("Run REINFORCE.")
    obs_space = transform_observation(env.reset())
    obs_space_size = obs_space.shape[0]
    policy_model = SimplePolicy(s_size=obs_space_size,
                                h1_size=80,
                                h2_size=40,
                                a_size=env.action_space.n)
    policy, scores, losses = reinforce(env=env,
                                       policy_model=policy_model,
                                       seed=42,
                                       learning_rate=1e-2,
                                       number_episodes=1000,
                                       max_episode_length=1500,
                                       gamma=1.0,
                                       verbose=True)
    with open('/home/clerence/Documents/MiniHack/reinforce1/state_msg4.pkl', 'wb') as f:
        pickle.dump([policy_model, scores, losses], f)

    moving_avg = moving_average(scores, 25)

    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(scores)
    plt.plot(moving_avg, '--')
    plt.legend(['Score', 'Moving Average (w=50)'])
    plt.title("REINFORCE learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig('/home/clerence/Documents/MiniHack/reinforce1/reinforcerl.png')


if __name__ == '__main__':
    run_reinforce()
    # mean, std, x_fill, y_upper, y_lower = investigate_variance_in_reinforce()
    # run_reinforce_with_naive_baseline(mean, std, x_fill, y_upper, y_lower)
