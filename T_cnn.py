import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import game_env

class DQN(nn.Module):
    def __init__(self, input_shape, out_actions):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=10*2*2, out_features=out_actions)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.layer_stack(x)
        return x
    
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class ROBOT_DQN():
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 100000
    mini_batch_size = 64

    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['U','TR','R','BR','B','BL','L','TL']

    def train(self, episodes, render='h', is_slippery=False):
        env = gym.make('robot-v0', render_mode='human')
        num_states = 100
        num_actions = 8

        epsilon = 1
        min_epsilon = 0.0001
        epsilon_decay = 0.995
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(input_shape=env.observation_space.shape[0], out_actions=num_actions)
        target_dqn = DQN(input_shape=env.observation_space.shape[0], out_actions=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        for i in range(episodes):
            step_count = 0
            state,_ = env.reset()
            terminated = False
            truncated = False
            reward_add = 0
            while(not terminated and not truncated):
                if(render=='human'):
                    env.render()
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                reward_add += reward
                memory.append((state, action, new_state, reward, terminated))
                
                state = new_state
                step_count += 1

            rewards_per_episode[i] = reward_add
            reward_add = 0

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    step_count = 0
                    torch.save(policy_dqn.state_dict(), f"robot{i+1}_R{rewards_per_episode[i]}.pt")
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    
        env.close()
        torch.save(policy_dqn.state_dict(), "ROBOT.pt")

        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121)
        plt.plot(sum_rewards)
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.savefig('ROBOT.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state))
            target_q[0][action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state: int) -> torch.Tensor:
        input_tensor = torch.zeros(1, 1, 10, 10)
        r = state // 10
        c = state % 10
        input_tensor[0][0][r][c] = 128 / 255
        return input_tensor

    def test(self, episodes):
        env = gym.make('robot-v0', render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        policy_dqn = DQN(input_shape=3, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql_cnn.pt", weights_only=True))
        policy_dqn.eval()

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
        env.close()

    def print_dqn(self, dqn):
        for s in range(16):
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s))[0].tolist():
                q_values += "{:+.2f}".format(q) + ' '
            q_values = q_values.rstrip()
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s)).argmax()]
            print(f'{s:02},{best_action},[{q_values}]', end=' ')
            if (s + 1) % 4 == 0:
                print()

if __name__ == '__main__':
    R = ROBOT_DQN()
    is_slippery = False
    R.train(10000)
