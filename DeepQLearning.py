import torch
from torch import nn
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import random
from collections import deque


class DQL:
    """
    Deep Q-Learning (DQL) agent for discrete action-space environments.

    Implements a Deep Q-Network with experience replay, target network, and 
    epsilon-greedy exploration.
    """        
    def __init__(
        self,
        env,
        hidden_layers=(64, 64),
        activation="ReLU",
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
    ) -> None:
        """
        Initialize the DQL agent.

        Parameters
        ----------
        env : gym.Env
            Environment for the agent.
        hidden_layers : tuple of int
            Sizes of hidden layers in the Q-network.
        activation : str
            Activation function for hidden layers ("ReLU" or "Tanh").
        learning_rate : float
            Learning rate for the optimizer.
        discount_factor : float
            Discount factor for future rewards (gamma).
        epsilon_start : float
            Initial exploration rate.
        epsilon_min : float
            Minimum exploration rate.
        epsilon_decay : float
            Decay factor for epsilon after each step.
        buffer_size : int
            Maximum size of the replay buffer.
        batch_size : int
            Mini-batch size for training.
        """
        if activation not in ["ReLU", "Tanh"]:
            raise ValueError("activation must be Tanh or ReLU")
        
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=buffer_size)
        
        self.activation = nn.ReLU if activation == "ReLU" else nn.Tanh
        
        layers = []
        in_features = self.state_size
        for h in hidden_layers:
            layers += [nn.Linear(in_features, h), self.activation()]
            in_features = h
        layers += [nn.Linear(in_features, self.action_size)]
        
        self.estimator = nn.Sequential(*layers)
        self.target_estimator = nn.Sequential(*layers)
        self.target_estimator.load_state_dict(self.estimator.state_dict())
        self.target_estimator.eval()
        
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.steps = 0
        
    def _select_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state of the environment.

        Returns
        -------
        int
            Action index.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.estimator(state)
        return np.argmax(q_values.cpu().numpy())

    def _update(self):
        """
        Update the Q-network using a mini-batch from the replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor(np.array([e[1] for e in batch]))
        rewards = torch.FloatTensor(np.array([e[2] for e in batch]))
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.FloatTensor(np.array([e[4] for e in batch]))

        current_q_values = self.estimator(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_estimator(next_states).max(1)[0]
            target_q = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, n_episodes=1000, target_update=1000, render=False):
        """
        Train the agent in the environment.

        Parameters
        ----------
        n_episodes : int
            Number of training episodes.
        target_update : int
            Steps interval to update the target network.
        render : bool
            Whether to render the environment during training.

        Returns
        -------
        list
            Total rewards per episode.
        """
        rewards = []
        for _ in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            truncated = False

            while (not done) and (not truncated):
                if render:
                    self.env.render()
                                        
                action = self._select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                self.memory.append((state, action, reward, next_state, done))
                self._update()  
                
                state = next_state
                total_reward += reward

                self.steps += 1
                if self.steps % target_update == 0:
                    self.target_estimator.load_state_dict(self.estimator.state_dict())

            rewards.append(total_reward)

        return rewards
    
    def test(self, env, n_episodes = 10):
        """
        Test the trained agent without updating weights.

        Parameters
        ----------
        env : gym.Env
            Environment to test the agent.
        n_episodes : int
            Number of test episodes.

        Returns
        -------
        list
            Total rewards per episode.
        """
        total_rewards = []
        
        for _ in range(n_episodes):
            state , _ = env.reset()
            state = torch.FloatTensor(state)
            sum_rewards = 0
            
            while True:
                q_values = self.estimator(state).detach().numpy()
                action = np.argmax(q_values)
                next_state, reward, done, truncated, _ = env.step(action)

                sum_rewards += reward
                state = torch.FloatTensor(next_state)
                
                env.render()
                
                if done or truncated:
                    break
            
            total_rewards.append(sum_rewards)
            
        env.close()
        return total_rewards
