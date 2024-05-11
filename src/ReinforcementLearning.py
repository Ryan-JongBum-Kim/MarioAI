import torch
import math
import random
import numpy as np
from NeuralNetwork import NeuralNetwork
from ReplayBuffer import ReplayBuffer

# Reinforcement Learning class to learn from the environment and create action weights
class ReinforcementLearning:
    def __init__(self, env, MemorySize, MemoryRetain, BatchSize, LearningRate, Gamma, EpsilonStart, EpsilonEnd, EpsilonDecay, NetworkUpdateIter, ReplayStartSize, DQN_DIM1, DQN_DIM2):
        # Storing the hyperparameters for Reinforcement Learning Agent
        self.gamma = Gamma
        self.epsilon_start = EpsilonStart
        self.epsilon_decay = EpsilonDecay
        self.epsilon_end = EpsilonEnd
        self.network_update_iter = NetworkUpdateIter
        self.replay_start_size = ReplayStartSize
        self.batch_size = BatchSize
        self.gamma = Gamma

        # Creating the policy and target neural networks and replay buffer
        self.memory = ReplayBuffer(env, MemorySize, MemoryRetain, BatchSize)
        self.policy_network = NeuralNetwork(env, DQN_DIM1, DQN_DIM2, LearningRate)  # Q
        self.target_network = NeuralNetwork(env, DQN_DIM1, DQN_DIM2, LearningRate)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())       # Initially set weights of Q to \hat{Q}
        self.learn_count = 0  # Number of learning iteration

    # Epsilon-greedy policy
    def choose_action(self, observation):
        # Only start decaying the epsilon once we start learning
        if self.memory.memory_count > self.replay_start_size:
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * self.learn_count / self.epsilon_decay)
        else:
            eps_threshold = 1.0

        # Rolling random value, if we roll lower than epsilon threshold, sample a random action
        if random.random() < eps_threshold:
            return np.random.choice(np.array(range(12)), p=[0.005, 0.2175, 0.1675, 0.1675, 0.1675, 0.025, 0.05, 0.05, 0.05, 0.05, 0.05, 0])  # Random action with set priors
        
        # Otherwise the policy network (Q) chooses an action with the highest estimated Q-value so far
        state = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.policy_network.device)
        self.policy_network.eval()

        with torch.no_grad():
            q_values = self.policy_network(state)  # Get Q-values from policy network

        return torch.argmax(q_values).item()  # Returning the largest estimated Q-value
    
    # Function to train/learn from experiences
    def learn(self):
        # Sampling a random batch of experiences and converting them to tensors
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(self.policy_network.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.policy_network.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.policy_network.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.policy_network.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.policy_network.device)
        batch_indices = torch.from_numpy(np.arange(self.batch_size, dtype=np.int64)).to(self.policy_network.device)

        self.policy_network.train(True)                 # Training the neural network
        q_values = self.policy_network(states)          # Getting predicted Q-values from neural network
        q_values = q_values[batch_indices, actions]     # Getting the Q-values for the sampled experience

        self.target_network.eval()
        with torch.no_grad():
            q_values_next = self.target_network(states_)  # Getting Q-values from target network

        q_values_next_max = torch.max(q_values_next, dim=1)[0]          # Getting max Q-values for next state
        q_target = rewards + self.gamma * q_values_next_max * dones     # Getting target Q-values

        loss = self.policy_network.loss(q_values, q_target)  # Calcualting the loss from target and pred Q-values

        # Computing the gradients and updating Q weights
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()  # Updating Q weights
        self.learn_count += 1  # Incrementing learning count

        # Set target network weights to policy network weights every set increment of learning steps
        if self.learn_count % self.network_update_iter == self.network_update_iter - 1:
            print("Updating target network")
            self.update_target_network()

        return loss.item()  # Returning the loss of this learning stage

    # Function to synchronize the weights of the target network with the policy network
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    # Function to return the exploration rate (epsilon) of the agent
    def returning_epsilon(self):
        return self.exploration_rate