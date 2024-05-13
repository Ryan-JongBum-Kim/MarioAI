import torch
import numpy as np

# Replay Buffer class for storing and retrieving sampled experiences
class ReplayBuffer:
    def __init__(self, env, MemorySize, MemoryRetain, BatchSize):
        # Initialising memory buffer parameters
        self.memory_count = 0               # Number of experiences stored in memory
        self.memory_size = MemorySize       # Max size of memory to store at once
        self.memory_retain = MemoryRetain   # Size of initial memory to retain
        self.batch_size = BatchSize         # Number of experiences to sample from memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialising arrays to store the experiences
        self.states = np.zeros((MemorySize, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MemorySize, dtype=np.int64)
        self.rewards = np.zeros(MemorySize, dtype=np.float32)
        self.states_ = np.zeros((MemorySize, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MemorySize, dtype=np.bool)

    # Function to add experiences to the memory buffer
    def add(self, state, action, reward, state_, done):
        # Checking how much memory has been stored to determine next memory index to store at
        if self.memory_count < self.memory_size:
            mem_index = self.memory_count  # Use memory_count if not at max memory size
        else:
            # Replacing old memory that has been stored (Retaining initial 10% to avoid catastrophic forgetting)
            mem_index = int(self.memory_count % ((1-self.memory_retain) * self.memory_size) + (self.memory_retain * self.memory_size))

        # Adding the states to the replay buffer memory
        self.states[mem_index]  = state     # Storing the state
        self.actions[mem_index] = action    # Storing the action
        self.rewards[mem_index] = reward    # Storing the reward
        self.states_[mem_index] = state_    # Storing the next state
        self.dones[mem_index] =  1 - done   # Storing the done flag
        self.memory_count += 1  # Incrementing memory count

    # Function to sample experiences from the memory buffer
    def sample(self):
        # Getting a series of memory indices from the memory
        MEM_MAX = min(self.memory_count, self.memory_size)
        batch_indices = np.random.choice(MEM_MAX, self.batch_size, replace=True)
        
        # Getting the sampled experiences specified by the batch indices
        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        # Returning the random sampled experiences
        return states, actions, rewards, states_, dones
    
    # Function to get the memory count
    def __len__(self):
        return self.memory_count