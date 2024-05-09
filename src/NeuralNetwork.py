import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Neural network class comprised of CNN and DQN to approximate Q-values for reinforcement learning
class NeuralNetwork(nn.Module):
    def __init__(self, env, DQN_DIM1, DQN_DIM2, LearningRate):
        super().__init__()  # Inheriting from torch.nn.Module constructor

        # Getting the input and output shapes for the neural network layers
        self.input_shape = env.observation_space.shape
        self.output_shape = env.action_space.n

        # Defining the convolutional layers for CNN
        # Used for processing image data from the environment and determining features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Getting the convolutional layer output to determine the input size for the linear layers
        conv_layer_out_shape = self._get_conv_out(self.input_shape)

        # Defining the linear layers for DQN
        self.layers = nn.Sequential(
            self.conv_layers,
            torch.nn.Flatten(),
            torch.nn.Linear(conv_layer_out_shape, DQN_DIM1),
            torch.nn.ReLU(),
            torch.nn.Linear(DQN_DIM1, DQN_DIM2),
            torch.nn.ReLU(),
            torch.nn.Linear(DQN_DIM2, self.output_shape)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LearningRate)  # Definining the Optimizer for the network
        self.loss = nn.MSELoss()  # Defining the Loss function as Mean Squared Error

        # Setting the device to GPU if available, else to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    # Function to get the convolutional layer output shape
    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    # Function to forward pass through the layers of the Neural Network
    def forward(self, x):
        return self.layers(x)