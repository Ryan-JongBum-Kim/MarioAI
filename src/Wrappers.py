import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# Creating class to skip frames within the environment
# Significantly reduces computer memory usage on redundant frames that are similar to previously stored frames
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)  # Inheriting from the environment
        self.skip = skip  # Storing number of frames to skip
    
    # Overwriting environment step function in environment - applies skip frames amount
    def step(self, action):
        total_reward = 0.0  # Variable to store total reward across the skipped frames
        done = False  # Variable to store if the environment finishes in between skipped frames

        # Looping through the skiped frames amount and completing environment steps
        for _ in range(self.skip):

            # Completing step in the environment
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward  # Updating the total reward across the steps

            # Checking if the episode is done in between skipped steps
            if done:
                break
        
        # Returning the final state, reward, done, trunc, info after the skipped steps
        return next_state, total_reward, done, trunc, info
    
# Function to apply the wrappers to the Super Mario Bros gym environment
def apply_wrappers(env):
    env = SkipFrame(env, skip=4)  # Number of frames to apply one action to (i.e. frames to skip)
    env = ResizeObservation(env, shape=84)  # Resizing the environment frame to 84x84
    env = GrayScaleObservation(env)  # Applying grayscale to observation to reduce memory usage
    env = FrameStack(env, num_stack=4, lz4_compress=True)  # Stacks the frames to introduce velocity data
    return env  # Returning the wrapped environment