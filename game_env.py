import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import game as R
import numpy as np

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='robot-v0',                                # call it whatever you want
    entry_point='game_env:RobotGameEnv', # module_name:class_name
)

class RobotGameEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, rows=10, columns=10, render_mode=None):
        self.truncated = 0
        self.rows=rows
        self.columns=columns
        self.render_mode = render_mode

        self.robot = R.RobotGame(rows=rows, columns=columns)
 
        self.action_space = spaces.Discrete(len(R.actions))

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(10,10,1),
            dtype=np.float32
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.
        
        if seed is not None:
            np.random.seed(seed)
        # Reset the WarehouseRobot. Optionally, pass in seed control randomness and reproduce scenarios.
        self.robot.reset(seed=seed)
        self.truncated = 0
        obs = self.robot._take_picture()
        # Additional info to return. For debugging or whatever.
        info = {}

        

        # Return observation and info
        return obs, info
    
    def step(self, action):
        # Perform action
        target_reached = self.robot.action(R.actions(action))
        self.truncated += 1
        # Determine reward and termination
        reward=0
        terminated=False
        truncated = False
        if target_reached==1:
            reward=10
            terminated=True
        elif target_reached==2:
            reward=-10
            terminated=True
        else:
            reward=-0.1
        
        
        obs = self.robot._take_picture()

        # Additional info to return. For debugging or whatever.
        info = {}

        if self.truncated >= 40:
            reward -= 10
            self.truncated = 0
            truncated = True

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, truncated, info

    # Gym required function to render environment
    def render(self):
        self.robot.render()

if __name__=="__main__":
    env = gym.make('robot-v0', render_mode='human')

    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")
    
    obs = env.reset()

    terminated = False
    # Take some random actions
    while(terminated == False):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()
            terminated = False