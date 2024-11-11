import random
from enum import Enum
import pygame
import sys
from os import path
import numpy as np

class actions(Enum):
    UP = 0
    TOP_R = 1
    RIGHT = 2
    BOT_R = 3
    DOWN = 4
    BOT_L = 5
    LEFT = 6
    TOP_L = 7

class GRID(Enum):
    _FLOOR = 0
    ROBOT = 1
    OBSTACLES = 2
    GOAL = 3

    def __str__(self):
        return self.name[:1]
    
# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
BLOCK_SIZE = 20
SPEED = 20

class RobotGame:
    def __init__(self, rows=10, columns=10,fps=1):
        self.rows = rows
        self.columns = columns
        self.__init__pygame__()
        self.reset()

        self.fps = fps
        self.last_action=''
        

    def __init__pygame__(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()
        self.action_font = pygame.font.SysFont("arial", 30)

        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 50
        self.cell_width = 50
        self.cell_size = (self.cell_width, self.cell_height)        

        # Define game window size (width, height)
        self.window_size = (self.cell_width * self.columns, self.cell_height * self.rows + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load & resize sprites
        file_name = path.join(path.dirname(__file__), "sprites/bot_blue.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/package.png")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size) 

        file_name = path.join(path.dirname(__file__), "sprites/asd.jpg")
        img = pygame.image.load(file_name)
        self.barriers_img = pygame.transform.scale(img, self.cell_size)

    def reset(self, seed=None):
        self._process_events()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.robot_pos = [random.randint(0,self.rows-1),random.randint(0,self.columns-1)]
        self.goal_pos = [random.randint(0,self.rows-1),random.randint(0,self.columns-1)]
        while self.robot_pos == self.goal_pos:
            self.goal_pos = [random.randint(0,self.rows-1),random.randint(0,self.columns-1)]
        self.barriers_pos = []
        self.barriers_pos.clear()
        """
        num_barriers = random.randint(0, 10)
        for _ in range(num_barriers):
            current_barrier = [random.randint(0,self.rows-1),random.randint(0,self.columns-1)]
            while current_barrier in [self.barriers_pos,self.robot_pos,self.goal_pos]:
                current_barrier = [random.randint(0,self.rows-1),random.randint(0,self.columns-1)]
            self.barriers_pos.extend([current_barrier])"""

    def action(self, action:actions) -> bool:
        self._process_events()
        self.last_action = action
        if action == action.RIGHT:
            if self.robot_pos[1]<self.columns-1:
                self.robot_pos[1]+=1
            else:
                 return 2
        elif action == action.LEFT :
            if self.robot_pos[1]>0:
                self.robot_pos[1]-=1
            else:
                 return 2
        elif action == action.DOWN :
            if self.robot_pos[0]<self.rows-1:
                self.robot_pos[0]+=1
            else:
                return 2
        elif action == action.UP:
            if self.robot_pos[0]>0:
                self.robot_pos[0]-=1
            else:
                return 2
        elif action == action.TOP_L:
            if self.robot_pos[0]>0 and self.robot_pos[1]>0:
                self.robot_pos[1]-=1
                self.robot_pos[0]-=1
            else:
                return 2
        elif action == action.TOP_R:
            if self.robot_pos[0]>0 and self.robot_pos[1]<self.columns-1:
                self.robot_pos[1]+=1
                self.robot_pos[0]-=1
            else:
                return 2
        elif action == action.BOT_R:
            if self.robot_pos[0]<self.rows-1 and self.robot_pos[1]<self.columns-1:
                self.robot_pos[1]+=1
                self.robot_pos[0]+=1
            else:
                return 2
        elif action == action.BOT_L:
            if self.robot_pos[0]<self.rows-1 and self.robot_pos[1]>0:
                self.robot_pos[1]-=1
                self.robot_pos[0]+=1
            else:
                return 2

        if self.robot_pos == self.goal_pos:
            return 1
        elif (self.robot_pos in self.barriers_pos):
            return 2
        else:
            return 0
    
    def _take_picture(self):
        # Initialize a 2D matrix based on the grid environment
        matrix = []
        for r in range(self.rows):
            row = []
            for c in range(self.columns):
                if [r, c] == self.robot_pos:
                    row.append(255)  # For example, this could be set to 255
                elif [r, c] == self.goal_pos:
                    row.append(128)
                elif [r, c] in self.barriers_pos:
                    row.append(64)   # For example, this could be set to 128
                else:
                    row.append(0) # For example, this could be set to 0
            matrix.append(row)

        # Convert the list of lists to a numpy array
        matrix = np.array(matrix, dtype=np.float32)

        # Normalize to 0-1 range if needed for CNN input
        matrix = matrix / 255.0
        print(matrix.shape)
        # Reshape to add the grayscale channel
        matrix = matrix.reshape((self.rows, self.columns, 1))
        return matrix
    
    def render(self):
        self._process_events()
        self.window_surface.fill((255,255,255))

        # Print current state on console
        for r in range(self.rows):
            for c in range(self.columns):
                
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if([r,c] == self.goal_pos):
                    # Draw target
                    self.window_surface.blit(self.goal_img, pos)

                if([r,c] == self.robot_pos):
                    # Draw robot
                    self.window_surface.blit(self.robot_img, pos)
                
                if([r,c] in self.barriers_pos):
                    # Draw robot
                    self.window_surface.blit(self.barriers_img, pos)

        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        # Limit frames per second
        self.clock.tick(self.fps)

    def reward(self,x):
        self._process_events()
        if x ==1:
            reward = 10
            self.reset()
            return reward
        elif x==2:
            reward = -10
            self.reset()
            return reward
        
    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()


if __name__ == "__main__":
    robotGame = RobotGame()
    robotGame.render()

    for i in range(25):
        rand_action = random.choice(list(actions))
        print(rand_action)
        robotGame.reward(robotGame.action(rand_action))
        robotGame.render()
        