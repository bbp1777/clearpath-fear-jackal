# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 17:50:40 2026

@author: rodne
"""


import numpy as np
import math
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

            
        
class POMDPNDTCWrapper(gym.Wrapper):
    """
        POMDP non descript terminal condition wrapper
    """
    
    def __init__(self,env_name: str,args,seed=12345,warm_start=40):
        self.warm_start=warm_start
        self.max_step=375
        self.current_step=0
        self.num_steps=4
        if args==None:
            self.env_name = env_name
            self.env = self._create_env()
            self.seed=seed
        else:
            self.args=args
            self.channel,self.height,self.width=self.args.env_shape
            self.env_name=self.args.env_name
            self.seed=self.args.seed
            self.env = self._create_env()
        cv2.startWindowThread()
        cv2.namedWindow('Window')
        self._create_env()
        
    def _create_env(self):
        env = gym.make(self.env_name,continuous=False,lap_complete_percent=0.95)
        # env=RTimeLimit(env,400)
        self.env=env
    
        
    def apply_vision_cone(self,image, theta=1.5708, origin=(48,75),cone_angle=np.pi/5):
    
    
        h, w = image.shape[:2]
        x0, y0 = origin
    
        alpha = cone_angle / 2.0
    
        radius = int(math.hypot(w, h))
    
        # Boundary angles
        theta_left = theta - alpha
        theta_right = theta + alpha
    
        x1 = int(x0 + radius * math.cos(theta_left))
        y1 = int(y0 - radius * math.sin(theta_left))   # minus because image y-axis is downward
    
        x2 = int(x0 + radius * math.cos(theta_right))
        y2 = int(y0 - radius * math.sin(theta_right))
    
        mask = np.zeros((h, w), dtype=np.uint8)
    
    
        cone_polygon = np.array([
            [x0, y0],
            [x1, y1],
            [x2, y2]
        ], dtype=np.int32)
    
        cv2.fillPoly(mask, [cone_polygon], 255)
    
        if image.ndim == 3:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = cv2.bitwise_and(image, mask)
    
        return masked_image, mask
            
    def region_of_interest(self,image, origin=(48,72), height=10, width=10):
    
        h_img, w_img = image.shape[:2]
        x0, y0 = origin
        half_w = width / 2.0
    
        top_left     = (int(x0 - half_w), int(y0-height))
        top_right    = (int(x0 + half_w), int(y0-height))
        bottom_right = (int(x0 + half_w), int(y0 + height))
        bottom_left  = (int(x0 - half_w), int(y0 + height))
    
        square_polygon = np.array([
            top_left,
            bottom_left,
            bottom_right,
            top_right
        ], dtype=np.int32)
    
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mask, [square_polygon], 255)
    
        if image.ndim == 3:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = cv2.bitwise_and(image, mask)
    
        return masked_image, mask
    
    
    
    def has_green_pixels(self,image):
    
        if image is None or image.size == 0:
            return False
    
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # HSV range for green
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
    
        # Threshold
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
        # Check for any green pixel
        return np.any(green_mask > 0)
    
    
    def step(self,action):
        for i in range(self.num_steps):
            state,_,terminal,_,info=self.env.step(action)
            self.current_step=self.current_step+1
        truncated=False
        reward=0
        non_terminal_region,_=self.region_of_interest(state)
        non_terminal_flag=self.has_green_pixels(non_terminal_region)
        if non_terminal_flag:
            terminal=True
            print("its in the grass")

        
        state,_=self.apply_vision_cone(state)
        
        state = cv2.resize(state, (self.width, self.height), interpolation=cv2.INTER_AREA)
        state=np.reshape(state, (3,self.width,self.height))
        if self.current_step >= self.max_step:
            truncated=True
            
            
        if terminal or truncated:
            reward=(self.env.tile_visited_count/len(self.env.unwrapped.track))*100+(self.current_step*(-.001))
            #reward=((self.env.tile_visited_count/self.max_step)*1000)+(self.current_step*(-.1))

        return state,reward,terminal,truncated,info
        
    def reset(self):
        state=self.env.reset()
        self.current_step=0
        if self.env.unwrapped.continuous:
            action=[0,0,0]
        else:
            action=0
        for i in range(self.warm_start):    
            state,_,_,_,_=self.env.step(action)
        state,_=self.apply_vision_cone(state)
        state = cv2.resize(state, (self.width, self.height), interpolation=cv2.INTER_AREA)
        state=np.reshape(state, (3,self.width,self.height))

        return state
            









# # Local test for car racing with  POMDPNDTC
# env=POMDPNDTCWrapper(gym.make("CarRacing-v2",continuous=True),84,84)
# #first testing the reset and moving past the zoo,
# state=env.reset()
# plt.imshow(state)
# # test step to assure reward and terminal are working
# action=[0,1,0]
# for i in range(100):
#     state,reward,terminal,truncated,info=env.step(action)
#     if terminal == True:
#         break
# print(f"check if we reached grass cause terminal is {terminal} and the reward is {reward}")
# plt.imshow(state)








