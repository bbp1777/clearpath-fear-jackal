# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:57:26 2025

@author: rodne
"""



import matplotlib.pyplot as plt
from DatasetMaker import DatasetKeeper
# from gym_handler import GymEnvManager
import numpy as np
# import gym_miniworld
import os
import cv2

from dataclasses import dataclass, field
import tyro

import numpy as np
import math
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

            
        
@dataclass
class EnvArgs:
    env_name:str="CarRacing-v2"
    """environment ID"""
    seed:int= np.random.randint(100000, 999999)
    env_shape:tuple=(3,84,84)
    
        
        
        
class POMDPNDTCWrapper(gym.Wrapper):
    """
        POMDP non descript terminal condition wrapper
    """
    
    def __init__(self,env_name: str,args,warm_start=50):
        self.warm_start=warm_start
        self.max_step=1500
        self.current_step=0
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
        # state=np.reshape(state, (3,self.width,self.height))
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
        # state=np.reshape(state, (3,self.width,self.height))

        return state
            




class CarRacingDatasetMaker():
    
    def __init__(self,env_name,env_params,lookback,wrappers,all_classes,seed,):
        self.env=POMDPNDTCWrapper(env_name,env_params)
        self.lookback=lookback
        self.all_classes=all_classes

    
    def reset(self,initial=False):
        if initial == True:
            obs=self.env.reset()
            obs_shape=obs.shape
            self.dimension_shape=(obs_shape[0],obs_shape[1])
            self.dataset=DatasetKeeper(self.lookback,self.dimension_shape
                                                             ,self.all_classes)
            plt.imshow(obs)
            return obs
        
        else:
            obs=self.env.reset()
            plt.imshow(obs)
            plt.show()
            return obs
        
    def step(self,action):
        observation, reward, done,terminal,info =self.env.step(action)
        plt.imshow(observation)
        plt.show()
        self.current_observation=observation
        self.dataset.append_current(self.current_observation)

        
    
    def add_dataset(self,class_type):
        self.dataset.add_dataset(class_type)
        
    def delete_dataset(self,class_type):
        self.dataset.del_latest_dataset()
        
    def export(self,location,run_name):
        final_location=location+"\\"+run_name
        if not os.path.exists(final_location):
                os.makedirs(final_location)
        self.dataset.prep_and_export_dataset(final_location,
                                             self.environment_name)
        
    def reformatset(self,location,channel_change,shape_change,folder):
        (new_width, new_height)=shape_change
        images=np.load(location)
        n_images,lookback,channel,height,width=images.shape
        total_images=n_images*lookback
        reshaped_img=np.reshape(images,(total_images,height,width,channel))
        shape_out=""
        black_white_out=""
        if (channel_change!=None and shape_change!=None): 
            processed_images = np.array([
                cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                for image in reshaped_img
            ])
            processed_images= np.reshape(processed_images,(n_images,lookback,new_height,new_width))
            shape_out="final_shape"
            black_white_out="bw_shape"
        elif shape_change != None:
            processed_images = np.array([
                cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                for image in reshaped_img
            ])
            processed_images= np.reshape(processed_images,(n_images,lookback,3,new_height,new_width))
            shape_out="final_shape"

        elif channel_change!=None:
            processed_images = np.array([
             cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for image in reshaped_img
            ])
            processed_images= np.reshape(processed_images,(n_images,lookback,height,width))
            black_white_out="bw"
        
        
        final_name=shape_out+black_white_out
        np.save(location+final_name,processed_images)
        
        
if __name__ == "__main__":
    from gymnasium.wrappers import TimeLimit
    env_params=tyro.cli(EnvArgs)
    env_name="CarRacing-v2"
    lookback=3
    all_classes=["avoid","neutral","good"]
    seed= np.random.rand(5)
    location="Data"
    run_name="CarRacing_2_Class"
    
    wrappers=None#[lambda env: TimeLimit(env, max_episode_steps=500)]
    manager = CarRacingDatasetMaker(env_name,env_params,lookback,wrappers,all_classes,seed)
    
    obs=manager.reset(initial=True)
    
    
    
    
    
    obs=manager.reset(initial=False)
    
    
    
    
    
    
    
    #forward
    manager.step(3)
    
    
    
    #1: steer right
    manager.step(1)
    
    
    
    #2: steer left
    manager.step(2)

    manager.step(4)
    
    manager.add_dataset("neutral")

    # Close the environment
    manager.close()    
   
    
    location="Data\\CarRacing_2_Class\\CarRacing-v2_lookback_3observations.npy"
    folder="Data\\CarRacing_2_Class\\"
    channel_change=1
    shape_change=(40,40)
    manager.export(location, run_name)
    manager.reformatset(location, channel_change, shape_change, folder)
    
    
    channel_change=None

    shape_change=(40,40)
    manager.reformatset(location, channel_change, shape_change, folder)
            
            
            
        
        