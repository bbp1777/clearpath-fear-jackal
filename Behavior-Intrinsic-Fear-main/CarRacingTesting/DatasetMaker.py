# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:49:29 2025

@author: rodne
"""



import numpy as np 
from collections import Counter
import os

class DatasetKeeper():
    
    def __init__(self,look_back,dimension_shape,all_classes):
        self.dataset_observations=[]
        self.dataset_class=[]
        self.dataset_class_number=[]
        self.allstates=[]
        self.look_back=look_back
        self.dimension_shape=dimension_shape
        self.all_classes=all_classes
        
    def append_current(self,observation):
        self.allstates.append(observation)
                
    def formulate_observations(self):
        # JACKAL RGB-D INTEGRATION NOTE:
        # Do not remove look_back here. The temporal stack is still the first
        # axis in each sample. The minimal Jackal change is to stop hardcoding
        # (3, 3, H, W) and instead reshape to [self.look_back, channels, H, W],
        # where channels becomes 4 for RGB-D.
        
        obs_to_keep=self.allstates[-self.look_back:]
        return np.reshape(np.asarray(obs_to_keep),(self.look_back,4,self.dimension_shape[0], self.dimension_shape[1]))
        
    def add_dataset(self,class_type):
        class_numerical_enc=self.all_classes.index(class_type)
        observations=self.formulate_observations()
        self.dataset_observations.append(observations)
        self.dataset_class.append(class_type)
        self.dataset_class_number.append(class_numerical_enc)
        
        
    def del_latest_dataset(self):
        self.dataset_observations.pop()
        self.dataset_class.pop()
        self.dataset_class_number.pop()
        
        
    def current_IID(self):
        return Counter(self.dataset_class)
        
        
    def prep_and_export_dataset(self,location,environment_name):
        
        dataset_observations_export=np.asarray(self.dataset_observations)
        dataset_class_export=np.asarray(self.dataset_class)
        dataset_class_number_export=np.asarray(self.dataset_class_number)
        
        
        location=location+"\\"
        filename=location+environment_name+"_lookback_"+str(self.look_back)
        if not os.path.exists(filename):
                os.makedirs(filename)
        
        np.save(filename+"observations", dataset_observations_export)
        np.save(filename+"class", dataset_class_export)
        np.save(filename+"class_number", dataset_class_number_export)
        
        
        
        
        
        